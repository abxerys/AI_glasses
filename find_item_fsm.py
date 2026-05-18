# find_item_fsm.py
# -*- coding: utf-8 -*-
"""
尋物狀態機 — 移植自 aiglass2 的 FindGrabFSM，整合至 aiglass3 架構。

主要差異：
  - 語音輸出改用 aiglass3 的 play_voice_text()，而非 Groq TTS
  - 意圖解析改用 aiglass3 已有的 qwen_extractor.extract_english_label()
  - 不再依賴 Groq，減少外部 API 依賴
  - 輸入相機幀由 aiglass3 的 ws_camera_esp 統一提供

狀態機五個狀態：
  WAITING_FOR_COMMAND  → 等待「找一下 xxx」語音指令
  SEARCHING_OBJECT     → 用 YOLO 在畫面中搜尋目標
  GUIDING_HEAD         → 目標找到，引導使用者轉頭對準
  GUIDING_HAND         → 目標置中，引導手部靠近
  GRAB_SUCCESS         → 已取得物品，短暫停留後重置
"""
from __future__ import annotations

import time
import threading
import concurrent.futures
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("find_item_fsm")


# ──────────────────────────────────────────────
# 狀態枚舉
# ──────────────────────────────────────────────
class FIState(Enum):
    WAITING_FOR_COMMAND = auto()
    SEARCHING_OBJECT    = auto()
    GUIDING_HEAD        = auto()
    GUIDING_HAND        = auto()
    GRAB_SUCCESS        = auto()


# ──────────────────────────────────────────────
# 偵測結果資料類別（與 aiglass2 Detection 相容）
# ──────────────────────────────────────────────
@dataclass
class Detection:
    label: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


@dataclass
class HandResult:
    """MediaPipe 手部偵測結果（歸一化座標）"""
    tip_x: float          # 食指指尖 x (0~1)
    tip_y: float          # 食指指尖 y (0~1)
    x1: float             # 手部 bounding box (歸一化)
    y1: float
    x2: float
    y2: float

    @property
    def bbox_norm(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


# ──────────────────────────────────────────────
# YOLO 物品偵測器
# ──────────────────────────────────────────────
class ItemDetector:
    """YOLOv8 輕量偵測器（懶載入）"""

    def __init__(self, weights: str = "yolov8n.pt",
                 device: str = "cpu", conf: float = 0.35) -> None:
        self.weights = weights
        self.device  = device
        self.conf    = conf
        self._model  = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            log.info("[ItemDetector] YOLOv8 loaded: %s", self.weights)
        return self._model

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        model = self._load()
        res = model.predict(frame_bgr,
                            device=self.device,
                            conf=self.conf,
                            verbose=False)[0]
        out: List[Detection] = []
        if res.boxes is None:
            return out
        names = res.names
        for b in res.boxes:
            cls  = int(b.cls[0].item())
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            out.append(Detection(
                label=names[cls],
                conf=float(b.conf[0].item()),
                x1=xyxy[0], y1=xyxy[1],
                x2=xyxy[2], y2=xyxy[3],
            ))
        return out


# ──────────────────────────────────────────────
# MediaPipe 手部偵測器
# ──────────────────────────────────────────────
class HandsDetector:
    """MediaPipe Hands 包裝（懶載入）"""

    def __init__(self, max_num_hands: int = 1) -> None:
        self._hands = None
        self._max   = max_num_hands

    def _load(self):
        if self._hands is None:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("[HandsDetector] MediaPipe Hands loaded")
        return self._hands

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        hands = self._load()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]
        xs = [l.x for l in lm]; ys = [l.y for l in lm]
        # 食指指尖 = landmark 8
        tip_x = lm[8].x; tip_y = lm[8].y
        return HandResult(
            tip_x=tip_x, tip_y=tip_y,
            x1=min(xs), y1=min(ys),
            x2=max(xs), y2=max(ys),
        )

    def close(self):
        if self._hands:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None


# ──────────────────────────────────────────────
# 狀態機設定
# ──────────────────────────────────────────────
@dataclass
class FindItemConfig:
    head_left_thresh: float        = 0.4
    head_right_thresh: float       = 0.6
    center_lo: float               = 0.35
    center_hi: float               = 0.65
    center_frames_required: int    = 10
    hand_tolerance: float          = 0.08
    grab_radius: float             = 0.10
    iou_threshold: float           = 0.30
    grab_success_hold_s: float     = 3.0
    head_prompt_interval_s: float  = 1.2
    hand_prompt_interval_s: float  = 0.6
    search_prompt_interval_s: float = 2.0
    object_memory_s: float         = 2.0


@dataclass
class _Timers:
    last_head_prompt:   float = 0.0
    last_hand_prompt:   float = 0.0
    last_search_prompt: float = 0.0
    grab_success_start: float = 0.0


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────
def _bbox_iou(a: Tuple[float, float, float, float],
              b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _draw_overlay(frame: np.ndarray,
                  state: FIState,
                  target_zh: Optional[str],
                  dets: List[Detection],
                  target_det: Optional[Detection],
                  hand: Optional[HandResult],
                  last_stt_text: str,
                  last_stt_time: float,
                  center_streak: int) -> np.ndarray:
    """在畫面上疊加物品框、手部框、狀態列（輕量版，不顯示中文避免字體問題）"""
    h, w = frame.shape[:2]

    # 物品框
    for d in dets:
        color = (0, 255, 255) if d is target_det else (100, 200, 100)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.putText(frame, f"{d.label} {d.conf:.2f}",
                    (d.x1, max(18, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 手部框
    if hand is not None:
        hx1 = int(hand.x1 * w); hy1 = int(hand.y1 * h)
        hx2 = int(hand.x2 * w); hy2 = int(hand.y2 * h)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
        tip = (int(hand.tip_x * w), int(hand.tip_y * h))
        cv2.circle(frame, tip, 6, (255, 0, 255), -1)

    # 中央參考框
    cv2.rectangle(frame,
                  (int(0.35 * w), int(0.35 * h)),
                  (int(0.65 * w), int(0.65 * h)),
                  (80, 80, 80), 1)

    # 狀態列
    banner = (f"FIND: {state.name} "
              f"target:{target_zh or '-'} "
              f"streak:{center_streak}")
    cv2.rectangle(frame, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(frame, banner, (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# 主狀態機
# ──────────────────────────────────────────────
class FindItemFSM:
    """
    尋物狀態機。
    使用方式：
      fsm = FindItemFSM(tts_fn=play_voice_text)
      fsm.set_target("手機", "cell phone")
      # 每幀：
      out_frame = fsm.step(bgr_frame)
    """

    def __init__(self,
                 cfg: Optional[FindItemConfig] = None,
                 tts_fn: Optional[Callable[[str], None]] = None) -> None:
        self.cfg     = cfg or FindItemConfig()
        self._tts    = tts_fn or (lambda s: print(f"[TTS] {s}", flush=True))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # 偵測器（懶載入）
        self._detector = ItemDetector()
        self._hands    = HandsDetector(max_num_hands=1)

        # 狀態
        self.state: FIState    = FIState.WAITING_FOR_COMMAND
        self.target_zh: Optional[str] = None
        self.target_en: Optional[str] = None
        self.center_streak: int  = 0
        self.t = _Timers()
        self.last_obj: Optional[Detection] = None
        self.last_obj_time: float          = 0.0
        self.last_stt_text: str            = ""
        self.last_stt_time: float          = 0.0

        # 交替幀：分離 YOLO 與 MediaPipe 減少 CPU 競爭
        self._frame_toggle: bool = True

        # 快取上一幀的結果
        self._last_dets: List[Detection]       = []
        self._last_hand: Optional[HandResult]  = None

        log.info("[FindItemFSM] 初始化完成")

    # ── 外部介面 ──────────────────────────────

    def set_target(self, zh: str, en: str) -> None:
        """設定尋找目標，立即開始搜尋。"""
        self.target_zh, self.target_en = zh, en
        self.center_streak = 0
        self.last_obj      = None
        self.t             = _Timers()
        self._goto(FIState.SEARCHING_OBJECT)
        log.info("[FindItemFSM] 目標設定：%s (%s)", zh, en)

    def confirm_found(self) -> None:
        """使用者語音確認「找到了」或「拿到了」。"""
        if self.state != FIState.WAITING_FOR_COMMAND:
            self._tts(f"好的，已確認取得{self.target_zh or '物品'}。")
            self._reset()

    def reset(self) -> None:
        """外部重置（切換模式時呼叫）。"""
        self._reset()

    def is_idle(self) -> bool:
        return self.state == FIState.WAITING_FOR_COMMAND

    # ── 每幀處理（主入口）────────────────────

    def step(self, bgr: np.ndarray) -> np.ndarray:
        """
        接收一幀 BGR 影像，執行偵測 + 狀態更新，回傳疊加標注後的影像。
        此函式在 asyncio 事件迴圈的相機幀回呼中同步呼叫，
        內部 IO（TTS）透過 self._tts 注入，避免 asyncio 問題。
        """
        if bgr is None or bgr.size == 0:
            return bgr

        w, h = bgr.shape[1], bgr.shape[0]
        frame = bgr.copy()

        # 若等待指令或成功狀態，直接回傳原始畫面
        if self.state == FIState.WAITING_FOR_COMMAND:
            return frame

        if self.state == FIState.GRAB_SUCCESS:
            if time.monotonic() - self.t.grab_success_start >= self.cfg.grab_success_hold_s:
                self._reset()
            return _draw_overlay(frame, self.state, self.target_zh,
                                 [], None, None,
                                 self.last_stt_text, self.last_stt_time,
                                 self.center_streak)

        # ── 交替幀偵測 ──
        run_hands = (self.state == FIState.GUIDING_HAND)

        if not run_hands:
            # 未進入手部引導：全力跑 YOLO
            dets = self._detector.infer(frame.copy())
            self._last_dets = dets
            hand = None
            self._last_hand = None
        else:
            if self._frame_toggle:
                # 這幀只跑 YOLO，手部沿用上次
                dets = self._detector.infer(frame.copy())
                self._last_dets = dets
                hand = self._last_hand
            else:
                # 這幀只跑 MediaPipe，物品沿用上次
                dets = self._last_dets
                hand = self._hands.detect(np.ascontiguousarray(frame))
                self._last_hand = hand
            self._frame_toggle = not self._frame_toggle

        # ── 狀態更新 ──
        now   = time.monotonic()
        obj   = self._resolve_obj(dets, now)
        target_det = self._find_target(dets)

        if self.state == FIState.SEARCHING_OBJECT:
            self._tick_search(obj, now)
        elif self.state == FIState.GUIDING_HEAD:
            self._tick_head(obj, w, h, now)
        elif self.state == FIState.GUIDING_HAND:
            self._tick_hand(obj, hand, w, h, now)

        return _draw_overlay(frame, self.state, self.target_zh,
                             dets, target_det, hand,
                             self.last_stt_text, self.last_stt_time,
                             self.center_streak)

    # ── 狀態 tick ─────────────────────────────

    def _tick_search(self, obj: Optional[Detection], now: float) -> None:
        if obj is None:
            if now - self.t.last_search_prompt >= self.cfg.search_prompt_interval_s:
                self.t.last_search_prompt = now
                self._say(f"正在尋找{self.target_zh}，請慢慢轉動方向。")
            return
        self._say(f"已發現{self.target_zh}，正在引導方向。")
        self._goto(FIState.GUIDING_HEAD)

    def _tick_head(self, obj: Optional[Detection], w: int, h: int, now: float) -> None:
        if obj is None:
            self.center_streak = 0
            self._say(f"失去{self.target_zh}，重新搜尋。")
            self._goto(FIState.SEARCHING_OBJECT)
            return

        x_norm = obj.cx / float(w)
        y_norm = obj.cy / float(h)
        in_center = (self.cfg.center_lo <= x_norm <= self.cfg.center_hi
                     and self.cfg.center_lo <= y_norm <= self.cfg.center_hi)

        if in_center:
            self.center_streak += 1
        else:
            self.center_streak = 0

        if now - self.t.last_head_prompt >= self.cfg.head_prompt_interval_s:
            self.t.last_head_prompt = now
            if x_norm < self.cfg.head_left_thresh:
                self._say(f"{self.target_zh}在你的左邊，請向左轉。")
            elif x_norm > self.cfg.head_right_thresh:
                self._say(f"{self.target_zh}在你的右邊，請向右轉。")
            else:
                self._say(f"{self.target_zh}就在你正前方，伸手可及。")

        if self.center_streak >= self.cfg.center_frames_required:
            self._say("目標已置中，請伸手取物。")
            self._goto(FIState.GUIDING_HAND)

    def _tick_hand(self, obj: Optional[Detection],
                   hand: Optional[HandResult],
                   w: int, h: int, now: float) -> None:
        if obj is None:
            self._say(f"失去{self.target_zh}，退回引導方向。")
            self.center_streak = 0
            self._goto(FIState.GUIDING_HEAD)
            return
        if hand is None:
            if now - self.t.last_hand_prompt >= self.cfg.hand_prompt_interval_s:
                self.t.last_hand_prompt = now
                self._say("找不到你的手，請把手伸進畫面。")
            return

        ox = obj.cx / float(w)
        oy = obj.cy / float(h)
        hx, hy = hand.tip_x, hand.tip_y
        dx = ox - hx
        dy = oy - hy
        dist = (dx * dx + dy * dy) ** 0.5

        obj_bbox = (obj.x1 / w, obj.y1 / h, obj.x2 / w, obj.y2 / h)
        iou = _bbox_iou(obj_bbox, hand.bbox_norm)

        if dist < self.cfg.grab_radius or iou >= self.cfg.iou_threshold:
            if now - self.t.last_hand_prompt >= self.cfg.hand_prompt_interval_s:
                self.t.last_hand_prompt = now
                self._say("快到了，有沒有拿到？說「找到了」來結束。")
            return

        if now - self.t.last_hand_prompt < self.cfg.hand_prompt_interval_s:
            return
        self.t.last_hand_prompt = now

        tol  = self.cfg.hand_tolerance
        msgs: list[str] = []
        if dx >  tol: msgs.append("手向右移")
        elif dx < -tol: msgs.append("手向左移")
        if dy >  tol: msgs.append("手向下移")
        elif dy < -tol: msgs.append("手向上移")
        if not msgs:
            msgs.append("繼續靠近")
        self._say("，".join(msgs) + "。")

    # ── 工具 ─────────────────────────────────

    def _resolve_obj(self, dets: List[Detection], now: float) -> Optional[Detection]:
        """考慮短暫記憶（物品短暫消失不立刻失效）"""
        current = self._find_target(dets)
        if current is not None:
            self.last_obj      = current
            self.last_obj_time = now
            return current
        if (self.last_obj is not None
                and (now - self.last_obj_time) <= self.cfg.object_memory_s):
            return self.last_obj
        self.last_obj = None
        return None

    def _find_target(self, dets: List[Detection]) -> Optional[Detection]:
        if not self.target_en:
            return None
        cands = [d for d in dets if d.label.lower() == self.target_en.lower()]
        return max(cands, key=lambda d: d.conf) if cands else None

    def _goto(self, s: FIState) -> None:
        if s == self.state:
            return
        self.state = s
        if s == FIState.GRAB_SUCCESS:
            self.t.grab_success_start = time.monotonic()
        elif s == FIState.WAITING_FOR_COMMAND:
            log.info("[FindItemFSM] 已重置，等待下一個指令")

    def _reset(self) -> None:
        self.state         = FIState.WAITING_FOR_COMMAND
        self.target_zh     = None
        self.target_en     = None
        self.center_streak = 0
        self.last_obj      = None
        self._last_dets    = []
        self._last_hand    = None
        self.t             = _Timers()

    def _say(self, text: str) -> None:
        """非阻塞 TTS：丟到執行緒池，不卡相機幀迴圈"""
        if text:
            self._executor.submit(self._tts, text)

    def close(self) -> None:
        """釋放資源"""
        self._hands.close()
        self._executor.shutdown(wait=False)