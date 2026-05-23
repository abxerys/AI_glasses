# find_item_window.py
# -*- coding: utf-8 -*-
"""
方案 A：獨立尋物視窗 — 移植自 aiglass2/edge/find_grab_main.py
整合 aiglass3 的 TTS 與 ESP32 WebSocket 影像來源。

TTS 選項（--tts）：
  esp32  透過 /ws/find_item_control 中繼給 app_main.py，
         由 app_main.py 在自己 process 呼叫 play_voice_text()
         → stream_clients 有 ESP32 連線 → ESP32 喇叭出聲（預設）
  pc     直接播放到 PC 喇叭（pygame+gTTS 或 pyttsx3，測試用）
  local  呼叫 audio_player.play_voice_text()（僅當與 app_main.py 共用同一 process 時有效）
  print  只印 terminal

來源選項（--source）：
  webcam  本機 Webcam（預設）
  ws      接收 app_main.py /ws/viewer 廣播（ESP32 銅頭用此）

快捷鍵： q/ESC 離開  r 重置  c 確認  t 杯子
  1 水壺  2 手機  3 筆電  4 碗  5 書  6 鍵盤  7 滑鼠

啟動範例（ESP32 銅頭）：
  # 1. 先啟動 app_main.py（處理 ESP32 連線）
  python app_main.py

  # 2. 再啟動 find_item_window.py（預設即為 ESP32 模式）
  python find_item_window.py --source ws

  # 使用 PC 麥克風（選配）
  python find_item_window.py --source ws --mic

  # PC 喇叭測試模式（不需要 app_main.py）
  python find_item_window.py --tts pc
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ssl
ssl.create_default_context()

import faulthandler
faulthandler.enable()

import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

import cv2
cv2.setNumThreads(1)

import argparse
import asyncio
import json
import logging
import re
import sys
import threading
import time
import queue
from pathlib import Path
from typing import Callable, List, Optional
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("find_item_window")

# ════════════════════════════════════════════════════════
#  資料類別
# ════════════════════════════════════════════════════════
@dataclass
class Detection:
    label: str
    conf:  float
    x1: int; y1: int; x2: int; y2: int

    @property
    def cx(self) -> int: return (self.x1 + self.x2) // 2
    @property
    def cy(self) -> int: return (self.y1 + self.y2) // 2
    @property
    def w(self)  -> int: return self.x2 - self.x1
    @property
    def h(self)  -> int: return self.y2 - self.y1


@dataclass
class HandResult:
    tip_x: float; tip_y: float
    x1: float; y1: float; x2: float; y2: float

    @property
    def bbox_norm(self):
        return (self.x1, self.y1, self.x2, self.y2)


# ════════════════════════════════════════════════════════
#  YOLO 偵測器
# ════════════════════════════════════════════════════════
class YoloDetector:
    def __init__(self, weights="yolov8s.pt", device="cpu", conf=0.45):
        self.weights = weights
        self.device  = device
        self.conf    = conf
        self._model  = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            log.info("[YOLO] 模型載入完成：%s", self.weights)
        return self._model

    def infer(self, frame: np.ndarray) -> List[Detection]:
        res = self._load().predict(
            frame, device=self.device, conf=self.conf, verbose=False)[0]
        out: List[Detection] = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls = int(b.cls[0].item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            out.append(Detection(res.names[cls],
                                 float(b.conf[0].item()),
                                 x1, y1, x2, y2))
        return out


# ════════════════════════════════════════════════════════
#  MediaPipe 手部偵測器
# ════════════════════════════════════════════════════════
class HandsDetector:
    def __init__(self, max_num_hands=1):
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
            log.info("[MediaPipe] Hands 載入完成")
        return self._hands

    def detect(self, frame: np.ndarray) -> Optional[HandResult]:
        res = self._load().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        lm  = res.multi_hand_landmarks[0].landmark
        xs  = [l.x for l in lm]
        ys  = [l.y for l in lm]
        return HandResult(lm[8].x, lm[8].y,
                          min(xs), min(ys), max(xs), max(ys))

    def close(self):
        if self._hands:
            try:    self._hands.close()
            except: pass
            self._hands = None


# ════════════════════════════════════════════════════════
#  狀態機
# ════════════════════════════════════════════════════════
class FIState(Enum):
    WAITING_FOR_COMMAND = auto()
    SEARCHING_OBJECT    = auto()
    GUIDING_HEAD        = auto()
    GUIDING_HAND        = auto()
    GRAB_SUCCESS        = auto()


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0., ix2 - ix1)
    ih = max(0., iy2 - iy1)
    inter = iw * ih
    if inter <= 0.: return 0.
    ua    = max(0., ax2-ax1) * max(0., ay2-ay1)
    ub    = max(0., bx2-bx1) * max(0., by2-by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.


class FindItemFSM:
    HEAD_LEFT   = 0.40
    HEAD_RIGHT  = 0.60
    CENTER_LO   = 0.35
    CENTER_HI   = 0.65
    CENTER_REQ  = 5
    HAND_TOL    = 0.08
    GRAB_R      = 0.10
    IOU_THRESH  = 0.30
    SUCCESS_S   = 3.0
    HEAD_INT    = 1.2
    HAND_INT    = 0.6
    SEARCH_INT  = 2.0
    OBJ_MEM     = 4.0

    def __init__(self, tts_fn: Callable[[str], None]):
        import concurrent.futures
        self._tts  = tts_fn
        self._exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.state      = FIState.WAITING_FOR_COMMAND
        self.target_zh: Optional[str] = None
        self.target_en: Optional[str] = None
        self.app_mode  = "IDLE"
        self.streak     = 0
        self.last_obj:  Optional[Detection] = None
        self.last_obj_t = 0.
        self.last_stt   = ""
        self.last_stt_t = 0.
        self._t_head = self._t_hand = self._t_search = self._t_success = 0.

    def set_target(self, zh: str, en: str):
        self.app_mode = "FIND_ITEM"
        self.target_zh, self.target_en = zh, en
        self.streak = 0
        self.last_obj = None
        self._t_head = self._t_hand = self._t_search = 0.
        self._goto(FIState.SEARCHING_OBJECT)
        log.info("[FSM] 目標設定：%s (%s)", zh, en)

    def set_app_mode(self, mode: str):
        self.app_mode = mode or "IDLE"
        if self.app_mode != "FIND_ITEM":
            self._reset()
            self.app_mode = mode or "IDLE"
        log.info("[FSM] app mode = %s", self.app_mode)

    def confirm(self):
        if self.state != FIState.WAITING_FOR_COMMAND:
            self._say(f"好的，已確認取得{self.target_zh or '物品'}！")
            self._reset()

    def on_speech(self, text: str):
        self.last_stt   = text
        self.last_stt_t = time.monotonic()
        self._exec.submit(self._process_speech, text)

    def is_idle(self) -> bool:
        return self.state == FIState.WAITING_FOR_COMMAND

    def close(self):
        self._exec.shutdown(wait=False)

    def step(self, frame: np.ndarray,
             dets: List[Detection],
             hand: Optional[HandResult]) -> None:
        if frame is None: return
        h, w = frame.shape[:2]
        now  = time.monotonic()

        if self.state == FIState.WAITING_FOR_COMMAND:
            return
        if self.state == FIState.GRAB_SUCCESS:
            if now - self._t_success >= self.SUCCESS_S:
                self._reset()
            return

        cur = self._find_target(dets)
        if cur:
            self.last_obj = cur
            self.last_obj_t = now
            obj = cur
        elif self.last_obj and (now - self.last_obj_t) <= self.OBJ_MEM:
            obj = self.last_obj
        else:
            self.last_obj = None
            obj = None

        if   self.state == FIState.SEARCHING_OBJECT: self._tick_search(obj, now)
        elif self.state == FIState.GUIDING_HEAD:      self._tick_head(obj, w, h, now)
        elif self.state == FIState.GUIDING_HAND:      self._tick_hand(obj, hand, w, h, now)

    def _tick_search(self, obj, now):
        if obj is None:
            if now - self._t_search >= self.SEARCH_INT:
                self._t_search = now
                self._say(f"正在尋找{self.target_zh}，請慢慢轉動方向。")
            return
        self._say(f"已發現{self.target_zh}，正在引導方向。")
        self._goto(FIState.GUIDING_HEAD)

    def _tick_head(self, obj, w, h, now):
        if obj is None:
            self.streak = 0
            self._say(f"失去{self.target_zh}，重新搜尋。")
            self._goto(FIState.SEARCHING_OBJECT)
            return
        xn = obj.cx / w
        yn = obj.cy / h
        if (self.CENTER_LO <= xn <= self.CENTER_HI and
                self.CENTER_LO <= yn <= self.CENTER_HI):
            self.streak += 1
        else:
            self.streak = 0
        if now - self._t_head >= self.HEAD_INT:
            self._t_head = now
            if   xn < self.HEAD_LEFT:  self._say(f"{self.target_zh}在你的左邊，請向左轉。")
            elif xn > self.HEAD_RIGHT: self._say(f"{self.target_zh}在你的右邊，請向右轉。")
            else:                       self._say(f"{self.target_zh}就在你正前方，伸手可及。")
        if self.streak >= self.CENTER_REQ:
            self._say("目標已置中，請伸手取物。")
            self._goto(FIState.GUIDING_HAND)

    def _tick_hand(self, obj, hand, w, h, now):
        if obj is None:
            self._say(f"失去{self.target_zh}，退回引導方向。")
            self.streak = 0
            self._goto(FIState.GUIDING_HEAD)
            return
        if hand is None:
            if now - self._t_hand >= self.HAND_INT:
                self._t_hand = now
                self._say("找不到你的手，請把手伸進畫面。")
            return

        ox   = obj.cx / w;  oy = obj.cy / h
        dx   = ox - hand.tip_x
        dy   = oy - hand.tip_y
        dist = (dx*dx + dy*dy) ** 0.5
        iou  = _bbox_iou(
            (obj.x1/w, obj.y1/h, obj.x2/w, obj.y2/h),
            hand.bbox_norm
        )

        if dist < self.GRAB_R or iou >= self.IOU_THRESH:
            if now - self._t_hand >= self.HAND_INT:
                self._t_hand = now
                self._say("快到了，有沒有拿到？說「找到了」來結束。")
            return

        if now - self._t_hand < self.HAND_INT:
            return
        self._t_hand = now
        msgs = []
        if   dx >  self.HAND_TOL: msgs.append("手向右移")
        elif dx < -self.HAND_TOL: msgs.append("手向左移")
        if   dy >  self.HAND_TOL: msgs.append("手向下移")
        elif dy < -self.HAND_TOL: msgs.append("手向上移")
        self._say(("，".join(msgs) or "繼續靠近") + "。")

    def _find_target(self, dets: List[Detection]) -> Optional[Detection]:
        if not self.target_en: return None
        cands = [d for d in dets if d.label.lower() == self.target_en.lower()]
        return max(cands, key=lambda d: d.conf) if cands else None

    def _goto(self, s: FIState):
        if s == self.state: return
        self.state = s
        if s == FIState.GRAB_SUCCESS:
            self._t_success = time.monotonic()

    def _reset(self):
        self.state      = FIState.WAITING_FOR_COMMAND
        self.target_zh  = self.target_en = None
        self.streak     = 0
        self.last_obj   = None
        self._t_head = self._t_hand = self._t_search = self._t_success = 0.
        log.info("[FSM] 已重置，等待指令")

    def _say(self, text: str):
        if text:
            self._exec.submit(self._tts, text)

    def _process_speech(self, text: str):
        confirm_kw = ["拿到", "確認", "完成", "找到", "到了", "好了"]
        stop_kw    = ["結束", "停止", "不要", "關閉", "停", "取消"]

        if any(k in text for k in confirm_kw):
            if self.state != FIState.WAITING_FOR_COMMAND:
                self.confirm()
            return
        if any(k in text for k in stop_kw):
            if self.state != FIState.WAITING_FOR_COMMAND:
                self._reset()
            return

        m = re.search(r"找(?:一下|一個|個|下)?\s*(.{1,8}?)(?:$|[，。？！,!?])", text)
        query = m.group(1).strip() if m else text.strip()

        try:
            from qwen_extractor import extract_english_label
            en, source = extract_english_label(query)
            if source == "fallback" and en == "bottle" and "水" not in query and "瓶" not in query:
                log.warning("[FSM] qwen_extractor 無法解析：%s，跳過", text)
                return
            zh = query
            log.info("[FSM] qwen_extractor '%s' → en='%s' (source=%s)", query, en, source)
            self.set_target(zh, en)
        except Exception as e:
            log.warning("[FSM] qwen_extractor 失敗：%s", e)


# ════════════════════════════════════════════════════════
#  字體快取
# ════════════════════════════════════════════════════════
_FONT_CACHE: dict = {}

def _put_chinese(img: np.ndarray, text: str, pos,
                 color=(0, 255, 255), size=24) -> np.ndarray:
    if not text: return img
    pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("msjh.ttc", size)
        except IOError:
            _FONT_CACHE[size] = ImageFont.load_default()
    draw.text(pos, text, font=_FONT_CACHE[size], fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ════════════════════════════════════════════════════════
#  draw_overlay
# ════════════════════════════════════════════════════════
def draw_overlay(frame: np.ndarray, fsm: FindItemFSM,
                 dets: List[Detection],
                 target_det: Optional[Detection],
                 hand: Optional[HandResult]) -> np.ndarray:
    h, w = frame.shape[:2]

    for d in dets:
        color = (0, 255, 255) if d is target_det else (100, 200, 100)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.putText(frame, f"{d.label} {d.conf:.2f}",
                    (d.x1, max(18, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if hand:
        hx1 = int(hand.x1 * w); hy1 = int(hand.y1 * h)
        hx2 = int(hand.x2 * w); hy2 = int(hand.y2 * h)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
        tip = (int(hand.tip_x * w), int(hand.tip_y * h))
        cv2.circle(frame, tip, 6, (255, 0, 255), -1)
        cv2.putText(frame, "hand", (hx1, max(18, hy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame,
                  (int(.35 * w), int(.35 * h)),
                  (int(.65 * w), int(.65 * h)),
                  (80, 80, 80), 1)

    banner = (f"APP: {getattr(fsm, 'app_mode', 'IDLE')}  "
              f"STATE: {fsm.state.name}  "
              f"target: {fsm.target_zh or '-'}  "
              f"streak: {fsm.streak}")
    cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(frame, banner, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    now = time.monotonic()
    if fsm.last_stt and (now - fsm.last_stt_t) < 4.0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        frame = _put_chinese(frame, f"語音：{fsm.last_stt}",
                             (20, h - 40), color=(0, 255, 255), size=24)
    return frame


# ════════════════════════════════════════════════════════
#  共享幀
# ════════════════════════════════════════════════════════
_latest_frame:      Optional[np.ndarray] = None
_latest_frame_lock  = threading.Lock()
_stop_flag          = threading.Event()


def _set_frame(f: np.ndarray):
    global _latest_frame
    with _latest_frame_lock:
        _latest_frame = f


def _get_frame() -> Optional[np.ndarray]:
    with _latest_frame_lock:
        return None if _latest_frame is None else _latest_frame.copy()


# ════════════════════════════════════════════════════════
#  影像來源
# ════════════════════════════════════════════════════════
def _webcam_producer(index: int = 0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log.error("Webcam 開啟失敗 (index=%d)", index)
        _stop_flag.set()
        return
    log.info("[SRC] Webcam %d 已連線", index)
    try:
        while not _stop_flag.is_set():
            ok, frame = cap.read()
            if ok:  _set_frame(frame)
            else:   time.sleep(0.01)
    finally:
        cap.release()


def _ws_producer(url: str):
    async def _loop():
        import websockets
        while not _stop_flag.is_set():
            try:
                log.info("[SRC] 連線至 %s ...", url)
                async with websockets.connect(
                    url, max_size=8 * 1024 * 1024,
                    ping_interval=20, ping_timeout=20,
                ) as ws:
                    log.info("[SRC] WebSocket 連線成功：%s", url)
                    async for msg in ws:
                        if _stop_flag.is_set(): break
                        if not isinstance(msg, (bytes, bytearray)): continue
                        arr   = np.frombuffer(msg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            _set_frame(frame)
            except Exception as e:
                log.warning("[SRC] WebSocket 中斷：%s，3 秒後重試...", e)
                await asyncio.sleep(3)
    asyncio.run(_loop())


# ════════════════════════════════════════════════════════
#  ★ 中演控制 WebSocket（雙向）
#
#  架構說明：
#    app_main.py 與 find_item_window.py 為不同 OS process。
#    audio_stream.stream_clients 模組層變數在兩個 process 各自一份。
#    ESP32 連接的是 app_main.py 的 /stream.wav，
#    所以從 find_item_window.py 的 process 中呼叫 broadcast_pcm16_realtime()
#    永遠看到空的 stream_clients，ESP32 絕對發不出聲。
#
#  修復方案：透過已有的 /ws/find_item_control 連線（雙向改造）
#    find_item_window 將 TTS 文字傳給 app_main，
#    app_main 在自己的 process 呼叫 play_voice_text()，
#    stream_clients 有 ESP32 → ESP32 喇叭真正出聲。
# ════════════════════════════════════════════════════════

# 控制 WebSocket 連線的模組層共享狀態
# (find_item_window 與 app_main 之間的雙向通道)
_ctrl_ws_ref:  Optional[Any] = None   # websockets.WebSocketClientProtocol
_ctrl_ws_loop: Optional[asyncio.AbstractEventLoop] = None


def _control_ws_listener(url: str, fsm: "FindItemFSM"):
    """Listen for app_main mode/target commands AND send TTS relay requests."""
    global _ctrl_ws_ref, _ctrl_ws_loop

    if not url or url.strip().lower() in {"none", "off", "false", "0"}:
        log.info("[CTRL] control WebSocket disabled")
        return

    async def _loop():
        global _ctrl_ws_ref, _ctrl_ws_loop
        import websockets

        _ctrl_ws_loop = asyncio.get_event_loop()
        warned_unavailable = False

        while not _stop_flag.is_set():
            try:
                if not warned_unavailable:
                    log.info("[CTRL] connecting %s ...", url)
                async with websockets.connect(
                    url, max_size=1024 * 1024,
                    ping_interval=20, ping_timeout=20,
                ) as ws:
                    _ctrl_ws_ref = ws          # ★ 暴露連線給 TTS relay 函式
                    warned_unavailable = False
                    log.info("[CTRL] connected: %s", url)
                    await ws.send(json.dumps(
                        {"type": "hello", "client": "find_item_window"},
                        ensure_ascii=False))
                    async for msg in ws:
                        if _stop_flag.is_set():
                            break
                        if isinstance(msg, (bytes, bytearray)):
                            continue
                        try:
                            data = json.loads(msg)
                        except Exception:
                            log.warning("[CTRL] invalid message: %s", msg)
                            continue
                        typ = data.get("type")
                        if typ == "find_target":
                            zh = str(data.get("zh") or "").strip()
                            en = str(data.get("en") or "").strip()
                            if zh and en:
                                fsm.set_target(zh, en)
                        elif typ == "mode":
                            fsm.set_app_mode(str(data.get("mode") or "IDLE"))
            except Exception as e:
                if not warned_unavailable:
                    log.warning(
                        "[CTRL] control WebSocket unavailable: %s; "
                        "retrying in background", e)
                    warned_unavailable = True
                _ctrl_ws_ref = None
                await asyncio.sleep(10)

    asyncio.run(_loop())


# ════════════════════════════════════════════════════════
#  TTS 後端
# ════════════════════════════════════════════════════════

def _make_relay_tts_fn() -> Callable[[str], None]:
    """
    ★ ESP32 喇叭 TTS 中繼方案（預設）

    發送流程：
      FSM._say(text)
        → _relay_tts(text)
          → WS 傳給 app_main: {"type": "tts", "text": text}
            → app_main.play_voice_text(text)        ←「正確的 process」
              → broadcast_pcm16_realtime(pcm)
                → stream_clients 有 ESP32 連線 → ESP32 喇叭出聲 ✔

    如果 app_main 未連線（_ctrl_ws_ref is None），退回 print。
    Cooldown 由 app_main 的 audio_player._voice_cooldown(4s) 統一管理。
    """
    def _relay_tts(text: str):
        if _ctrl_ws_loop is None or _ctrl_ws_ref is None:
            print(f"[TTS-RELAY] app_main 未連線，退回 print: {text}", flush=True)
            return

        async def _send():
            try:
                await _ctrl_ws_ref.send(
                    json.dumps({"type": "tts", "text": text},
                               ensure_ascii=False))
            except Exception as e:
                log.warning("[TTS-RELAY] 傳送失敗: %s → print 退回", e)
                print(f"[TTS] {text}", flush=True)

        asyncio.run_coroutine_threadsafe(_send(), _ctrl_ws_loop)

    return _relay_tts


def _make_pc_tts_fn() -> Callable[[str], None]:
    """
    PC 喇叭 TTS（對應 --tts pc，不需要 ESP32 或 app_main.py）。
    優先 pygame+gTTS，退回 pyttsx3。
    """
    _PC_COOLDOWN = 4.0
    _lock        = threading.Lock()
    _last_text   = [""]
    _last_time   = [0.0]

    try:
        import pygame
        import pygame.mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        from gtts import gTTS
        import tempfile

        def _pygame_tts(text: str):
            now = time.time()
            with _lock:
                if text == _last_text[0] and now - _last_time[0] < _PC_COOLDOWN:
                    return
                _last_text[0] = text
                _last_time[0] = now

            def _play():
                try:
                    fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                    gTTS(text=text, lang="zh-tw").save(mp3_path)
                    sound = pygame.mixer.Sound(mp3_path)
                    ch = sound.play()
                    while ch and ch.get_busy():
                        time.sleep(0.05)
                    try: os.remove(mp3_path)
                    except: pass
                except Exception as e:
                    log.warning("[TTS-PC] 失敗: %s", e)

            threading.Thread(target=_play, daemon=True, name="pc_tts").start()

        log.info("[TTS-PC] 使用 pygame + gTTS")
        return _pygame_tts

    except ImportError:
        pass
    except Exception as e:
        log.warning("[TTS-PC] pygame 初始化失敗: %s", e)

    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 200)
        _engine_lock = threading.Lock()

        def _pyttsx3_tts(text: str):
            now = time.time()
            with _lock:
                if text == _last_text[0] and now - _last_time[0] < _PC_COOLDOWN:
                    return
                _last_text[0] = text
                _last_time[0] = now

            def _play():
                with _engine_lock:
                    try:
                        _engine.say(text)
                        _engine.runAndWait()
                    except Exception as e:
                        log.warning("[TTS-PC] pyttsx3 失敗: %s", e)

            threading.Thread(target=_play, daemon=True, name="pc_tts").start()

        log.info("[TTS-PC] 使用 pyttsx3（離線）")
        return _pyttsx3_tts

    except ImportError:
        pass

    log.warning("[TTS-PC] 沒有可用 TTS 後端，退回 print。"
                "若要有聲音：pip install pygame gtts。")
    return lambda t: print(f"[TTS] {t}", flush=True)


# ════════════════════════════════════════════════════════
#  麥克風 VAD
# ════════════════════════════════════════════════════════
class _MicListener:
    def __init__(self, on_text: Callable[[str], None],
                 model_size: str = "tiny",
                 sample_rate: int = 16000,
                 energy_thresh: int = 500,
                 silence_s: float = 0.8,
                 max_utt_s: float = 8.0):
        self.on_text       = on_text
        self.sample_rate   = sample_rate
        self.chunk         = int(sample_rate * 0.1)
        self.energy_thresh = energy_thresh
        self.silence_s     = silence_s
        self.max_utt_s     = max_utt_s
        self._model_size   = model_size
        self._stop         = threading.Event()
        self._thread       = threading.Thread(
            target=self._run, daemon=True, name="mic")

    def start(self): self._thread.start()
    def stop(self):  self._stop.set()

    def _load_stt(self):
        try:
            from faster_whisper import WhisperModel
            _model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
            log.info("[MIC] faster-whisper 載入成功: %s", self._model_size)

            class _W:
                def __init__(self, m): self._m = m
                def transcribe_pcm(self, pcm, sr):
                    audio = pcm.astype(np.float32) / 32768.0
                    segs, _ = self._m.transcribe(
                        audio, language="zh", beam_size=5,
                        vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": 300})
                    return "".join(s.text for s in segs).strip()

            return _W(_model)
        except ImportError:
            log.warning("[MIC] faster-whisper 未安裝，嘗試 openai-whisper...")

        try:
            import whisper as _w
            m = _w.load_model(self._model_size)
            log.info("[MIC] openai-whisper 載入成功: %s", self._model_size)

            class _OW:
                def __init__(self, model): self._m = model
                def transcribe_pcm(self, pcm, sr):
                    import whisper
                    return whisper.transcribe(
                        self._m, pcm.astype(np.float32)/32768.0,
                        language="zh")["text"]

            return _OW(m)
        except Exception as e:
            log.error("[MIC] Whisper 載入失敗: %s", e)
            return None

    def _run(self):
        try:
            import sounddevice as sd
        except ImportError:
            log.error("[MIC] 請安裝: pip install sounddevice")
            return

        stt = self._load_stt()
        if stt is None: return

        buf: list  = []
        speaking   = False
        last_t = 0.0; utt_start = 0.0
        log.info("[MIC] 開始監聽")

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                                 dtype="int16", blocksize=self.chunk) as stream:
                while not self._stop.is_set():
                    block, _ = stream.read(self.chunk)
                    pcm    = block[:, 0] if block.ndim == 2 else block
                    energy = int(np.abs(pcm).mean())
                    now    = time.monotonic()
                    if energy > self.energy_thresh:
                        if not speaking:
                            speaking = True; utt_start = now; buf.clear()
                        last_t = now; buf.append(pcm.copy())
                    elif speaking:
                        buf.append(pcm.copy())
                        if (now - last_t > self.silence_s or
                                now - utt_start > self.max_utt_s):
                            audio = np.concatenate(buf)
                            buf = []; speaking = False
                            if len(audio) > self.sample_rate * 0.3:
                                try:
                                    text = (stt.transcribe_pcm(
                                        audio, self.sample_rate) or "").strip()
                                    if text:
                                        log.info("[MIC] 聽到：%s", text)
                                        self.on_text(text)
                                except Exception as e:
                                    log.warning("[MIC] STT 失敗: %s", e)
        except Exception:
            log.exception("[MIC] 執行緒崩潰")


# ════════════════════════════════════════════════════════
#  注冊到 app_main
# ════════════════════════════════════════════════════════
def register_to_app_main(fsm: "FindItemFSM") -> bool:
    try:
        _am = sys.modules.get("app_main")
        if _am is None:
            log.info("[REGISTER] app_main 不在同一 process，使用 control WebSocket")
            return False
        _am.register_find_window_fsm(fsm)
        log.info("[REGISTER] FSM 已注冊到 app_main")
        return True
    except Exception as e:
        log.warning("[REGISTER] 注冊失敗: %s", e)
        return False


def unregister_from_app_main() -> None:
    try:
        _am = sys.modules.get("app_main")
        if _am is None: return
        _am.register_find_window_fsm(None)
    except Exception:
        pass


# ════════════════════════════════════════════════════════
#  主函式
# ════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="aiglass3 獨立尋物視窗")
    p.add_argument("--source",       choices=["webcam", "ws"], default="webcam")
    p.add_argument("--ws-url",       default="ws://127.0.0.1:8765/ws/viewer")
    p.add_argument("--control-url",  default="ws://127.0.0.1:8765/ws/find_item_control")
    p.add_argument("--webcam-index", type=int, default=0)
    p.add_argument("--yolo-weights", default="yolov8s.pt",
                   help="yolov8s.pt 對 cell phone 辨識準確； yolov8n.pt 速度較快")
    p.add_argument("--yolo-device",  default="cpu")
    p.add_argument("--yolo-conf",    type=float, default=0.45,
                   help="0.35 太低易把手機誤判為 suitcase，預設提高到 0.45")
    p.add_argument("--tts",
                   choices=["esp32", "pc", "local", "print"],
                   default="esp32",
                   help=(
                       "esp32  = 透過 WS 傳文字給 app_main 播放到 ESP32 喇叭（預設）；"
                       "pc     = PC 喇叭（pygame/pyttsx3）；"
                       "local  = 直接呼叫 audio_player（與 app_main 同 process 時才有效）；"
                       "print  = 只印 terminal"
                   ))
    p.add_argument("--mic",          action="store_true")
    p.add_argument("--stt-model",    default="tiny")
    p.add_argument("--log-level",    default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── TTS 後端 ─────────────────────────────────────────
    # ★ 修復說明：
    #   兩個 process 不共享 audio_stream.stream_clients。
    #   ESP32 連接的是 app_main.py 的 /stream.wav，
    #   必須將 TTS 文字傳給 app_main.py 確保從正確 process 呼叫 play_voice_text()。
    #   --tts esp32 即為此修復方案。
    if args.tts == "esp32":
        tts_fn = _make_relay_tts_fn()
        log.info("[TTS] ESP32 中繼模式：將文字傳給 app_main.py 播放到 ESP32 喇叭")
        log.info("[TTS] 請確認 app_main.py 已啟動，且 ESP32 已連接 /stream.wav")
    elif args.tts == "pc":
        tts_fn = _make_pc_tts_fn()
    elif args.tts == "local":
        try:
            from audio_player import play_voice_text, initialize_audio_system
            initialize_audio_system()
            tts_fn = play_voice_text
            log.warning("[TTS] local 模式：僅當與 app_main.py 共用同一 process 時才會出聲")
        except ImportError:
            log.warning("[TTS] 無法匯入 audio_player，改用 esp32 模式")
            tts_fn = _make_relay_tts_fn()
    else:
        tts_fn = lambda t: print(f"[TTS] {t}", flush=True)

    detector = YoloDetector(args.yolo_weights, args.yolo_device, args.yolo_conf)
    hands    = HandsDetector(max_num_hands=1)
    fsm      = FindItemFSM(tts_fn=tts_fn)

    register_to_app_main(fsm)

    t_ctrl = threading.Thread(
        target=_control_ws_listener,
        args=(args.control_url, fsm),
        daemon=True, name="find_item_ctrl")
    t_ctrl.start()

    if args.source == "webcam":
        t_src = threading.Thread(
            target=_webcam_producer,
            args=(args.webcam_index,),
            daemon=True, name="webcam_src")
    else:
        t_src = threading.Thread(
            target=_ws_producer,
            args=(args.ws_url,),
            daemon=True, name="ws_src")
    t_src.start()

    mic: Optional[_MicListener] = None
    if args.mic:
        mic = _MicListener(on_text=fsm.on_speech, model_size=args.stt_model)
        mic.start()
        log.info("[MIC] 麥克風已啟動")
    else:
        log.info("[TIP] 可按視窗內 1~7 快捷鍵設定目標")

    WIN = "Find & Grab — aiglass3  (q 離開)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1200, 720)

    log.info("[READY] 尋物視窗已就緒")
    tts_fn("尋物模式已啟動，請按鍵選擇目標。")

    frame_toggle = True
    last_dets: List[Detection]      = []
    last_hand: Optional[HandResult] = None

    try:
        while not _stop_flag.is_set():
            frame = _get_frame()
            if frame is None:
                time.sleep(0.01)
                cv2.waitKey(1)
                continue

            # cv2.flip(frame, 1) 已注解（翻轉會加重誤判）
            frame = np.ascontiguousarray(frame)

            run_hands = (fsm.state == FIState.GUIDING_HAND)

            if not run_hands:
                dets      = detector.infer(frame.copy())
                last_dets = dets
                hand      = None
                last_hand = None
            else:
                if frame_toggle:
                    dets      = detector.infer(frame.copy())
                    last_dets = dets
                    hand      = last_hand
                else:
                    dets      = last_dets
                    hand      = hands.detect(
                        np.array(frame, copy=True, order="C"))
                    last_hand = hand
                frame_toggle = not frame_toggle

            fsm.step(frame, dets, hand)

            target_det = fsm._find_target(dets)
            overlay    = draw_overlay(frame, fsm, dets, target_det, hand)
            cv2.imshow(WIN, overlay)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27): break
            elif k == ord('r'): fsm._reset()
            elif k == ord('c'): fsm.confirm()
            elif k == ord('t'): fsm.set_target("杯子", "cup")
            elif k == ord('1'): fsm.set_target("水壺", "bottle")
            elif k == ord('2'): fsm.set_target("手機", "cell phone")
            elif k == ord('3'): fsm.set_target("筆電", "laptop")
            elif k == ord('4'): fsm.set_target("碗",   "bowl")
            elif k == ord('5'): fsm.set_target("書",   "book")
            elif k == ord('6'): fsm.set_target("鍵盤", "keyboard")
            elif k == ord('7'): fsm.set_target("滑鼠", "mouse")

    finally:
        _stop_flag.set()
        unregister_from_app_main()
        if mic:    mic.stop()
        hands.close()
        fsm.close()
        cv2.destroyAllWindows()
        log.info("[SHUTDOWN] 已關閉")


if __name__ == "__main__":
    main()
