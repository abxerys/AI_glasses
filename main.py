# main.py
# -*- coding: utf-8 -*-
from api.voice_engine import recognize_audio_from_file, text_to_speech_file
from asr_core import process_voice_file_to_ai
import os, sys, time, json, asyncio, base64, audioop
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
import re

from qwen_extractor import extract_english_label
from navigation_master import NavigationMaster, OrchestratorResult
from workflow_blindpath import BlindPathNavigator
from workflow_crossstreet import CrossStreetNavigator
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from obstacle_detector_client import ObstacleDetectorClient

import mediapipe as mp
import bridge_io
import threading

# ===== 尋物狀態機（移植自 aiglass2）=====
from find_item_fsm import FindItemFSM, FindItemConfig

# ---- Windows 事件循环策略 ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- DashScope ASR 基础 ----
from groq import AsyncGroq
GROQ_API_KEY = ""  # ← 換成你的 key
groq_client  = AsyncGroq(api_key=GROQ_API_KEY)

MODEL        = "paraformer-realtime-v2"
SAMPLE_RATE  = 16000
AUDIO_FMT    = "pcm"
CHUNK_MS     = 20
BYTES_CHUNK  = SAMPLE_RATE * CHUNK_MS // 1000 * 2
SILENCE_20MS = bytes(BYTES_CHUNK)

from audio_stream import (
    register_stream_route,
    broadcast_pcm16_realtime,
    hard_reset_audio,
    BYTES_PER_20MS_16K,
    is_playing_now,
    current_ai_task,
)
from omni_client import stream_chat, OmniStreamPiece
from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
)
from audio_player import initialize_audio_system, play_voice_text

import sync_recorder
import signal
import atexit

UDP_IP   = "0.0.0.0"
UDP_PORT = 12345

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)

camera_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None

# 导航相关全局变量
blind_path_navigator = None
navigation_active = False
yolo_seg_model = None
obstacle_detector = None
cross_street_navigator = None
cross_street_active = False
orchestrator = None

# omni对话状态标志
omni_conversation_active = False
omni_previous_nav_state = None

# ===== 尋物狀態機全域實例 =====
find_item_fsm: Optional[FindItemFSM] = None


def _get_or_create_fsm() -> FindItemFSM:
    global find_item_fsm
    if find_item_fsm is None:
        find_item_fsm = FindItemFSM(
            cfg=FindItemConfig(),
            tts_fn=play_voice_text,
        )
        print("[FIND_ITEM] FindItemFSM 已初始化")
    return find_item_fsm


def load_navigation_models():
    global yolo_seg_model, obstacle_detector
    try:
        seg_model_path = os.getenv("BLIND_PATH_MODEL", r"model\yolo-seg.pt")
        if os.path.exists(seg_model_path):
            print(f"[NAVIGATION] 模型文件存在，开始加载...")
            yolo_seg_model = YOLO(seg_model_path)
            if torch.cuda.is_available():
                yolo_seg_model.to("cuda")
                print(f"[NAVIGATION] 盲道分割模型加载成功并放到GPU: {yolo_seg_model.device}")
            else:
                print("[NAVIGATION] CUDA不可用，模型仍在CPU")
            try:
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                yolo_seg_model.predict(test_img,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False)
                print(f"[NAVIGATION] 模型测试成功")
            except Exception as e:
                print(f"[NAVIGATION] 模型测试失败: {e}")
        else:
            print(f"[NAVIGATION] 错误：找不到模型文件: {seg_model_path}")

        obstacle_model_path = os.getenv("OBSTACLE_MODEL", r"model\yoloe-11l-seg.pt")
        if os.path.exists(obstacle_model_path):
            try:
                obstacle_detector = ObstacleDetectorClient(model_path=obstacle_model_path)
                print(f"[NAVIGATION] ========== YOLO-E 障碍物检测器加载成功 ==========")
            except Exception as e:
                print(f"[NAVIGATION] 障碍物检测器加载失败: {e}")
                import traceback; traceback.print_exc()
                obstacle_detector = None
        else:
            print(f"[NAVIGATION] 警告：找不到障碍物检测模型: {obstacle_model_path}")
    except Exception as e:
        print(f"[NAVIGATION] 模型加载失败: {e}")
        import traceback; traceback.print_exc()


print("[NAVIGATION] 开始加载导航模型...")
load_navigation_models()
print(f"[NAVIGATION] 模型加载完成 - yolo_seg_model: {yolo_seg_model is not None}")

print("[RECORDER] 启动同步录制系统...")
sync_recorder.start_recording()
print("[RECORDER] 录制系统已启动，将自动保存视频和音频")


def cleanup_on_exit():
    print("\n[SYSTEM] 正在关闭录制器...")
    try:
        sync_recorder.stop_recording()
        print("[SYSTEM] 录制文件已保存")
    except Exception as e:
        print(f"[SYSTEM] 关闭录制器时出错: {e}")
    global find_item_fsm
    if find_item_fsm is not None:
        try:
            find_item_fsm.close()
            print("[FIND_ITEM] FSM 資源已釋放")
        except Exception as e:
            print(f"[FIND_ITEM] 釋放 FSM 時出錯: {e}")


def signal_handler(sig, frame):
    print("\n[SYSTEM] 收到中断信号，正在安全退出...")
    cleanup_on_exit()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_on_exit)
print("[RECORDER] 已注册退出处理器 - Ctrl+C时会自动保存录制文件")

try:
    import trafficlight_detection
    print("[TRAFFIC_LIGHT] 开始预加载红绿灯检测模型...")
    if trafficlight_detection.init_model():
        print("[TRAFFIC_LIGHT] 红绿灯检测模型预加载成功")
        try:
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = trafficlight_detection.process_single_frame(test_img)
            print("[TRAFFIC_LIGHT] 模型预热完成")
        except Exception as e:
            print(f"[TRAFFIC_LIGHT] 模型预热失败: {e}")
    else:
        print("[TRAFFIC_LIGHT] 红绿灯检测模型预加载失败")
except Exception as e:
    print(f"[TRAFFIC_LIGHT] 红绿灯模型预加载出错: {e}")

interrupt_lock = asyncio.Lock()


# ════════════════════════════════════════════════════════════
# 本機 Webcam 備援（localhost 測試用）
# ════════════════════════════════════════════════════════════
LOCAL_WEBCAM_INDEX = int(os.getenv("LOCAL_WEBCAM_INDEX", "0"))


async def _local_webcam_broadcaster():
    """
    無 ESP32 相機連線時，從本機 webcam 讀取畫面並廣播給所有 camera_viewers。

    行為：
      · ESP32 連線後自動退讓（釋放 webcam）
      · ESP32 斷線後重新開啟 webcam 接管廣播
      · 沒有任何 viewer 時暫停讀取（省 CPU）
      · blocking cap.read() 丟進 ThreadPoolExecutor，不阻塞 asyncio event loop

    停用方法：設定環境變數 DISABLE_LOCAL_CAM=1
    切換攝影機：設定環境變數 LOCAL_WEBCAM_INDEX=1（預設 0）
    """
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="LocalCam"
    )
    loop = asyncio.get_event_loop()
    cap = None
    was_esp32 = False

    def _open():
        c = cv2.VideoCapture(LOCAL_WEBCAM_INDEX)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_FPS,          30)
        return c

    def _read(c): return c.read()
    def _rel(c):  c.release()

    try:
        while True:
            # ── ESP32 已連接 → 讓出相機 ──
            if esp32_camera_ws is not None:
                if cap is not None:
                    await loop.run_in_executor(executor, _rel, cap)
                    cap = None
                    print("[LOCAL_CAM] ESP32 已連接，本機 webcam 暫停")
                was_esp32 = True
                await asyncio.sleep(0.5)
                continue

            if was_esp32:
                was_esp32 = False
                print("[LOCAL_CAM] ESP32 已斷線，嘗試開啟本機 webcam…")

            # ── 無觀看者 → 暫停讀取 ──
            if not camera_viewers:
                if cap is not None:
                    await loop.run_in_executor(executor, _rel, cap)
                    cap = None
                await asyncio.sleep(0.3)
                continue

            # ── 開啟相機 ──
            if cap is None:
                cap = await loop.run_in_executor(executor, _open)
                if not cap.isOpened():
                    print(f"[LOCAL_CAM] 無法開啟 webcam #{LOCAL_WEBCAM_INDEX}，2s 後重試")
                    cap = None
                    await asyncio.sleep(2.0)
                    continue
                print(f"[LOCAL_CAM] 本機 webcam #{LOCAL_WEBCAM_INDEX} 已開啟，開始廣播")

            # ── 讀取一幀 ──
            ok, frame = await loop.run_in_executor(executor, _read, cap)
            if not ok or frame is None:
                await asyncio.sleep(0.05)
                continue

            try:
                ok2, enc = cv2.imencode(".jpg", frame,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            except Exception:
                ok2 = False

            if ok2:
                jpeg_data = enc.tobytes()
                try:
                    last_frames.append((time.time(), jpeg_data))
                except Exception:
                    pass
                dead = []
                for vws in list(camera_viewers):
                    try:
                        await vws.send_bytes(jpeg_data)
                    except Exception:
                        dead.append(vws)
                for d in dead:
                    camera_viewers.discard(d)

            await asyncio.sleep(1 / 25)   # ~25 FPS

    except asyncio.CancelledError:
        if cap is not None:
            try: cap.release()
            except Exception: pass
        executor.shutdown(wait=False)


# ════════════════════════════════════════════════════════════
# 廣播輔助
# ════════════════════════════════════════════════════════════
async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)


async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)


async def full_system_reset(reason: str = ""):
    await hard_reset_audio(reason or "full_system_reset")
    await stop_current_recognition()
    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []
    try:
        last_frames.clear()
    except Exception:
        pass
    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass
    print("[SYSTEM] full reset done.", flush=True)


# ===== 尋物模式啟停 =====
def start_item_search(item_cn: str, label_en: str):
    fsm = _get_or_create_fsm()
    fsm.set_target(item_cn, label_en)
    print(f"[FIND_ITEM] 開始尋找：{item_cn} ({label_en})", flush=True)


def stop_item_search_fsm(confirm: bool = False):
    global find_item_fsm
    if find_item_fsm is not None:
        if confirm:
            find_item_fsm.confirm_found()
        else:
            find_item_fsm.reset()
    print("[FIND_ITEM] 尋物已停止", flush=True)


# ========= 指令路由器 =========
async def start_ai_with_text_custom(user_text: str):
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator

    if orchestrator:
        current_state = orchestrator.get_state()
        if current_state not in ["CHAT", "IDLE"]:
            allowed_keywords = ["帮我看", "帮我看下", "帮我找", "找一下", "看看", "识别一下"]
            is_allowed_query = any(keyword in user_text for keyword in allowed_keywords)
            nav_control_keywords = [
                "开始过马路", "过马路结束", "开始导航", "盲道导航",
                "停止导航", "结束导航", "检测红绿灯", "看红绿灯",
                "停止检测", "停止红绿灯"
            ]
            is_nav_control = any(keyword in user_text for keyword in nav_control_keywords)
            if not is_allowed_query and not is_nav_control:
                mode_name = "红绿灯检测" if current_state == "TRAFFIC_LIGHT_DETECTION" else "导航"
                print(f"[{mode_name}模式] 丢弃非对话语音: {user_text}")
                return

    if "开始过马路" in user_text or "帮我过马路" in user_text:
        stop_item_search_fsm()
        if orchestrator:
            orchestrator.start_crossing()
            play_voice_text("过马路模式已启动。")
            await ui_broadcast_final("[系统] 过马路模式已启动")
        else:
            play_voice_text("启动过马路模式失败，请稍后重试。")
            await ui_broadcast_final("[系统] 导航系统未就绪")
        return

    if "过马路结束" in user_text or "结束过马路" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            play_voice_text("已停止导航。")
            await ui_broadcast_final("[系统] 过马路模式已停止")
        else:
            await ui_broadcast_final("[系统] 导航系统未运行")
        return

    if "检测红绿灯" in user_text or "看红绿灯" in user_text:
        try:
            import trafficlight_detection
            if orchestrator:
                orchestrator.start_traffic_light_detection()
            success = trafficlight_detection.init_model()
            trafficlight_detection.reset_detection_state()
            await ui_broadcast_final("[系统] 红绿灯检测已启动" if success else "[系统] 红绿灯模型加载失败")
        except Exception as e:
            await ui_broadcast_final(f"[系统] 启动失败: {e}")
        return

    if "停止检测" in user_text or "停止红绿灯" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
        await ui_broadcast_final("[系统] 红绿灯检测已停止")
        return

    if "开始导航" in user_text or "盲道导航" in user_text or "帮我导航" in user_text:
        stop_item_search_fsm()
        if orchestrator:
            orchestrator.start_blind_path_navigation()
            await ui_broadcast_final("[系统] 盲道导航已启动")
        else:
            await ui_broadcast_final("[系统] 导航系统未就绪")
        return

    if "停止导航" in user_text or "结束导航" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            await ui_broadcast_final("[系统] 盲道导航已停止")
        else:
            await ui_broadcast_final("[系统] 导航系统未运行")
        return

    nav_cmd_keywords = [
        "开始过马路", "过马路结束", "开始导航", "盲道导航",
        "停止导航", "结束导航", "立即通过", "现在通过", "继续"
    ]
    if any(k in user_text for k in nav_cmd_keywords):
        if orchestrator:
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final("[系统] 导航模式已更新")
        else:
            await ui_broadcast_final("[系统] 导航统领器未初始化")
        return

    # ── 尋物 ──
    find_pattern = r"(?:^\s*帮我)?\s*找一下\s*(.+?)(?:。|！|？|$)"
    match = re.search(find_pattern, user_text)
    if match:
        item_cn = match.group(1).strip()
        if item_cn:
            label_en, src = extract_english_label(item_cn)
            print(f"[FIND_ITEM] 尋物指令：'{item_cn}' -> '{label_en}' (src={src})", flush=True)
            if orchestrator:
                orchestrator.start_item_search()
            start_item_search(item_cn, label_en)
            play_voice_text(f"好的，正在幫你尋找{item_cn}。")
            await ui_broadcast_final(f"[找物品] 正在寻找 {item_cn}...")
            return

    if "找到了" in user_text or "拿到了" in user_text:
        print("[FIND_ITEM] 使用者確認找到物品", flush=True)
        stop_item_search_fsm(confirm=True)
        if orchestrator:
            orchestrator.stop_item_search(restore_nav=True)
            current_state = orchestrator.get_state()
            if current_state in [
                "BLINDPATH_NAV", "SEEKING_CROSSWALK",
                "WAIT_TRAFFIC_LIGHT", "CROSSING", "SEEKING_NEXT_BLINDPATH"
            ]:
                await ui_broadcast_final("[找物品] 已找到物品，继续导航。")
            else:
                await ui_broadcast_final("[找物品] 已找到物品。")
        else:
            await ui_broadcast_final("[找物品] 已找到物品。")
        return

    # ── 一般 AI 對話 ──
    global omni_conversation_active, omni_previous_nav_state
    omni_conversation_active = True
    if orchestrator:
        current_state = orchestrator.get_state()
        if current_state not in ["CHAT", "IDLE"]:
            omni_previous_nav_state = current_state
            orchestrator.force_state("CHAT")
        else:
            omni_previous_nav_state = None

    fsm = find_item_fsm
    if fsm is not None and not fsm.is_idle():
        print("[AI] 尋物 FSM 執行中，略過 AI 對話", flush=True)
        return

    await start_ai_with_text(user_text)


# ========= Omni 播放 =========
async def start_ai_with_text(user_text: str):
    async def _runner():
        txt_buf: List[str] = []
        rate_state = None
        content_list = []
        if last_frames:
            try:
                _, jpeg_bytes = last_frames[-1]
                img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            except Exception:
                pass
        content_list.append({"type": "text", "text": user_text})

        try:
            async for piece in stream_chat(content_list, voice="Cherry", audio_format="wav"):
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try:
                        await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                    except Exception:
                        pass
                if piece.audio_b64:
                    try:
                        pcm24 = base64.b64decode(piece.audio_b64)
                    except Exception:
                        pcm24 = b""
                    if pcm24:
                        pcm8k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)
                        pcm8k = audioop.mul(pcm8k, 2, 0.60)
                        if pcm8k:
                            await broadcast_pcm16_realtime(pcm8k)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            try:
                await ui_broadcast_final(f"[AI] 发生错误：{e}")
            except Exception:
                pass
        finally:
            global omni_conversation_active, omni_previous_nav_state
            omni_conversation_active = False
            if orchestrator and omni_previous_nav_state:
                orchestrator.force_state(omni_previous_nav_state)
                omni_previous_nav_state = None
            from audio_stream import stream_clients
            for sc in list(stream_clients):
                if not sc.abort_event.is_set():
                    try: sc.q.put_nowait(b"\x00"*BYTES_PER_20MS_16K)
                    except Exception: pass
                    try: sc.q.put_nowait(None)
                    except Exception: pass
            final_text = ("".join(txt_buf)).strip() or "（空响应）"
            try:
                await ui_broadcast_final("[AI] " + final_text)
            except Exception:
                pass

    await hard_reset_audio("start_ai_with_text")
    loop = asyncio.get_running_loop()
    from audio_stream import __dict__ as _as_dict
    task = loop.create_task(_runner())
    _as_dict["current_ai_task"] = task


# ---------- 页面 / 健康 ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


register_stream_route(app)


# ---------- WebSocket：WebUI 文本 ----------
@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)


# ---------- WebSocket：ESP32 音频入口 ----------
@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("\n[AUDIO] ESP32 audio client connected")
    recognition = None
    streaming = False
    last_ts = time.monotonic()
    keepalive_task: Optional[asyncio.Task] = None

    async def stop_rec(send_notice: Optional[str] = None):
        nonlocal recognition, streaming, keepalive_task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try: await keepalive_task
            except Exception: pass
        keepalive_task = None
        if recognition:
            try: recognition.stop()
            except Exception: pass
            recognition = None
        await set_current_recognition(None)
        streaming = False
        if send_notice:
            try: await ws.send_text(send_notice)
            except Exception: pass

    async def on_sdk_error(_msg: str):
        await stop_rec(send_notice="RESTART")

    async def keepalive_loop():
        nonlocal last_ts, recognition, streaming
        try:
            while streaming and recognition is not None:
                idle = time.monotonic() - last_ts
                if idle > 0.35:
                    try:
                        for _ in range(30):
                            recognition.send_audio_frame(SILENCE_20MS)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("keepalive send failed")
                        return
                await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            return

    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call \"receive\"" in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    print("[AUDIO] START received")
                    await stop_rec()
                    loop = asyncio.get_running_loop()
                    def post(coro):
                        asyncio.run_coroutine_threadsafe(coro, loop)

                    cb = ASRCallback(
                        on_sdk_error=lambda s: post(on_sdk_error(s)),
                        post=post,
                        ui_broadcast_partial=ui_broadcast_partial,
                        ui_broadcast_final=ui_broadcast_final,
                        is_playing_now_fn=is_playing_now,
                        start_ai_with_text_fn=start_ai_with_text_custom,
                        full_system_reset_fn=full_system_reset,
                        interrupt_lock=interrupt_lock,
                    )
                    recognition = groq_client.audio.transcriptions
                    streaming = True
                    recognition.start()
                    await set_current_recognition(recognition)
                    streaming = True
                    last_ts = time.monotonic()
                    keepalive_task = asyncio.create_task(keepalive_loop())
                    await ui_broadcast_partial("（已开始接收音频…）")
                    await ws.send_text("OK:STARTED")

                elif cmd == "STOP":
                    if recognition:
                        for _ in range(15):
                            try: recognition.send_audio_frame(SILENCE_20MS)
                            except Exception: break
                    await stop_rec(send_notice="OK:STOPPED")
                    latest_wav = sync_recorder.get_latest_audio_path()
                    if latest_wav:
                        asyncio.create_task(process_voice_file_to_ai(latest_wav, cb))

                elif raw.startswith("PROMPT:"):
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                if streaming and recognition:
                    try:
                        recognition.send_audio_frame(msg["bytes"])
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("send_audio_frame failed")

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        await stop_rec()
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[WS] ESP32 audio connection closed")


# ---------- WebSocket：瀏覽器麥克風入口（localhost 測試用）----------
@app.websocket("/ws/browser_audio")
async def ws_browser_audio(ws: WebSocket):
    """
    接收瀏覽器 getUserMedia 麥克風的 PCM16/16000Hz 音訊。
    與 /ws_audio (ESP32) 協議完全相容：START / STOP / binary PCM bytes。

    使用場景：localhost 測試（無 ESP32 時用瀏覽器 mic 代替）。
    當 ESP32 已連線時拒絕瀏覽器連線，避免兩路同時使用衝突。
    """
    if esp32_audio_ws is not None:
        await ws.close(code=1013, reason="ESP32 audio already connected; disconnect it first")
        return

    await ws.accept()
    print("\n[BROWSER_MIC] 瀏覽器麥克風已連接")
    await ui_broadcast_partial("（瀏覽器麥克風模式）")

    recognition = None
    streaming = False
    last_ts = time.monotonic()
    keepalive_task: Optional[asyncio.Task] = None

    async def stop_rec(send_notice: Optional[str] = None):
        nonlocal recognition, streaming, keepalive_task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try: await keepalive_task
            except Exception: pass
        keepalive_task = None
        if recognition:
            try: recognition.stop()
            except Exception: pass
            recognition = None
        await set_current_recognition(None)
        streaming = False
        if send_notice:
            try: await ws.send_text(send_notice)
            except Exception: pass

    async def on_sdk_error(_msg: str):
        await stop_rec(send_notice="RESTART")

    async def keepalive_loop():
        nonlocal last_ts, recognition, streaming
        try:
            while streaming and recognition is not None:
                idle = time.monotonic() - last_ts
                if idle > 0.35:
                    try:
                        for _ in range(30):
                            recognition.send_audio_frame(SILENCE_20MS)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("browser_mic keepalive failed")
                        return
                await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            return

    try:
        while True:
            if ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call \"receive\"" in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    print("[BROWSER_MIC] START received")
                    await stop_rec()
                    loop = asyncio.get_running_loop()
                    def post(coro):
                        asyncio.run_coroutine_threadsafe(coro, loop)

                    cb = ASRCallback(
                        on_sdk_error=lambda s: post(on_sdk_error(s)),
                        post=post,
                        ui_broadcast_partial=ui_broadcast_partial,
                        ui_broadcast_final=ui_broadcast_final,
                        is_playing_now_fn=is_playing_now,
                        start_ai_with_text_fn=start_ai_with_text_custom,
                        full_system_reset_fn=full_system_reset,
                        interrupt_lock=interrupt_lock,
                    )
                    recognition = groq_client.audio.transcriptions
                    streaming = True
                    recognition.start()
                    await set_current_recognition(recognition)
                    streaming = True
                    last_ts = time.monotonic()
                    keepalive_task = asyncio.create_task(keepalive_loop())
                    await ui_broadcast_partial("（瀏覽器麥克風錄音中…）")
                    await ws.send_text("OK:STARTED")

                elif cmd == "STOP":
                    if recognition:
                        for _ in range(15):
                            try: recognition.send_audio_frame(SILENCE_20MS)
                            except Exception: break
                    await stop_rec(send_notice="OK:STOPPED")

                elif raw.startswith("PROMPT:"):
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")

            elif "bytes" in msg and msg["bytes"] is not None:
                if streaming and recognition:
                    try:
                        recognition.send_audio_frame(msg["bytes"])
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("browser_mic send_audio_frame failed")

    except Exception as e:
        print(f"[BROWSER_MIC ERROR] {e}")
    finally:
        await stop_rec()
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        print("[BROWSER_MIC] 瀏覽器麥克風已斷線")


# ---------- WebSocket：ESP32 相机入口 ----------
@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    global esp32_camera_ws, blind_path_navigator, cross_street_navigator, \
           cross_street_active, navigation_active, orchestrator

    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CAMERA] ESP32 connected")

    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        print("[NAVIGATION] 盲道导航器已初始化")

    if cross_street_navigator is None and yolo_seg_model:
        cross_street_navigator = CrossStreetNavigator(
            seg_model=yolo_seg_model, coco_model=None, obs_model=None
        )
        print("[CROSS_STREET] 过马路导航器已初始化")

    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        print("[NAV MASTER] 统领状态机已初始化")

    fsm = _get_or_create_fsm()
    frame_counter = 0

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                frame_counter += 1

                try:
                    sync_recorder.record_frame(data)
                except Exception as e:
                    if frame_counter % 100 == 0:
                        print(f"[RECORDER] 录制帧失败: {e}")

                try:
                    last_frames.append((time.time(), data))
                except Exception:
                    pass

                try:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None or bgr.size == 0:
                        bgr = None
                except Exception:
                    bgr = None

                if frame_counter % 30 == 0:
                    state_dbg = orchestrator.get_state() if orchestrator else "N/A"
                    fsm_state = fsm.state.name if fsm else "N/A"
                    print(f"[DEBUG] 帧:{frame_counter} orch={state_dbg} fsm={fsm_state}")

                # ── 尋物模式：FSM 接管 ──
                if orchestrator and orchestrator.get_state() == "ITEM_SEARCH" and bgr is not None:
                    out_frame = fsm.step(bgr)
                    if camera_viewers and out_frame is not None:
                        ok, enc = cv2.imencode(".jpg", out_frame,
                                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for d in dead:
                                camera_viewers.discard(d)
                    continue

                # ── 其他導航模式 ──
                if orchestrator and bgr is not None:
                    current_state = orchestrator.get_state()
                    out_img = bgr
                    try:
                        if current_state == "TRAFFIC_LIGHT_DETECTION":
                            import trafficlight_detection
                            result = trafficlight_detection.process_single_frame(
                                bgr, ui_broadcast_callback=ui_broadcast_final)
                            out_img = result['vis_image'] if result['vis_image'] is not None else bgr
                        else:
                            res = orchestrator.process_frame(bgr)
                            if res.guidance_text:
                                try:
                                    play_voice_text(res.guidance_text)
                                    await ui_broadcast_final(f"[导航] {res.guidance_text}")
                                except Exception:
                                    pass
                            out_img = res.annotated_image if res.annotated_image is not None else bgr
                    except Exception as e:
                        if frame_counter % 100 == 0:
                            print(f"[NAV MASTER] 处理帧时出错: {e}")

                    if camera_viewers and out_img is not None:
                        ok, enc = cv2.imencode(".jpg", out_img,
                                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for d in dead:
                                camera_viewers.discard(d)
                    continue

                # ── 回退：傳原始畫面 ──
                if camera_viewers and bgr is not None:
                    try:
                        ok, enc = cv2.imencode(".jpg", bgr,
                                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for d in dead:
                                camera_viewers.discard(d)
                    except Exception as e:
                        print(f"[CAMERA] Broadcast error: {e}")

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        print("[CAMERA] ESP32 disconnected")
        if blind_path_navigator:
            blind_path_navigator.reset()
        if cross_street_navigator:
            cross_street_navigator.reset()
        if orchestrator:
            orchestrator.reset()
            print("[NAV MASTER] 统领器已重置")
        if find_item_fsm:
            find_item_fsm.reset()
            print("[FIND_ITEM] FSM 已重置")


# ---------- WebSocket：浏览器订阅相机帧 ----------
@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try:
            camera_viewers.remove(ws)
        except Exception:
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)


# ---------- WebSocket：浏览器订阅 IMU ----------
@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)


async def imu_broadcast(msg: str):
    if not imu_ws_clients:
        return
    dead = []
    for ws in list(imu_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        imu_ws_clients.discard(ws)


# ---------- 服务端 IMU 估计 ----------
from math import atan2, hypot, pi
GRAV_BETA   = 0.98
STILL_W     = 0.4
YAW_DB      = 0.08
YAW_LEAK    = 0.2
ANG_EMA     = 0.15
AUTO_REZERO = True
USE_PROJ    = True
FREEZE_STILL= True
G     = 9.807
A_TOL = 0.08 * G
gLP = {"x":0.0, "y":0.0, "z":0.0}
gOff= {"x":0.0, "y":0.0, "z":0.0}
BIAS_ALPHA = 0.002
yaw  = 0.0
Rf = Pf = Yf = 0.0
ref = {"roll":0.0, "pitch":0.0, "yaw":0.0}
holdStart = 0.0
isStill   = False
last_ts_imu = 0.0
last_wall = 0.0
imu_store: List[Dict[str, Any]] = []


def _wrap180(a: float) -> float:
    a = a % 360.0
    if a >= 180.0: a -= 360.0
    if a < -180.0: a += 360.0
    return a


def process_imu_and_maybe_store(d: Dict[str, Any]):
    global gLP, gOff, yaw, Rf, Pf, Yf, ref, holdStart, isStill, last_ts_imu, last_wall

    t_ms = float(d.get("ts", 0.0))
    now_wall = time.monotonic()
    if t_ms <= 0.0:
        t_ms = (now_wall * 1000.0)
    if last_ts_imu <= 0.0 or t_ms <= last_ts_imu or (t_ms - last_ts_imu) > 3000.0:
        dt = 0.02
    else:
        dt = (t_ms - last_ts_imu) / 1000.0
    last_ts_imu = t_ms

    ax = float(((d.get("accel") or {}).get("x", 0.0)))
    ay = float(((d.get("accel") or {}).get("y", 0.0)))
    az = float(((d.get("accel") or {}).get("z", 0.0)))
    wx = float(((d.get("gyro")  or {}).get("x", 0.0)))
    wy = float(((d.get("gyro")  or {}).get("y", 0.0)))
    wz = float(((d.get("gyro")  or {}).get("z", 0.0)))

    gLP["x"] = GRAV_BETA * gLP["x"] + (1.0 - GRAV_BETA) * ax
    gLP["y"] = GRAV_BETA * gLP["y"] + (1.0 - GRAV_BETA) * ay
    gLP["z"] = GRAV_BETA * gLP["z"] + (1.0 - GRAV_BETA) * az
    gmag = hypot(gLP["x"], gLP["y"], gLP["z"]) or 1.0
    gHat = {"x": gLP["x"]/gmag, "y": gLP["y"]/gmag, "z": gLP["z"]/gmag}

    roll  = (atan2(az, ay)   * 180.0 / pi)
    pitch = (atan2(-ax, ay)  * 180.0 / pi)

    aNorm = hypot(ax, ay, az); wNorm = hypot(wx, wy, wz)
    nearFlat = (abs(roll) < 2.0 and abs(pitch) < 2.0)
    stillCond = (abs(aNorm - G) < A_TOL) and (wNorm < STILL_W)

    if stillCond:
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0: isStill = True
        gOff["x"] = (1.0 - BIAS_ALPHA)*gOff["x"] + BIAS_ALPHA*wx
        gOff["y"] = (1.0 - BIAS_ALPHA)*gOff["y"] + BIAS_ALPHA*wy
        gOff["z"] = (1.0 - BIAS_ALPHA)*gOff["z"] + BIAS_ALPHA*wz
    else:
        holdStart = 0.0; isStill = False

    if USE_PROJ:
        yawdot = ((wx - gOff["x"])*gHat["x"] + (wy - gOff["y"])*gHat["y"] + (wz - gOff["z"])*gHat["z"])
    else:
        yawdot = (wy - gOff["y"])

    if abs(yawdot) < YAW_DB: yawdot = 0.0
    if FREEZE_STILL and stillCond: yawdot = 0.0

    yaw = _wrap180(yaw + yawdot * dt)

    if (YAW_LEAK > 0.0) and nearFlat and stillCond and abs(yaw) > 0.0:
        step = YAW_LEAK * dt * (-1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0))
        if abs(yaw) <= abs(step): yaw = 0.0
        else: yaw += step

    global Rf, Pf, Yf, ref, last_wall
    Rf = ANG_EMA * roll  + (1.0 - ANG_EMA) * Rf
    Pf = ANG_EMA * pitch + (1.0 - ANG_EMA) * Pf
    Yf = ANG_EMA * yaw   + (1.0 - ANG_EMA) * Yf

    if AUTO_REZERO and nearFlat and (wNorm < STILL_W):
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            ref.update({"roll": Rf, "pitch": Pf, "yaw": Yf})
            isStill = True

    R = _wrap180(Rf - ref["roll"])
    P = _wrap180(Pf - ref["pitch"])
    Y = _wrap180(Yf - ref["yaw"])

    now_wall = time.monotonic()
    if last_wall <= 0.0 or (now_wall - last_wall) >= 0.100:
        last_wall = now_wall
        item = {
            "ts": t_ms/1000.0,
            "angles": {"roll": R, "pitch": P, "yaw": Y},
            "accel":  {"x": ax, "y": ay, "z": az},
            "gyro":   {"x": wx, "y": wy, "z": wz},
        }
        imu_store.append(item)


# ---------- UDP 接收 IMU ----------
class UDPProto(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")

    def datagram_received(self, data, addr):
        try:
            s = data.decode('utf-8', errors='ignore').strip()
            d = json.loads(s)
            if 'ts' not in d and 'timestamp_ms' in d:
                d['ts'] = d.pop('timestamp_ms')
            process_imu_and_maybe_store(d)
            asyncio.create_task(imu_broadcast(json.dumps(d)))
        except Exception:
            pass


@app.on_event("startup")
async def on_startup_init_audio():
    def _init():
        try:
            initialize_audio_system()
        except Exception as e:
            print(f"[AUDIO] 初始化失败: {e}")
    threading.Thread(target=_init, daemon=True).start()


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT))


@app.on_event("startup")
async def on_startup_local_cam():
    """
    啟動本機 webcam 備援廣播任務。
    無 ESP32 時自動廣播本機鏡頭畫面到 /ws/viewer。
    設定環境變數 DISABLE_LOCAL_CAM=1 可停用。
    """
    if os.getenv("DISABLE_LOCAL_CAM", "0") != "1":
        asyncio.create_task(_local_webcam_broadcaster())
        print("[LOCAL_CAM] 本機 webcam 備援任務已啟動（設定 DISABLE_LOCAL_CAM=1 可停用）")
    else:
        print("[LOCAL_CAM] 本機 webcam 備援已停用（DISABLE_LOCAL_CAM=1）")


@app.on_event("shutdown")
async def on_shutdown():
    print("[SHUTDOWN] 开始清理资源...")
    global find_item_fsm
    if find_item_fsm is not None:
        find_item_fsm.close()
    await hard_reset_audio("shutdown")
    print("[SHUTDOWN] 资源清理完成")


def get_last_frames():
    return last_frames


def get_camera_ws():
    return esp32_camera_ws


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8765,
        log_level="warning", access_log=False,
        loop="asyncio", workers=1, reload=False
    )