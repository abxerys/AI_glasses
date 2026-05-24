# asr_core.py
# -*- coding: utf-8 -*-
import os, json, asyncio
from typing import Any, Dict, List, Optional, Callable, Tuple
from openai import OpenAI

ASR_DEBUG_RAW = os.getenv("ASR_DEBUG_RAW", "0") == "1"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
groq_client = (
    OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    if GROQ_API_KEY else None
)

def _shorten(s: str, limit: int = 200) -> str:
    if not s:
        return ""
    return s if len(s) <= limit else (s[:limit] + "…")

def _safe_to_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict): return x
    for attr in ("to_dict", "model_dump", "__dict__"):
        try:
            v = getattr(x, attr, None)
        except Exception:
            v = None
        if callable(v):
            try:
                d = v()
                if isinstance(d, dict): return d
            except Exception:
                pass
        elif isinstance(v, dict):
            return v
    try:
        s = str(x)
        if s and s.lstrip().startswith("{") and s.rstrip().endswith("}"):
            return json.loads(s)
    except Exception:
        pass
    return {"_raw": str(x)}

def _extract_sentence(event_obj: Any) -> Tuple[Optional[str], Optional[bool]]:
    d = _safe_to_dict(event_obj)
    cands: List[Dict[str, Any]] = [d]
    for k in ("output", "data", "result"):
        v = d.get(k)
        if isinstance(v, dict):
            cands.append(v)
    for obj in cands:
        sent = obj.get("sentence")
        if isinstance(sent, dict):
            text = sent.get("text")
            is_end = sent.get("sentence_end")
            if is_end is not None:
                is_end = bool(is_end)
            return text, is_end
    for obj in cands:
        if "text" in obj and isinstance(obj.get("text"), str):
            return obj.get("text"), None
    return None, None

# ====== 仅热词触发的“全清零复位”配置 ======
INTERRUPT_KEYWORDS = set(
    os.getenv("INTERRUPT_KEYWORDS", "停下,别说了,停止").split(",")
)

def _normalize_cn(s: str) -> str:
    try:
        import unicodedata
        s = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in s)
        s = s.strip().lower()
    except Exception:
        s = (s or "").strip().lower()
    return s

# ============ ASR 全局总闸 ============
_current_recognition: Optional[object] = None
_rec_lock = asyncio.Lock()

async def set_current_recognition(r):
    global _current_recognition
    async with _rec_lock:
        _current_recognition = r

async def stop_current_recognition():
    global _current_recognition
    async with _rec_lock:
        r = _current_recognition
        _current_recognition = None
    if r:
        try:
            r.stop()  # DashScope SDK 的实时识别停止
        except Exception:
            pass

# ============ ASR 回调 ============
class ASRCallback:
    """
    设计目标：
    1) “停下 / 别说了 …”等热词一出现 → 立刻全清零复位（恢复到刚启动后的状态）。
    2) 除此之外【不接受打断】；AI 正在播报时，用户说话只做展示，不触发新一轮。
    3) 不再用 partial 叠加字符串；partial 只用于 UI 临时展示；只有 final sentence 用于驱动 AI。
    """

    def __init__(
        self,
        on_sdk_error: Callable[[str], None],
        post: Callable[[asyncio.Future], None],
        ui_broadcast_partial,
        ui_broadcast_final,
        is_playing_now_fn: Callable[[], bool],
        start_ai_with_text_fn,               # async (text)
        full_system_reset_fn,                 # async (reason)
        interrupt_lock: asyncio.Lock,
    ):
        self._on_sdk_error = on_sdk_error
        self._post = post
        self._last_partial_for_ui: str = ""   # 只用于 UI 展示
        self._last_final_text: str = ""       # 以句末 final 为准
        self._hot_interrupted: bool = False   # 本句是否因热词触发过复位（防抖）

        self._ui_partial = ui_broadcast_partial
        self._ui_final   = ui_broadcast_final
        self._is_playing = is_playing_now_fn
        self._start_ai   = start_ai_with_text_fn
        self._full_reset = full_system_reset_fn
        self._interrupt_lock = interrupt_lock

    def on_open(self):  pass
    def on_close(self): pass
    def on_complete(self): pass

    def on_error(self, err):
        try:
            self._post(self._ui_partial(""))
            self._on_sdk_error(str(err))
        except Exception:
            pass

    def on_result(self, result): self._handle(result)
    def on_event(self,  event):  self._handle(event)

    def _has_hotword(self, text: str) -> bool:
        t = _normalize_cn(text)
        if not t: return False
        for w in INTERRUPT_KEYWORDS:
            if w and _normalize_cn(w) in t:
                return True
        return False

    def _handle(self, event: Any):
        if ASR_DEBUG_RAW:
            try:
                rawd = _safe_to_dict(event)
                print("[ASR EVENT RAW]", json.dumps(rawd, ensure_ascii=False), flush=True)
            except Exception:
                pass

        text, is_end = _extract_sentence(event)
        if text is None:
            return
        text = text.strip()
        if not text:
            return

        # ---------- ① 热词优先：命中就全清零并短路，绝不送 LLM ----------
        if not self._hot_interrupted and self._has_hotword(text):
            self._hot_interrupted = True

            async def _hot_reset():
                async with self._interrupt_lock:
                    print(f"[ASR HOTWORD] '{text}' -> FULL RESET, skip LLM", flush=True)
                    await self._full_reset("Hotword interrupt")
            try:
                self._post(_hot_reset())
            except Exception:
                pass
            return

        # ---------- ② partial：仅用于 UI 展示 ----------
        self._last_partial_for_ui = text
        try:
            print(f"[ASR PARTIAL] len={len(text)} text='{_shorten(text)}'", flush=True)
            self._post(self._ui_partial(self._last_partial_for_ui))
        except Exception:
            pass

        # ---------- ③ final：仅 final 驱动 LLM（若未在播报） ----------
        if is_end is True:
            final_text = text
            try:
                print(f"[ASR FINAL]  len={len(final_text)} text='{final_text}'", flush=True)
                self._post(self._ui_final(final_text))
            except Exception:
                pass

            if (not self._is_playing()) and final_text:
                async def _run_final():
                    async with self._interrupt_lock:
                        print(f"[LLM INPUT TEXT] {final_text}", flush=True)
                        await self._start_ai(final_text)
                try:
                    self._post(_run_final())
                except Exception:
                    pass

            # 复位进入下一句
            self._last_partial_for_ui = ""
            self._last_final_text = ""
            self._hot_interrupted = False

# ====== 新增：Gemini / Google ASR 轉接邏輯 ======
def recognize_speech_whisper(audio_file_path: str) -> str:
    """呼叫 Groq Whisper API 將音檔實質轉換為文字"""
    try:
        if not os.path.exists(audio_file_path):
            print(f"[ASR Error] 找不到音檔: {audio_file_path}")
            return ""
        if groq_client is None:
            print("[ASR Error] 未設定 GROQ_API_KEY，無法使用 Groq Whisper。")
            return ""

        with open(audio_file_path, "rb") as audio_file:
            print(f"[Groq Whisper] 正在上傳音檔至 Groq 進行辨識...")
            # 呼叫 Groq 支援的 whisper-large-v3 模型
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3", 
                file=audio_file,
                language="zh",  # 強制中文辨識
                response_format="json"
            )
        return transcription.text
    except Exception as e:
        print(f"[Groq API 錯誤]: {e}")
        return ""
    
async def process_voice_file_to_ai(wav_path: str, asr_callback: ASRCallback):
    """
    1. 讀取 ESP32 傳來的錄音檔
    2. 使用 OpenAI Whisper API 辨識文字
    3. 觸發 UI 顯示與後續 AI 邏輯
    """
    try:
        print(f"[ASR_CORE] 開始處理音檔 (Whisper): {wav_path}")
        
        # 1. 呼叫 Whisper 進行辨識 (這是同步操作，但在這個非同步函式裡執行)
        text = recognize_speech_whisper(wav_path)
        
        if not text:
            print("[ASR_CORE] Whisper 辨識結果為空，跳過。")
            return

        text = text.strip()
        allowed_keywords = [
            "过马路", "過馬路", "斑马线", "斑馬線", "紅燈", "綠燈"
            "红绿灯", "紅綠燈", "盲道", "导航", "導航",
            "找", "识别", "識別", "停止", "结束", "繼續"
        ]
        allowed_keywords.extend([
            "我要找", "幫我找", "帮我找", "尋找", "寻找",
            "手機", "手机", "幫我導航", "帮我导航",
            "交通燈", "交通灯", "停止", "結束", "取消",
        ])
        if not any(keyword in text for keyword in allowed_keywords):
            print(f"[ASR_CORE] 未包含觸發關鍵字，忽略此語音: {text}")
            return
        
        print(f"[ASR_CORE] 辨識文字成功: {text}")

        # 2. 處理熱詞中斷 (沿用原本 ASRCallback 的邏輯)
        if asr_callback._has_hotword(text):
            print(f"[ASR HOTWORD] 命中熱詞 '{text}'，進行系統重置")
            async with asr_callback._interrupt_lock:
                await asr_callback._full_reset("Hotword interrupt from Whisper")
            return

        # 3. 觸發 UI 顯示
        # 注意：這裡可能需要 await 或透過 post 投遞，取決於原專案架構
        try:
            asr_callback._post(asr_callback._ui_final(text))
        except Exception as e:
            print(f"[ASR_CORE] UI 更新失敗: {e}")

        # 4. 觸發 AI 大腦 (例如你原本要接的 Gemini 或是後續的導航邏輯)
        if not asr_callback._is_playing():
            async with asr_callback._interrupt_lock:
                print(f"[LLM INPUT TEXT] {text}", flush=True)
                await asr_callback._start_ai(text)
        else:
            print("[ASR_CORE] AI 正在說話中，忽略此指令以防重疊。")

    except Exception as e:
        print(f"[ASR_CORE] Whisper 處理語音流程發生錯誤: {e}")
