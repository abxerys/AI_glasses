# gui_window.py
# -*- coding: utf-8 -*-
"""
aiglass3 桌面視窗（含自動啟動後端）

直接執行此檔案即可：
    python gui_window.py

流程：
  1. 在背景執行緒中 import main 並以 uvicorn 啟動 FastAPI 後端
  2. 等待後端 TCP 連線就緒（最多 90 秒，模型載入時間）
  3. 開啟 tkinter 視窗，連接 /ws/viewer 和 /ws_ui

若後端已在執行（遠端機器），可跳過啟動：
    python gui_window.py --host 192.168.1.100 --port 8081 --no-server
"""
from __future__ import annotations

import argparse
import queue
import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Optional
import sys
import io

# ── 依賴檢查 ──────────────────────────────────────────────
try:
    from PIL import Image, ImageTk
except ImportError:
    print("[ERROR] 缺少 Pillow，請執行：pip install Pillow")
    sys.exit(1)

try:
    import websocket  # websocket-client
except ImportError:
    print("[ERROR] 缺少 websocket-client，請執行：pip install websocket-client")
    sys.exit(1)


# ════════════════════════════════════════════════════════════
# 常數
# ════════════════════════════════════════════════════════════
SERVER_HOST      = "localhost"
SERVER_PORT      = 8765
HEALTH_TIMEOUT_S = 90        # 等後端啟動最多幾秒（模型載入較慢）
RECONNECT_DELAY  = 3.0
MSG_MAX          = 200

# 深色主題
BG          = "#0b0f14"
CARD        = "#121821"
TEXT        = "#e6edf3"
MUTED       = "#9fb0c3"
OK_COLOR    = "#7ee787"
ERR_COLOR   = "#ff8080"
LINE        = "#1f2937"
USER_BUBBLE = "#2a6df4"
AI_BUBBLE   = "#111a2e"
NAV_BUBBLE  = "#1a2e1a"
SYS_BUBBLE  = "#2a1a2e"
FIND_BUBBLE = "#2a2010"


# ════════════════════════════════════════════════════════════
# 後端啟動（背景執行緒）
# ════════════════════════════════════════════════════════════
def _start_backend_thread(host: str, port: int) -> threading.Thread:
    """
    在獨立執行緒中 import main.py 並以 uvicorn.Server 啟動 FastAPI。
    uvicorn 內部會建立自己的 asyncio event loop，
    與 tkinter 主執行緒完全隔離，不會互相干擾。
    """
    def _run():
        try:
            import uvicorn
            # import main 會觸發 load_navigation_models 等初始化
            import app_main as _app_module
            config = uvicorn.Config(
                app=_app_module.app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
                loop="asyncio",
                workers=1,
            )
            server = uvicorn.Server(config)
            server.run()
        except Exception as e:
            print(f"[BACKEND] 啟動失敗: {e}", flush=True)

    t = threading.Thread(target=_run, daemon=True, name="BackendServer")
    t.start()
    return t


def _wait_for_backend(host: str, port: int,
                      timeout: float = HEALTH_TIMEOUT_S,
                      progress_cb=None,
                      ui_update_fn=None) -> bool:
    """
    輪詢 TCP，等待後端埠開放。

    ui_update_fn 可傳入 tkinter 的 root.update()，
    確保 splash 視窗在等待期間保持響應（不凍結）。
    原本的 time.sleep(1.0) 會完全阻塞主執行緒導致視窗凍結，
    改為 20 × 0.05s 的短迴圈，每次都呼叫 ui_update_fn()。
    """
    deadline  = time.monotonic() + timeout
    start     = time.monotonic()
    last_msg  = 0.0

    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                if progress_cb:
                    progress_cb("後端已就緒，正在開啟視窗…")
                return True
        except OSError:
            pass

        now = time.monotonic()
        if now - last_msg >= 1.0:
            last_msg = now
            if progress_cb:
                elapsed = int(now - start)
                progress_cb(f"等待後端啟動… ({elapsed}s)　AI 模型載入中，請稍候")

        # ── 關鍵修復：改為短間隔迴圈，持續呼叫 ui_update_fn() ──
        # 原本的 time.sleep(1.0) 會凍結主執行緒，讓 tkinter 視窗無法重繪
        for _ in range(20):       # 20 × 0.05s ≈ 1 秒
            if ui_update_fn:
                try:
                    ui_update_fn()
                except Exception:
                    pass
            time.sleep(0.05)

    return False


# ════════════════════════════════════════════════════════════
# 啟動畫面（等後端就緒前顯示）
# ════════════════════════════════════════════════════════════
class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("aiglass3 啟動中")
        self.root.configure(bg=BG)
        self.root.geometry("440x190")
        self.root.resizable(False, False)
        self.root.eval("tk::PlaceWindow . center")

        tk.Label(self.root, text="aiglass3",
                 bg=BG, fg=TEXT,
                 font=("Microsoft YaHei", 24, "bold")).pack(pady=(30, 6))

        self._msg_var = tk.StringVar(value="正在啟動後端伺服器…")
        tk.Label(self.root, textvariable=self._msg_var,
                 bg=BG, fg=MUTED,
                 font=("Microsoft YaHei", 10),
                 wraplength=400).pack()

        import tkinter.ttk as ttk
        style = ttk.Style(self.root)
        style.theme_use("default")
        style.configure("dark.Horizontal.TProgressbar",
                         troughcolor="#1f2937",
                         background="#1f6feb",
                         bordercolor=BG,
                         lightcolor="#1f6feb",
                         darkcolor="#1f6feb")
        self._bar = ttk.Progressbar(
            self.root, style="dark.Horizontal.TProgressbar",
            mode="indeterminate", length=360,
        )
        self._bar.pack(pady=20)
        self._bar.start(10)
        # 確保視窗第一次就顯示出來
        self.root.update()

    def set_message(self, msg: str):
        self._msg_var.set(msg)
        try:
            # ── 關鍵修復：改用 update() 而非 update_idletasks() ──
            # update_idletasks() 只處理 idle 事件，不會觸發視窗重繪
            # update() 才能完整處理所有 pending 事件（含 Expose/Configure）
            self.root.update()
        except Exception:
            pass

    def close(self):
        try:
            self._bar.stop()
            self.root.destroy()
        except Exception:
            pass


# ════════════════════════════════════════════════════════════
# WebSocket 執行緒基底
# ════════════════════════════════════════════════════════════
class _WSThread(threading.Thread):
    def __init__(self, url: str, name: str):
        super().__init__(daemon=True, name=name)
        self.url       = url
        self._ws: Optional[websocket.WebSocketApp] = None
        self._stop     = threading.Event()
        self.connected = False

    def stop(self):
        self._stop.set()
        if self._ws:
            try: self._ws.close()
            except Exception: pass

    def run(self):
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self.url,
                    on_open    = self._on_open,
                    on_message = self._on_message,
                    on_error   = self._on_error,
                    on_close   = self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[{self.name}] WS error: {e}")
            if not self._stop.is_set():
                time.sleep(RECONNECT_DELAY)

    def _on_open(self, ws):
        self.connected = True; self.on_open(ws)
    def _on_message(self, ws, msg):
        self.on_message(ws, msg)
    def _on_error(self, ws, err):
        self.connected = False; self.on_error(ws, err)
    def _on_close(self, ws, code, msg):
        self.connected = False; self.on_close(ws, code, msg)

    def on_open(self, ws): pass
    def on_message(self, ws, msg): pass
    def on_error(self, ws, err): pass
    def on_close(self, ws, code, msg): pass


class CameraWSThread(_WSThread):
    def __init__(self, url, frame_q, status_cb):
        super().__init__(url, "CameraWS")
        self.frame_q = frame_q; self.status_cb = status_cb
    def on_open(self, ws):
        self.status_cb("Camera", True, "connected")
    def on_message(self, ws, msg):
        if isinstance(msg, bytes):
            try: self.frame_q.put_nowait(msg)
            except queue.Full: pass
    def on_error(self, ws, err):
        self.status_cb("Camera", False, "error")
    def on_close(self, ws, code, msg):
        self.status_cb("Camera", False, "disconnected")


class UIWSThread(_WSThread):
    def __init__(self, url, msg_q, status_cb):
        super().__init__(url, "UIWS")
        self.msg_q = msg_q; self.status_cb = status_cb
    def on_open(self, ws):
        self.status_cb("ASR", True, "connected")
    def on_message(self, ws, msg):
        if isinstance(msg, str):
            try: self.msg_q.put_nowait(msg)
            except queue.Full: pass
    def on_error(self, ws, err):
        self.status_cb("ASR", False, "error")
    def on_close(self, ws, code, msg):
        self.status_cb("ASR", False, "disconnected")


# ════════════════════════════════════════════════════════════
# 主視窗
# ════════════════════════════════════════════════════════════
class AIGlassWindow:
    def __init__(self, host: str, port: int):
        self.host     = host
        self.port     = port
        self.base_url = f"ws://{host}:{port}"

        self.frame_q: queue.Queue[bytes] = queue.Queue(maxsize=5)
        self.msg_q:   queue.Queue[str]   = queue.Queue(maxsize=100)

        self.root = tk.Tk()
        self.root.title("aiglass3 監控視窗")
        self.root.configure(bg=BG)
        self.root.geometry("1200x720")
        self.root.minsize(900, 540)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Configure>", lambda e: None)

        self._cam_status = ("connecting...", False)
        self._asr_status = ("connecting...", False)
        self._fps_count  = 0
        self._fps_timer  = time.monotonic()

        self._cam_ws = CameraWSThread(f"{self.base_url}/ws/viewer",
                                      self.frame_q, self._on_status_update)
        self._ui_ws  = UIWSThread(f"{self.base_url}/ws_ui",
                                  self.msg_q, self._on_status_update)
        self._cam_ws.start()
        self._ui_ws.start()

        self._poll_frames()
        self._poll_messages()
        self._update_status_bar()

    # ── 建立 UI ──────────────────────────────────────────
    def _build_ui(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               bg=BG, sashwidth=6, relief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        # 左側：相機
        left = tk.Frame(paned, bg=BG)
        paned.add(left, minsize=400, width=640)
        cam_border = tk.Frame(left, bg=LINE)
        cam_border.pack(fill=tk.BOTH, expand=True)
        self._cam_label = tk.Label(cam_border, bg="#000", cursor="crosshair")
        self._cam_label.pack(fill=tk.BOTH, expand=True)

        # 右側：訊息
        right = tk.Frame(paned, bg=BG)
        paned.add(right, minsize=300)

        # 工具列
        bar = tk.Frame(right, bg=CARD, pady=4)
        bar.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Label(bar, text="語音 / AI 訊息", bg=CARD, fg=TEXT,
                 font=("Microsoft YaHei", 11, "bold")).pack(side=tk.LEFT, padx=8)
        tk.Button(bar, text="清空", bg="#1f2937", fg=MUTED,
                  activebackground="#2a3446", relief=tk.FLAT,
                  padx=10, pady=2,
                  command=self._clear_chat).pack(side=tk.RIGHT, padx=4)
        tk.Button(bar, text="重新連線", bg="#1f6feb", fg="#fff",
                  activebackground="#388bfd", relief=tk.FLAT,
                  padx=10, pady=2,
                  command=self._reconnect).pack(side=tk.RIGHT, padx=4)

        # Partial
        pf = tk.Frame(right, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        pf.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(pf, text="串流辨識（Partial）", bg=CARD, fg=MUTED,
                 font=("Microsoft YaHei", 9)).pack(anchor=tk.W, padx=8, pady=(4, 0))
        self._partial_var = tk.StringVar(value="（等待音频…）")
        tk.Label(pf, textvariable=self._partial_var,
                 bg=CARD, fg=TEXT,
                 font=("Microsoft YaHei", 13),
                 wraplength=400, justify=tk.LEFT,
                 anchor=tk.W).pack(fill=tk.X, padx=8, pady=(0, 6))

        # 對話框
        co = tk.Frame(right, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        co.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        tk.Label(co, text="最終文字（Final）", bg=CARD, fg=MUTED,
                 font=("Microsoft YaHei", 9)).pack(anchor=tk.W, padx=8, pady=(4, 0))
        tf = tk.Frame(co, bg=CARD)
        tf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        sc = tk.Scrollbar(tf)
        sc.pack(side=tk.RIGHT, fill=tk.Y)
        self._chat = tk.Text(
            tf, bg="#0b1020", fg=TEXT,
            font=("Microsoft YaHei", 12),
            wrap=tk.WORD, state=tk.DISABLED,
            yscrollcommand=sc.set,
            bd=0, relief=tk.FLAT, padx=10, pady=8,
            spacing1=4, spacing3=4,
        )
        self._chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sc.config(command=self._chat.yview)

        for tag, fg, bg, lm, rm in [
            ("user", "#fff",    USER_BUBBLE, 80, 10),
            ("ai",   "#e6edf3", AI_BUBBLE,   10, 80),
            ("nav",  "#a8d8a8", NAV_BUBBLE,  10, 80),
            ("sys",  "#c8a8d8", SYS_BUBBLE,  10, 80),
            ("find", "#d8c8a8", FIND_BUBBLE, 10, 80),
        ]:
            self._chat.tag_configure(tag, foreground=fg, background=bg,
                                     lmargin1=lm, lmargin2=lm, rmargin=rm,
                                     spacing1=3, spacing3=3)

        # 狀態列
        sb = tk.Frame(self.root, bg="#0a0d12", height=28)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        sb.pack_propagate(False)
        self._cam_lbl = tk.Label(sb, text="Camera: connecting…",
                                  bg="#0a0d12", fg=MUTED, font=("Consolas", 9))
        self._cam_lbl.pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(sb, text="│", bg="#0a0d12", fg=LINE).pack(side=tk.LEFT, padx=4)
        self._asr_lbl = tk.Label(sb, text="ASR: connecting…",
                                  bg="#0a0d12", fg=MUTED, font=("Consolas", 9))
        self._asr_lbl.pack(side=tk.LEFT)
        tk.Label(sb, text="│", bg="#0a0d12", fg=LINE).pack(side=tk.LEFT, padx=4)
        self._fps_lbl = tk.Label(sb, text="FPS: --",
                                  bg="#0a0d12", fg=MUTED, font=("Consolas", 9))
        self._fps_lbl.pack(side=tk.LEFT)
        tk.Label(sb, text=f"Server: {self.host}:{self.port}",
                 bg="#0a0d12", fg="#4d5f73",
                 font=("Consolas", 9)).pack(side=tk.RIGHT, padx=12)

    # ── 輪詢 ──────────────────────────────────────────────
    def _poll_frames(self):
        data = None
        try:
            while True:
                data = self.frame_q.get_nowait()
        except queue.Empty:
            pass
        if data is not None:
            if hasattr(self, '_no_signal_shown'):
                self._no_signal_shown = False
            try:
                img = Image.open(io.BytesIO(data))
                lw  = max(320, self._cam_label.winfo_width())
                lh  = max(240, self._cam_label.winfo_height())
                img.thumbnail((lw, lh), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._cam_label.configure(image=photo)
                self._cam_label.image = photo
                self._fps_count += 1
                now = time.monotonic()
                if now - self._fps_timer >= 1.0:
                    self._fps_lbl.config(text=f"FPS: {self._fps_count}")
                    self._fps_count = 0
                    self._fps_timer = now
            except Exception:
                pass
        else:
        # 無畫面：顯示等待提示（每秒只更新一次避免閃爍）
            now = time.monotonic()
            if not getattr(self, '_no_signal_shown', False) or \
                now - getattr(self, '_no_signal_ts', 0) > 1.0:
                self._no_signal_shown = True
                self._no_signal_ts = now
                self._cam_label.configure(
                    image="",
                    text="📡 等待 ESP32 相機連線…",
                    fg="#4d5f73",
                    font=("Microsoft YaHei", 14),
                    compound=tk.CENTER,
                )
                self._cam_label.image = None
                self._fps_lbl.config(text="FPS: --")
        self.root.after(16, self._poll_frames)

    def _poll_messages(self):
        try:
            while True:
                self._handle_msg(self.msg_q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(50, self._poll_messages)

    def _update_status_bar(self):
        ct, co = self._cam_status
        at, ao = self._asr_status
        self._cam_lbl.config(text=f"Camera: {ct}", fg=OK_COLOR if co else ERR_COLOR)
        self._asr_lbl.config(text=f"ASR: {at}",    fg=OK_COLOR if ao else ERR_COLOR)
        self.root.after(1000, self._update_status_bar)

    # ── 訊息處理 ──────────────────────────────────────────
    def _handle_msg(self, raw: str):
        if raw.startswith("INIT:"):
            try:
                import json
                d = json.loads(raw[5:])
                if d.get("partial"):
                    self._partial_var.set(d["partial"])
                for t in (d.get("finals") or []):
                    self._add_msg(t)
            except Exception:
                pass
        elif raw.startswith("PARTIAL:"):
            self._partial_var.set(raw[8:] or "（等待音频…）")
        elif raw.startswith("FINAL:"):
            self._add_msg(raw[6:])
            self._partial_var.set("（等待音频…）")

    def _add_msg(self, text: str):
        if text.startswith("[AI]"):
            self._append("🤖 AI：" + text[4:].strip(), "ai")
        elif text.startswith("[导航]"):
            cross = ["斑马线","绿灯","红灯","黄灯","过马路"]
            lbl = "🚦 斑马线" if any(k in text for k in cross) else "🦯 盲道"
            self._append(f"{lbl}：{text[4:].strip()}", "nav")
        elif text.startswith("[找物品]"):
            self._append("🔍 尋物：" + text[5:].strip(), "find")
        elif text.startswith("[系统]"):
            self._append("⚙ 系統：" + text[4:].strip(), "sys")
        else:
            self._append(text, "user")

    def _append(self, text: str, tag: str):
        self._chat.config(state=tk.NORMAL)
        lines = int(self._chat.index("end-1c").split(".")[0])
        if lines > MSG_MAX * 3:
            self._chat.delete("1.0", f"{MSG_MAX}l.0")
        self._chat.insert(tk.END, text + "\n", tag)
        self._chat.config(state=tk.DISABLED)
        self._chat.see(tk.END)

    # ── 狀態回呼 ──────────────────────────────────────────
    def _on_status_update(self, service, ok, msg):
        def _u():
            if service == "Camera": self._cam_status = (msg, ok)
            elif service == "ASR":  self._asr_status  = (msg, ok)
        self.root.after(0, _u)

    # ── 按鈕 ──────────────────────────────────────────────
    def _clear_chat(self):
        self._chat.config(state=tk.NORMAL)
        self._chat.delete("1.0", tk.END)
        self._chat.config(state=tk.DISABLED)

    def _reconnect(self):
        self._cam_ws.stop()
        self._ui_ws.stop()
        time.sleep(0.3)
        self._cam_ws = CameraWSThread(f"{self.base_url}/ws/viewer",
                                      self.frame_q, self._on_status_update)
        self._ui_ws  = UIWSThread(f"{self.base_url}/ws_ui",
                                  self.msg_q, self._on_status_update)
        self._cam_ws.start()
        self._ui_ws.start()

    def _on_close(self):
        self._cam_ws.stop()
        self._ui_ws.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ════════════════════════════════════════════════════════════
# 主程式
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="aiglass3 桌面視窗")
    parser.add_argument("--host", default=SERVER_HOST,
                        help="伺服器位址（預設 localhost）")
    parser.add_argument("--port", type=int, default=SERVER_PORT,
                        help="伺服器埠（預設 8081）")
    parser.add_argument("--no-server", action="store_true",
                        help="不啟動後端（後端已在執行時使用）")
    args = parser.parse_args()

    # 1. 啟動畫面
    splash = SplashScreen()

    # 2. 在背景啟動後端
    if not args.no_server:
        splash.set_message("正在啟動後端，AI 模型載入中（約 20～60 秒）…")
        _start_backend_thread(args.host, args.port)

    # 3. 等後端就緒
    # ── 關鍵修復：傳入 ui_update_fn 讓 splash 在等待期間不凍結 ──
    ready = _wait_for_backend(
        args.host, args.port,
        progress_cb=splash.set_message,
        ui_update_fn=lambda: splash.root.update(),
    )
    splash.close()

    if not ready:
        # 建立一個新的隱藏根視窗來顯示錯誤（舊的 splash 已被 destroy）
        _err_root = tk.Tk()
        _err_root.withdraw()
        messagebox.showerror(
            "aiglass3",
            f"後端啟動逾時（{HEALTH_TIMEOUT_S}s），請確認環境是否正確。\n"
            f"  - 若後端在遠端，請確認 {args.host}:{args.port} 可連線\n"
            f"  - 若在本機，請先確認 python main.py 沒有錯誤"
        )
        _err_root.destroy()
        sys.exit(1)

    # 4. 開啟主視窗
    win = AIGlassWindow(host=args.host, port=args.port)
    win.run()


if __name__ == "__main__":
    main()
