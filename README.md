# AI Glasses

低成本 AI 智慧輔助眼鏡。第一階段實作三個核心功能：

1. **語音搜尋物品** — 對麥克風說「找手機」，YOLO 偵測物品後給方向引導；當物品被定位
   且使用者靠近時切到第二階段，用 MediaPipe 手部追蹤指引「手再往左/右/上/下」直到對準。
2. **安全過街輔助** — `trafficlight.pt` 判讀紅綠燈狀態，`yolo-seg.pt` 偵測斑馬線
   位置，語音播報「紅燈請等候 / 綠燈可前進 / 向左修正 / 向右修正 / 已通過斑馬線」。
3. **盲道導航** — 同樣使用 `yolo-seg.pt`（含盲道類別），語音播報「方向正確請直行 /
   向左平移 / 向右平移 / 丟失路徑」。

## 系統架構

```
ESP32-S3 (Camera/Mic/Speaker)  ──WebSocket──▶  PC Python edge runtime
                                                ├─ shoppingbest5.pt      (item det)
                                                ├─ trafficlight.pt       (red/green)
                                                ├─ yolo-seg.pt           (crosswalk + 盲道)
                                                ├─ hand_landmarker.task  (MediaPipe)
                                                ├─ faster-whisper STT (zh-TW)
                                                ├─ edge-tts + pre-recorded WAV
                                                └─ asyncio state machine
```

- `/ws/video`     ESP32 → PC，JPEG frames
- `/ws/audio_in`  ESP32 → PC，PCM16 16 kHz mono
- `/ws/audio_out` PC → ESP32，WAV / MP3（語音播報）

## 目錄

| 路徑 | 內容 |
|---|---|
| [`edge/`](edge/) | Python 邊緣運算（WS server、YOLO、MediaPipe、STT/TTS、狀態機、模式） |
| [`tools/`](tools/) | 開發工具：`fake_esp32.py`、`video_replay.py`、`inspect_model.py` |
| [`firmware/esp32_s3/`](firmware/esp32_s3/) | ESP32-S3 韌體（PlatformIO + Arduino） |
| [`models/`](models/) | 模型權重（不入 git；上傳清單見下） |
| [`assets/voice/`](assets/voice/) | 預錄語音檔（不入 git）與 `map.zh-CN.json` 對照表 |

## 安裝

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Linux / Codespaces：補裝 OpenGL 系統 lib

`ultralytics` 與 `mediapipe` 透過 `cv2` 載入時會 link 到 OpenGL，
Ubuntu base image 預設沒裝。**第一次設定**請執行：

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libgles2-mesa-dev libegl1
```

不裝會看到 `ImportError: libGL.so.1` 或 `libGLESv2.so.2`。
Windows / macOS 不會遇到這問題。

### 需要上傳的模型檔（放到 [models/](models/)）

| 檔名 | 用途 | 必要性 |
|---|---|---|
| `shoppingbest5.pt` | 物品辨識（**目前只含 `AD鈣奶`、`紅牛` 兩類**） | ✅ 語音搜尋必須 |
| `trafficlight.pt` | 紅綠燈狀態 | ✅ 過街必須 |
| `yolo-seg.pt` | 斑馬線 + 盲道分割（同一個檔同時服務兩個模式） | ✅ 過街 / 盲道導航必須 |
| `hand_landmarker.task` | MediaPipe 手部關鍵點 | ✅ 語音搜尋第二階段「手部對焦」 |
| `yolov8s.pt` | COCO 80 類（備援） | 🟡 沒有 `shoppingbest5.pt` 時自動使用 |

**不需要上傳**：`yoloe-11l-seg.pt`（開放詞彙，留待後續進階功能）。

### 語音資產（放到 [assets/voice/](assets/voice/)）

把您現有的所有 `.wav` 與 `map.zh-CN.json` 整包複製到 [assets/voice/](assets/voice/)。
程式會優先用預錄音檔，找不到對應 key 時才用 edge-tts 即時合成（fallback 文字以繁體中文）。

## 跑起來（不需 ESP32 也能 demo）

> ⚠️ **這一節必須在本機電腦執行，不能在 Codespaces / 雲端 IDE**
> Codespaces 沒有 webcam 也沒有麥克風，`fake_esp32.py` 抓不到輸入裝置。
> Codespaces 適合做「載模型驗證類別、跑單元測試、改程式」，
> 端到端 demo 請在 Windows / macOS / Linux 桌機操作。

開兩個終端：

**Terminal 1 — 邊緣 server：**
```bash
python -m edge.main
```

**Terminal 2 — 用筆電假裝 ESP32：**
```bash
python tools/fake_esp32.py
```

對著筆電麥克風說：
- 「找紅牛」 / 「找鈣奶」 → 進入語音搜尋（含手部對焦）
  - 註：目前 `shoppingbest5.pt` 只訓練了 AD 鈣奶與紅牛兩種飲料；若改用 `yolov8s.pt`
    （COCO 80 類）則可用「找手機 / 找杯子 / 找瓶子」等通用詞
- 「過馬路」 / 「斑馬線」 → 進入過街輔助
- 「盲道」 / 「盲道導航」 → 進入盲道導航
- 「停」 → 退出當前模式

## 跑測試

```bash
python -m pytest edge/tests/ -v
```

## 模型類別命名相容性

各模型的類別名稱由 .pt 本身內嵌（`model.names`）。系統會以**別名 list** 對應到語意角色：

- 紅綠燈：`red` / `red_light` / `traffic_light_red` / `紅燈` / ... → red
- 斑馬線：`crosswalk` / `zebra` / `zebra_crossing` / `斑馬線` / ...
- 盲道：  `blind_path` / `tactile_paving` / `paving` / `盲道` / `導盲` / ...

如果您的模型用了其他命名，可在 [edge/modes/crossing.py](edge/modes/crossing.py) 與
[edge/modes/blind_path.py](edge/modes/blind_path.py) 的 `_ALIASES` 處追加；不必重訓。

確認類別實際名稱：
```bash
python tools/inspect_model.py models/yolo-seg.pt
python tools/inspect_model.py models/trafficlight.pt
```

## 不在第一階段範圍

語音導航（Google Maps API）、跌倒偵測、即時翻譯、雲端 Gemini 串接、
ESP32 端 I2S 麥克風與喇叭驅動（韌體目前僅做相機串流；音訊先用
[`tools/fake_esp32.py`](tools/fake_esp32.py) 模擬）。
