# ESP32-S3 Firmware

韌體採 **PlatformIO + Arduino framework**。Stage 1 重點是把鏡頭 JPEG 串到 PC；
喇叭 I2S 已預留 pin map 但播放邏輯尚未實作（語音播報暫由筆電喇叭出聲）。

## 支援的 build 目標

| `default_envs` | 板 / 鏡頭 / 音訊 |
|---|---|
| `xiao_esp32s3_ov3660` | **Seeed XIAO ESP32-S3（基本版）+ OV3660 + MAX98357A**（本專案參考硬體） |
| `xiao_esp32s3_sense` | Seeed XIAO ESP32-S3 **Sense**（板上 OV2640 + 板上 PDM 麥克風） |
| `esp32s3_cam` | 通用 ESP32-S3-CAM / Freenove |

選擇方式：直接修改 [platformio.ini](platformio.ini) 的 `default_envs`，或執行 `pio run -e <env>`。

## 接線（XIAO ESP32-S3 + OV3660 + MAX98357A）

OV3660 鏡頭接 XIAO 的 FPC 接頭（卡進去即可，不需要焊）。

MAX98357A → XIAO 預設腳位（可在 [src/pins.h](src/pins.h) 改）：

| MAX98357A | XIAO 標號 | GPIO |
|---|---|---|
| LRC (WS) | D0 | GPIO 1 |
| BCLK | D1 | GPIO 2 |
| DIN | D2 | GPIO 3 |
| GAIN | 浮空（= 9 dB）或接 GND（= 12 dB） | — |
| SD | 接 3V3（永遠不靜音） | — |
| VIN | 5V 或 3V3 | — |
| GND | GND | — |

⚠️ 如果您實際焊在不同腳位，請依實際線路修改 [src/pins.h](src/pins.h) 的 `I2S_SPK_*` 三個值。

## 燒錄步驟

1. 安裝 [PlatformIO Core](https://platformio.org/install/cli) 或裝 VS Code 的 PlatformIO IDE extension。

2. 設定 Wi-Fi 與 server 位址：
   ```bash
   cp src/wifi_config.h.example src/wifi_config.h
   ```
   編輯 `src/wifi_config.h`：
   ```c
   #define WIFI_SSID     "你家的WiFi"
   #define WIFI_PASSWORD "WiFi密碼"
   #define EDGE_HOST     "192.168.0.XX"   // 跑 edge.main 那台 PC 的 LAN IP
   #define EDGE_PORT     8765
   ```

   找出 PC 的 LAN IP：Windows 用 `ipconfig`，看「IPv4 位址」開頭 192.168.x.x 那個。

3. 用 USB-C 線接 XIAO ESP32-S3，從 PC 端跑：
   ```bash
   cd firmware/esp32_s3
   pio run -t upload
   pio device monitor
   ```

4. Serial monitor 上應該看到：
   ```
   === AI Glasses ESP32-S3 boot ===
   [wifi] connecting to <SSID> .....
   [wifi] ip=192.168.0.YY
   [video] ws connected
   ```

5. PC 端（同時跑 `python -m edge.main` 的那個視窗）會看到：
   ```
   ws connect path=/ws/video peer=('192.168.0.YY', xxxxx)
   ```

## 同時使用筆電麥克風 + ESP32 鏡頭

ESP32 韌體目前只負責影像。麥克風與喇叭仍走筆電：

```powershell
python tools\fake_esp32.py --no-video --preview
```

`--no-video` 告訴 fake_esp32.py 不要再從筆電 webcam 推影像（影像由真正的 ESP32 送）。
`--preview` 仍然有用嗎？沒有用了（因為 fake_esp32 不再開鏡頭），可以拿掉。

完整指令：

```powershell
# 視窗 1
python -m edge.main

# 視窗 2 — 只跑麥克風 + 喇叭
python tools\fake_esp32.py --no-video
```

## PC 端防火牆

Windows Defender 第一次接到從 ESP32 過來的 WS 連線，會跳出「允許/拒絕」對話框。**允許「私人網路」存取**即可。

如果視窗 1 的 server log 完全沒看到 ESP32 的連線（卻看到 ESP32 serial monitor 一直在重連），多半是防火牆擋住了 8765 port，到「進階防火牆設定 → 輸入規則」加一條允許 8765 即可。

## 音訊輸出（之後做）

目前 [src/audio_io.cpp](src/audio_io.cpp) 是 stub。要讓 ESP32 喇叭播語音，後續要做：

1. 開一條 `/ws/audio_out` WebSocket client（PC → ESP32）
2. 接到 MP3 / WAV bytes 後，用 `ESP8266Audio` 或自行解碼到 PCM
3. 用 `i2s_write()` 推到 MAX98357A

或更簡單：請 PC 端先把 TTS 解成固定取樣率的 PCM16 再送，ESP32 只做 I2S write。
