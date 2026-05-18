# ESP32-S3 Firmware

韌體採 **PlatformIO + Arduino framework**。第一階段只先做相機 JPEG 串流，
麥克風/喇叭等 I2S 音訊先在 PC 端用 [`tools/fake_esp32.py`](../../tools/fake_esp32.py) 模擬，
韌體中 [src/audio_io.cpp](src/audio_io.cpp) 為 stub，等實際麥克風與功放接好再啟用。

## 已支援的板子

| `default_envs` | 板子 |
|---|---|
| `xiao_esp32s3_sense` | Seeed Studio XIAO ESP32S3 Sense（內建 OV2640 與 PDM 麥克風） |
| `esp32s3_cam` | 通用 ESP32-S3-CAM / Freenove 模組（pin map 需依板子矽印更動 [src/pins.h](src/pins.h)） |

選擇方式：直接修改 [platformio.ini](platformio.ini) 的 `default_envs`，或 `pio run -e <env>`。

## 燒錄步驟

1. 安裝 [PlatformIO Core](https://platformio.org/install/cli)。
2. 複製 Wi-Fi 設定樣板：
   ```bash
   cp src/wifi_config.h.example src/wifi_config.h
   ```
   編輯 `WIFI_SSID`、`WIFI_PASSWORD`，以及 `EDGE_HOST`（執行 `edge.main` 那台電腦的 LAN IP）。
3. 接上板子並燒錄：
   ```bash
   pio run -t upload
   pio device monitor
   ```
4. Serial 應該看到 `[wifi] ip=...` 與 `[video] ws connected`，PC 端 `edge.main` 的 log
   會收到 `ws connect path=/ws/video`，並開始進行 YOLO 推論。

## 麥克風 / 喇叭硬體建議

- 麥克風：INMP441（I2S）或板載 PDM（XIAO Sense）。
- 喇叭/功放：MAX98357A I2S Class-D 功放搭 8Ω 小喇叭。
- 接腳請依實際 wiring 更新 [src/pins.h](src/pins.h)。
- I2S 初始化會於 stage 1.5 加入；目前 [src/audio_io.cpp](src/audio_io.cpp) 不會驅動任何 I2S，
  避免在錯誤腳位上輸出造成功放損壞。
