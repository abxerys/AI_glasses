#pragma once

// ─────────────────────────────────────────────────────────────────────
// XIAO ESP32-S3 family (Sense and non-Sense) — camera FPC connector
// pinout is identical on both. OV2640 and OV3660 use the same connector.
// ─────────────────────────────────────────────────────────────────────
#if defined(AIGLS_BOARD_XIAO_S3)

#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    10
#define SIOD_GPIO_NUM    40
#define SIOC_GPIO_NUM    39
#define Y9_GPIO_NUM      48
#define Y8_GPIO_NUM      11
#define Y7_GPIO_NUM      12
#define Y6_GPIO_NUM      14
#define Y5_GPIO_NUM      16
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM      17
#define Y2_GPIO_NUM      15
#define VSYNC_GPIO_NUM   38
#define HREF_GPIO_NUM    47
#define PCLK_GPIO_NUM    13

// On-board PDM mic on the XIAO Sense (only on Sense; ignored on base S3)
#define I2S_MIC_BCLK     -1
#define I2S_MIC_WS       42
#define I2S_MIC_DATA     41

// MAX98357A I2S Class-D amp wiring (XIAO ESP32-S3 base).
// These are the recommended defaults — verify against your actual wiring.
//
//   MAX98357A pin  ──→  XIAO pad   GPIO
//   LRC (WS)            D0         GPIO 1
//   BCLK                D1         GPIO 2
//   DIN                 D2         GPIO 3
//   GAIN                — leave floating (= 9 dB) or tie to GND (= 12 dB)
//   SD                  — tie to 3V3 (always on) or via a GPIO to mute
//   VIN                 — 5 V or 3V3
//   GND                 — GND
#define I2S_SPK_LRC       1
#define I2S_SPK_BCLK      2
#define I2S_SPK_DIN       3

// ─────────────────────────────────────────────────────────────────────
// ESP32-S3-CAM (Freenove / generic)
// ─────────────────────────────────────────────────────────────────────
#elif defined(AIGLS_BOARD_ESP32S3_CAM)

#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM      16
#define Y8_GPIO_NUM      17
#define Y7_GPIO_NUM      18
#define Y6_GPIO_NUM      12
#define Y5_GPIO_NUM      10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM      11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM    13

#define I2S_MIC_BCLK      1
#define I2S_MIC_WS        2
#define I2S_MIC_DATA      3

#define I2S_SPK_LRC      38
#define I2S_SPK_BCLK     39
#define I2S_SPK_DIN      40

#else
#error "Define an AIGLS_BOARD_* symbol in platformio.ini build_flags."
#endif
