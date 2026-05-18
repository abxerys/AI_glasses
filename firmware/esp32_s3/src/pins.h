#pragma once

#if defined(AIGLS_BOARD_XIAO_S3_SENSE)

// Seeed Studio XIAO ESP32S3 Sense (camera on-board OV2640)
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

// On-board PDM mic on the XIAO Sense
#define I2S_MIC_BCLK     -1
#define I2S_MIC_WS       42
#define I2S_MIC_DATA     41

// External I2S DAC / amp (e.g. MAX98357A) - adjust to your wiring
#define I2S_SPK_BCLK      7
#define I2S_SPK_LRC       8
#define I2S_SPK_DIN       9

#elif defined(AIGLS_BOARD_ESP32S3_CAM)

// ESP32-S3-CAM (Freenove / generic) — change to match your board's silkscreen
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

#define I2S_SPK_BCLK     38
#define I2S_SPK_LRC      39
#define I2S_SPK_DIN      40

#else
#error "Define an AIGLS_BOARD_* symbol in platformio.ini build_flags."
#endif
