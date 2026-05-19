// Audio I/O for ESP32-S3.
//
// Mic path:    on-board PDM microphone (XIAO ESP32-S3 Sense)
//              → I2S RX in PDM mode @ 16 kHz, 16-bit, mono
//              → /ws/audio_in WebSocket (raw PCM16 little-endian)
//
// Speaker path: /ws/audio_out WebSocket
//              → PCM16 mono @ 16 kHz (server already decoded MP3/WAV for us)
//              → I2S TX → MAX98357A Class-D amp
//
// The PC sends a fixed PCM16 16 kHz mono format, so we never need an MP3
// decoder on the device — just write bytes straight to the I2S TX queue.

#include "audio_io.h"

#include <Arduino.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>

#include "pins.h"

namespace {

constexpr int kSampleRate    = 16000;
constexpr int kMicChunkMs    = 250;
constexpr int kMicChunkBytes = (kSampleRate * kMicChunkMs / 1000) * sizeof(int16_t);

constexpr i2s_port_t kMicPort = I2S_NUM_0;
constexpr i2s_port_t kSpkPort = I2S_NUM_1;

WebSocketsClient gAudioInWs;
WebSocketsClient gAudioOutWs;
bool gAudioInOpen  = false;
bool gAudioOutOpen = false;

void onAudioInEvent(WStype_t type, uint8_t *, size_t) {
  if (type == WStype_CONNECTED) { gAudioInOpen = true;  Serial.println("[audio_in]  ws connected"); }
  if (type == WStype_DISCONNECTED) { gAudioInOpen = false; Serial.println("[audio_in]  ws disconnected"); }
}

void onAudioOutEvent(WStype_t type, uint8_t *payload, size_t len) {
  switch (type) {
    case WStype_CONNECTED:
      gAudioOutOpen = true;
      Serial.println("[audio_out] ws connected");
      break;
    case WStype_DISCONNECTED:
      gAudioOutOpen = false;
      Serial.println("[audio_out] ws disconnected");
      break;
    case WStype_BIN: {
      // Each binary frame is PCM16 mono at kSampleRate. Stream straight
      // into the I2S TX DMA buffers.
      size_t written = 0;
      i2s_write(kSpkPort, payload, len, &written, portMAX_DELAY);
      break;
    }
    default:
      break;
  }
}

#if defined(AIGLS_HAS_PDM_MIC)
bool initMicPDM() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
  cfg.sample_rate = kSampleRate;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 4;
  cfg.dma_buf_len = 1024;
  cfg.use_apll = false;

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = I2S_PIN_NO_CHANGE;          // PDM has no BCLK
  pins.ws_io_num  = I2S_MIC_WS;
  pins.data_in_num  = I2S_MIC_DATA;
  pins.data_out_num = I2S_PIN_NO_CHANGE;

  if (i2s_driver_install(kMicPort, &cfg, 0, nullptr) != ESP_OK) return false;
  if (i2s_set_pin(kMicPort, &pins) != ESP_OK) return false;
  return true;
}
#endif

#if defined(AIGLS_HAS_I2S_SPEAKER)
bool initSpeaker() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = kSampleRate;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 512;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = true;

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = I2S_SPK_BCLK;
  pins.ws_io_num  = I2S_SPK_LRC;
  pins.data_out_num = I2S_SPK_DIN;
  pins.data_in_num  = I2S_PIN_NO_CHANGE;

  if (i2s_driver_install(kSpkPort, &cfg, 0, nullptr) != ESP_OK) return false;
  if (i2s_set_pin(kSpkPort, &pins) != ESP_OK) return false;
  i2s_zero_dma_buffer(kSpkPort);
  return true;
}
#endif

// Background task: read mic chunks, push over WebSocket.
void micTask(void *) {
#if defined(AIGLS_HAS_PDM_MIC)
  static uint8_t buf[kMicChunkBytes];
  while (true) {
    size_t read = 0;
    esp_err_t err = i2s_read(kMicPort, buf, sizeof(buf), &read, pdMS_TO_TICKS(500));
    if (err != ESP_OK || read == 0) continue;
    if (gAudioInOpen) {
      gAudioInWs.sendBIN(buf, read);
    }
  }
#else
  vTaskDelete(nullptr);
#endif
}

bool gStarted = false;

}  // namespace

void audio_io_begin(const char *host, uint16_t port) {
  if (gStarted) return;
  gStarted = true;

#if defined(AIGLS_HAS_PDM_MIC)
  if (initMicPDM()) {
    Serial.println("[audio] PDM mic initialised");
    xTaskCreatePinnedToCore(micTask, "audio_mic", 4096, nullptr, 5, nullptr, 1);
  } else {
    Serial.println("[audio] PDM mic init FAILED");
  }
#else
  Serial.println("[audio] no mic configured for this board");
#endif

#if defined(AIGLS_HAS_I2S_SPEAKER)
  if (initSpeaker()) {
    Serial.println("[audio] I2S speaker initialised");
  } else {
    Serial.println("[audio] I2S speaker init FAILED");
  }
#else
  Serial.println("[audio] no speaker configured for this board");
#endif

  gAudioInWs.begin(host, port, "/ws/audio_in");
  gAudioInWs.onEvent(onAudioInEvent);
  gAudioInWs.setReconnectInterval(2000);

  gAudioOutWs.begin(host, port, "/ws/audio_out");
  gAudioOutWs.onEvent(onAudioOutEvent);
  gAudioOutWs.setReconnectInterval(2000);
}

void audio_io_loop() {
  gAudioInWs.loop();
  gAudioOutWs.loop();
}
