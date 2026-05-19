#pragma once

#include <Arduino.h>
#include <WebSocketsClient.h>

// Initialise I2S (mic + speaker) and open two WebSocket clients to the PC:
//   /ws/audio_in   — ESP32 → PC (PDM mic, PCM16 mono 16 kHz)
//   /ws/audio_out  — PC → ESP32 (PCM16 mono at AUDIO_OUT_SR)
// Safe to call once from setup(); subsequent calls are no-ops.
void audio_io_begin(const char *host, uint16_t port);

// Service the WebSocket clients. Must be called frequently from loop().
void audio_io_loop();
