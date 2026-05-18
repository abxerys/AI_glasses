// Audio I/O for ESP32-S3.
//
// **Stage 1 status: stub.** This file intentionally does NOT initialise I2S yet.
// The mic and amp pin map varies per board (see pins.h) and an incorrect map
// can damage the amplifier. We open the audio WS channels only after a known-
// good mic + amp wiring is confirmed in hardware.
//
// Planned implementation (stage 1.5):
//   - Bring up I2S RX (mic, e.g. INMP441 or on-board PDM): 16 kHz, 16-bit, mono.
//     Every ~250 ms send accumulated PCM16 via ws://EDGE_HOST:EDGE_PORT/ws/audio_in.
//   - Bring up I2S TX (e.g. MAX98357A): receive MP3 bytes on /ws/audio_out,
//     decode with ESP8266Audio's AudioGeneratorMP3, output to AudioOutputI2S.
//
// Until that lands, the PC-side STT input must come from `tools/fake_esp32.py`
// (laptop mic), and the PC plays TTS locally instead of streaming it back.

#include <Arduino.h>

void audio_io_begin() {
  // intentionally empty in stage 1
}

void audio_io_loop() {
  // intentionally empty in stage 1
}
