#include <stdint.h>

namespace audio_io {

bool initAudioInput() {
  // TODO: Initialize microphone input (I2S/PDM).
  return true;
}

bool initAudioOutput() {
  // TODO: Initialize speaker output path.
  return true;
}

int readAudioFrame(int16_t* buffer, int length) {
  // TODO: Fill buffer from microphone.
  (void)buffer;
  return length;
}

void playAudioFrame(const int16_t* buffer, int length) {
  // TODO: Send PCM data to DAC/amplifier.
  (void)buffer;
  (void)length;
}

}  // namespace audio_io
