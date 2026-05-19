#pragma once
#include <stddef.h>
#include <stdint.h>

// Initialise OV2640 with sensible defaults for streaming
// (VGA, JPEG quality ~12). Returns true on success.
bool camera_begin();

// Capture one JPEG frame. Caller MUST call camera_release(buf) after sending.
// out_len receives the JPEG buffer length in bytes.
uint8_t *camera_capture_jpeg(size_t *out_len);
void camera_release(uint8_t *buf);
