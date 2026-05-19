#include "camera_stream.h"

#include <Arduino.h>
#include <esp_camera.h>

#include "pins.h"

bool camera_begin() {
  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0       = Y2_GPIO_NUM;
  cfg.pin_d1       = Y3_GPIO_NUM;
  cfg.pin_d2       = Y4_GPIO_NUM;
  cfg.pin_d3       = Y5_GPIO_NUM;
  cfg.pin_d4       = Y6_GPIO_NUM;
  cfg.pin_d5       = Y7_GPIO_NUM;
  cfg.pin_d6       = Y8_GPIO_NUM;
  cfg.pin_d7       = Y9_GPIO_NUM;
  cfg.pin_xclk     = XCLK_GPIO_NUM;
  cfg.pin_pclk     = PCLK_GPIO_NUM;
  cfg.pin_vsync    = VSYNC_GPIO_NUM;
  cfg.pin_href     = HREF_GPIO_NUM;
  cfg.pin_sccb_sda = SIOD_GPIO_NUM;
  cfg.pin_sccb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn     = PWDN_GPIO_NUM;
  cfg.pin_reset    = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_JPEG;
  cfg.frame_size   = FRAMESIZE_VGA;   // 640x480 — keep YOLO input small + fast
  cfg.jpeg_quality = 12;
  cfg.fb_count     = 2;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;
  cfg.fb_location  = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("camera init failed: 0x%x\n", err);
    return false;
  }

  // OV3660 orientation is corrected server-side (edge/server.py VIDEO_FLIP_CODE = -1).
  // Do not apply sensor-side flip here; it would double-flip and produce the
  // wrong orientation again.
#if defined(AIGLS_CAMERA_OV3660)
  sensor_t *s = esp_camera_sensor_get();
  if (s != nullptr) {
    s->set_brightness(s, 1);
    s->set_saturation(s, 0);
  }
#endif

  return true;
}

uint8_t *camera_capture_jpeg(size_t *out_len) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    *out_len = 0;
    return nullptr;
  }
  *out_len = fb->len;
  // Return the framebuffer's buffer; release() must give back the same fb.
  // We stash the fb pointer right before the buffer for release().
  // For simplicity, callers should call camera_release() with the same ptr.
  // Here we leak the fb* by stashing it via a static map of latest fb,
  // which is fine because we only ever hold one outstanding frame at a time.
  static camera_fb_t *latest = nullptr;
  if (latest) esp_camera_fb_return(latest);
  latest = fb;
  return fb->buf;
}

void camera_release(uint8_t * /*buf*/) {
  // No-op: camera_capture_jpeg manages the framebuffer lifecycle itself.
}
