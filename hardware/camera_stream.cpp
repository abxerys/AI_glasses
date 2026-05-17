#include <WiFi.h>

namespace camera_stream {

bool initCamera() {
  // TODO: Initialize ESP32-S3 camera sensor and frame buffers.
  return true;
}

bool connectWiFi(const char* ssid, const char* password) {
  WiFi.begin(ssid, password);
  // TODO: Add retry and timeout handling for production.
  return true;
}

void startVideoStream(const char* host, uint16_t port) {
  // TODO: Encode camera frames and stream over HTTP/WebSocket.
  (void)host;
  (void)port;
}

}  // namespace camera_stream
