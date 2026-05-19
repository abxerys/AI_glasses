// AI Glasses — ESP32-S3 firmware entry point.
//
// Responsibilities for stage 1 (this build):
//   1. Connect to Wi-Fi.
//   2. Open a WebSocket client to the PC edge runtime at ws://EDGE_HOST:EDGE_PORT/ws/video
//      and stream OV2640 JPEG frames continuously.
//   3. (Stubbed) audio I/O — see audio_io.cpp. Implementation depends on the chosen
//      I2S mic + amp; the current file documents the wiring contract but does not
//      yet initialise I2S to avoid bricking boards with the wrong pin map.
//
// When audio is added, two more WebSocket clients will connect to /ws/audio_in
// (mic out -> PC) and /ws/audio_out (PC -> speaker).

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>

#include "audio_io.h"
#include "camera_stream.h"
#include "wifi_config.h"

static WebSocketsClient gVideoWs;
static bool gVideoConnected = false;

static void onVideoEvent(WStype_t type, uint8_t * /*payload*/, size_t /*len*/) {
  switch (type) {
    case WStype_CONNECTED:
      gVideoConnected = true;
      Serial.println("[video] ws connected");
      break;
    case WStype_DISCONNECTED:
      gVideoConnected = false;
      Serial.println("[video] ws disconnected");
      break;
    default:
      break;
  }
}

static void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[wifi] connecting to %s", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print('.');
  }
  Serial.printf("\n[wifi] ip=%s\n", WiFi.localIP().toString().c_str());
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n=== AI Glasses ESP32-S3 boot ===");

  if (!camera_begin()) {
    Serial.println("camera init failed; halting");
    while (true) delay(1000);
  }

  connectWifi();

  Serial.printf("[edge] connecting ws://%s:%d/ws/video\n", EDGE_HOST, EDGE_PORT);
  Serial.println("[edge] if this hangs, check:");
  Serial.println("       1. python -m edge.main is running on the PC");
  Serial.println("       2. EDGE_HOST in wifi_config.h matches PC's LAN IPv4");
  Serial.println("       3. Windows Defender Firewall allows inbound TCP 8765");

  gVideoWs.begin(EDGE_HOST, EDGE_PORT, "/ws/video");
  gVideoWs.onEvent(onVideoEvent);
  gVideoWs.setReconnectInterval(2000);

  audio_io_begin(EDGE_HOST, EDGE_PORT);
}

static unsigned long gLastFrameMs = 0;
static const unsigned long kFramePeriodMs = 100;  // 10 fps

void loop() {
  gVideoWs.loop();
  audio_io_loop();

  unsigned long now = millis();
  if (gVideoConnected && (now - gLastFrameMs) >= kFramePeriodMs) {
    gLastFrameMs = now;
    size_t len = 0;
    uint8_t *buf = camera_capture_jpeg(&len);
    if (buf && len > 0) {
      gVideoWs.sendBIN(buf, len);
    }
  }
}
