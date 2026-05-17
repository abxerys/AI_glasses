import asyncio
import websockets
import cv2
import time
import os

WS_URL = "ws://127.0.0.1:8081/ws/camera"
VIDEO_PATH = r"videos\green_trafficlight.mp4"
TARGET_FPS = 5  

async def simulate_esp32_camera():
    if not os.path.exists(VIDEO_PATH):
        print(f"找不到影片檔案: {VIDEO_PATH}")
        return

    print(f"準備連線至 {WS_URL} ...")
    try:
        async with websockets.connect(WS_URL) as ws:
            print("連線成功！開始傳送預錄影片...")
            
            cap = cv2.VideoCapture(VIDEO_PATH)
            frame_delay = 1.0 / TARGET_FPS
            frame_count = 0

            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    print("影片播放完畢，循環播放...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
                success, enc_img = cv2.imencode('.jpg', frame, encode_param)
                
                if success:
                    await ws.send(enc_img.tobytes())
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"已傳送 {frame_count} 幀影像...")

                sleep_time = frame_delay - (time.time() - start_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

    except Exception as e:
        print(f"模擬測試發生錯誤: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        print("模擬結束。")

if __name__ == "__main__":
    asyncio.run(simulate_esp32_camera())