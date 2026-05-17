import asyncio
import websockets
import cv2
import time

async def stream_video(video_path, ws_url):
    # 打開影片檔案
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法打開影片檔案 {video_path}")
        return

    # 計算影片的原始幀率，用來控制發送速度
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 0.05

    # 連接至後端 WebSocket
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"✅ 已成功連接到伺服器 {ws_url}，開始推送影片...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("🔄 影片播放完畢，重新從頭播放...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # 壓縮成 JPEG (品質設為 50，模擬 ESP32 的畫質)
                success, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if success:
                    # 將二進位數據送出
                    await websocket.send(encoded_image.tobytes())
                
                # 模擬相機的拍攝間隔
                await asyncio.sleep(delay)
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ 連線已關閉: {e}")
    except ConnectionRefusedError:
        print(f"❌ 無法連線至伺服器，請確認 app_main.py 是否已啟動。")

if __name__ == "__main__":
    # 參數設定
    # 1. 請確認 videos 資料夾下有這支影片，或更改為您的影片檔名
    VIDEO_FILE = "videos/blind_path.mp4" 
    
    # 2. 如果伺服器跑在同一台電腦，使用 127.0.0.1 即可
    SERVER_WS_URL = "ws://127.0.0.1:8081/ws/camera"
    
    asyncio.run(stream_video(VIDEO_FILE, SERVER_WS_URL))