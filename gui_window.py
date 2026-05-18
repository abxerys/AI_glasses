# gui_window.py
# -*- coding: utf-8 -*-
import asyncio
import websockets
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SERVER_BROADCAST_URL = "ws://127.0.0.1:8765/ws/viewer"

class IntegratedGuiWindow:
    def __init__(self):
        # 畫布總尺寸設定為標準 720p (16:9)
        self.canvas_w = 1280
        self.canvas_h = 720
        self.cam_w = 960  # 相機區塊寬度 (映射 4:3 比例的完美大小)
        self.sidebar_w = self.canvas_w - self.cam_w
        
        self.logs = [
            "[系統] 核心整合主視窗啟動成功。",
            "[系統] 正在連接本地埠號 8765...",
            "[提示] 智慧眼鏡對話日誌將實時滾動顯示。"
        ]
        
        # 載入系統字型支援繁體中文
        try:
            self.font_title = ImageFont.truetype("msjh.ttc", 22)  # 微軟正黑體
            self.font_text = ImageFont.truetype("msjh.ttc", 16)
        except IOError:
            self.font_title = ImageFont.load_default()
            self.font_text = ImageFont.load_default()

    def add_dialogue_log(self, text: str):
        """新增一筆對話紀錄"""
        if text and (not self.logs or self.logs[-1] != text):
            self.logs.append(text)
            if len(self.logs) > 18:  # 限制最大行數
                self.logs.pop(0)

    def resize_with_pad(self, img, target_w, target_h):
        """等比例縮放影像並填充黑邊"""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 使用 INTER_AREA 確保縮小影像時的畫質
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 計算上下左右需要填充的黑邊寬度
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def construct_combined_interface(self, cv2_frame) -> np.ndarray:
        """拼接相機畫面與右側深色對話面板"""
        # 1. 建立整體標準 16:9 底圖 (深色主題)
        canvas = np.ones((self.canvas_h, self.canvas_w, 3), dtype=np.uint8) * 35
        
        # 2. 處理並貼上相機畫面
        cam_frame_resized = self.resize_with_pad(cv2_frame, self.cam_w, self.canvas_h)
        canvas[0:self.canvas_h, 0:self.cam_w] = cam_frame_resized
        
        # 3. 繪製左右區塊的分隔線
        cv2.line(canvas, (self.cam_w, 0), (self.cam_w, self.canvas_h), (80, 80, 80), 2)
        
        # 4. 轉為 PIL 處理文字與 UI 繪製
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 面板標題
        title_x = self.cam_w + 20
        draw.text((title_x, 25), "AI Glasses - 系統對話框", font=self.font_title, fill=(240, 240, 240))
        draw.line([(title_x, 60), (self.canvas_w - 20, 60)], fill=(100, 100, 100), width=2)
        
        # 對話歷史紀錄底框
        draw.rectangle([(self.cam_w + 10, 75), (self.canvas_w - 10, self.canvas_h - 40)], fill=(45, 45, 45), outline=(70, 70, 70))
        
        # 滾動渲染日誌
        y_offset = 90
        for log in self.logs:
            color = (100, 200, 255) if "[系統]" in log else (220, 220, 220)
            draw.text((self.cam_w + 25, y_offset), log, font=self.font_text, fill=color)
            y_offset += 30
            
        # 底部提示
        draw.text((title_x, self.canvas_h - 30), "操作提示: 於視窗內按下 'q' 鍵可關閉", font=self.font_text, fill=(130, 130, 130))
        
        # 5. 轉回 OpenCV BGR 格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

async def main_gui_loop():
    gui = IntegratedGuiWindow()
    print(f"[GUI] 正在連接至核心廣播端點: {SERVER_BROADCAST_URL} ...")
    
    while True:
        try:
            async with websockets.connect(SERVER_BROADCAST_URL) as ws:
                print("[GUI] 成功連接核心伺服器！已開啟 720p 整合視窗。")
                cv2.namedWindow("AI Glasses - Combined Terminal Window", cv2.WINDOW_NORMAL)
                gui.add_dialogue_log("[系統] 成功接入遠端影像流，即時同步中。")
                
                while True:
                    frame_bytes = await ws.recv()
                    
                    np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # 渲染新版標準化畫布
                        combined_ui = gui.construct_combined_interface(image)
                        cv2.imshow("AI Glasses - Combined Terminal Window", combined_ui)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[GUI] 使用者主動關閉視窗。")
                        return
                        
                    await asyncio.sleep(0.001)
                    
        except Exception as e:
            print(f"[GUI 異常斷線] 連線中斷: {e}")
            print("[GUI] 將在 3 秒後自動重試...")
            cv2.destroyAllWindows()
            await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(main_gui_loop())
    except KeyboardInterrupt:
        print("[GUI] 程式已終止。")
        cv2.destroyAllWindows()