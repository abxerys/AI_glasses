import asyncio
import websockets

# 伺服器的音訊 WebSocket 位址
WS_AUDIO_URL = "ws://127.0.0.1:8081/ws_audio"

async def send_command():
    print(f"準備連線至 {WS_AUDIO_URL} ...")
    try:
        async with websockets.connect(WS_AUDIO_URL) as ws:
            print("連線成功！這是一個模擬麥克風的文字輸入器。")
            print("請輸入你要測試的指令 (例如：我要過馬路)，輸入 q 離開：")
            
            while True:
                # 這裡使用內建的 input() 讓你在終端機打字
                cmd = input(">> ")
                if cmd.lower() == 'q':
                    break
                if cmd.strip():
                    # 加上 PROMPT: 前綴，欺騙伺服器這是語音辨識結果
                    await ws.send(f"PROMPT:{cmd}")
                    print(f"已發送模擬語音指令: {cmd}")
                    
                    # 接收伺服器回傳的確認訊息 (OK:PROMPT_ACCEPTED)
                    response = await ws.recv()
                    print(f"伺服器回應: {response}")
                    
    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(send_command())