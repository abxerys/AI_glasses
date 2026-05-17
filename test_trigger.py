import asyncio
import websockets

async def trigger_test():
    uri = "ws://localhost:8081/ws_audio"  # 請確認這是不是 app_main.py 監聽的 port
    try:
        async with websockets.connect(uri) as ws:
            print("1. 連線成功，發送 START...")
            await ws.send("START")
            await asyncio.sleep(2) # 假裝錄了兩秒的空白音
            
            print("2. 發送 STOP...")
            await ws.send("STOP")
            
            # 聽一下伺服器回傳了什麼
            response = await ws.recv()
            print(f"3. 收到回覆: {response}")
            
    except Exception as e:
        print(f"連線失敗: {e}")

if __name__ == "__main__":
    asyncio.run(trigger_test())