import os
from openai import OpenAI

GROQ_API_KEY 

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def test_groq_audio(audio_file_path):
    if not os.path.exists(audio_file_path):
        print(f"找不到檔案: {audio_file_path}")
        return None # 找不到檔案就回傳空值
        
    print(f"正在上傳音檔給 Groq 處理: {audio_file_path}...")
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3", 
                response_format="text", # 因為設定了這個，回傳的直接就是純文字！
                language="zh",  
                prompt="這是一段測試導航眼鏡的語音指令，可能包含斑馬線、紅綠燈、盲道、尋找物品、過馬路等，請只要根據錄音檔的內容輸出文字即可。",
                temperature=0.7
            )
            
        print("-" * 30)
        print(f"辨識結果: {transcription}")
        print("-" * 30)
        
        # 🌟 關鍵修改一：把辨識出來的字串回傳出去！
        return transcription
        
    except Exception as e:
        print(f"發生錯誤: {e}")
        return None

# ==========================================
# 階段二：意圖分析 (LLM大腦)
# ==========================================
def analyze_intent_with_llm(user_text):
    print(f"\n[LLM] 正在分析使用者指令: {user_text}")
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # 🌟 修改 1：換上 Groq 最新支援的模型
        temperature=0.3,        
        messages=[
            {
                "role": "system",
                "content": """你是一個穿戴式盲人導航眼鏡的 AI 助理。
                請簡短地回覆使用者，並在句尾附上要給系統的執行指令。
                可用的指令代碼有：
                - [CMD: YOLO_CROSSWALK] (使用者想找斑馬線)
                - [CMD: YOLO_TRAFFIC_LIGHT] (使用者想看紅綠燈)
                - [CMD: YOLO_BLIND_PATH] (使用者想找盲道或導盲磚)  # 🌟 修改 2：教它聽懂盲磚！
                - [CMD: NONE] (只是普通聊天)
                
                範例回覆：「好的，馬上為您尋找斑馬線。[CMD: YOLO_CROSSWALK]」"""
            },
            {
                "role": "user",
                "content": user_text
            }
        ]
    )
    
    response = completion.choices[0].message.content
    print(f"[LLM] AI 回覆: {response}")
    return response
# ==========================================
# 🚀 主程式：將 STT 與 LLM 完美串接！
# ==========================================
if __name__ == "__main__":
    # 設定你要測試的音檔路徑
    audio_path = r"recordings\crossing1.wav"
    
    # 🌟 關鍵修改二：把階段一回傳的文字，抓出來存進 user_text
    user_text = test_groq_audio(audio_path)
    
    # 檢查是否真的有辨識出文字 (避免音檔空白或報錯)
    if user_text:
        # 把抓到的 user_text 丟給階段二的大腦！
        analyze_intent_with_llm(user_text)
    else:
        print("沒有辨識出文字，停止分析。")