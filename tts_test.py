import os
from openai import OpenAI

GROQ_API_KEY = ""

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def analyze_intent_with_llm(user_text):
    print(f"[LLM] 正在分析使用者指令: {user_text}")
    
    # 這裡的 System Prompt 是導航眼鏡的靈魂！
    completion = client.chat.completions.create(
        model="llama3-8b-8192", # 或是 gpt-3.5-turbo
        temperature=0.3,        # 溫度調低一點，讓 AI 的回答簡潔精準
        messages=[
            {
                "role": "system",
                "content": """你是一個穿戴式盲人導航眼鏡的 AI 助理。
                請簡短地回覆使用者，並在句尾附上要給系統的執行指令。
                可用的指令代碼有：
                - [CMD: YOLO_CROSSWALK] (使用者想找斑馬線)
                - [CMD: YOLO_TRAFFIC_LIGHT] (使用者想看紅綠燈)
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

# --- 測試看看 ---
# text = "前面有斑馬線嗎"
# result = analyze_intent_with_llm(text)