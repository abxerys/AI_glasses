# qwen_extractor.py (Gemini REST API 版)
# -*- coding: utf-8 -*-
from typing import List, Tuple
import os
import requests
import json

# —— 本地优先映射（保留原作者的設定）——
LOCAL_CN2EN = {
    "红牛": "Red_Bull",
    "ad钙奶": "AD_milk",
    "ad 钙奶": "AD_milk",
    "ad": "AD_milk",
    "钙奶": "AD_milk",
    "矿泉水": "bottle",
    "水瓶": "bottle",
    "可乐": "coke",
    "雪碧": "sprite",
}

# 🔑 請在這裡填入你剛剛在 Google AI Studio 申請的 Gemini API Key
GEMINI_API_KEY = "AIzaSyAVPna5gJwEsmxcnhAhaPwy2y1qz-vi4Sc"

# 給 Gemini 的系統提示詞（要求它扮演翻譯官，只輸出英文單字）
PROMPT_SYS = (
    "You are a label normalizer. Convert the given Chinese object "
    "description into a short, lowercase English YOLO/vision class name "
    "(1~3 words). If multiple are given, return the single most likely one. "
    "Output ONLY the label, no punctuation."
)

def extract_english_label(query_cn: str) -> Tuple[str, str]:
    """
    返回 (label_en, source)；source ∈ {'local', 'gemini', 'fallback'}
    """
    q = (query_cn or "").strip().lower()
    if q in LOCAL_CN2EN:
        return LOCAL_CN2EN[q], "local"

    # 简单规则：去掉前缀修饰词
    for k, v in LOCAL_CN2EN.items():
        if k in q:
            return v, "local"

    # 🚀 核心修改：使用 Requests 呼叫 Gemini REST API，完美避開 protobuf 衝突
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        
        # 將系統提示詞與使用者要翻譯的中文合併
        combined_prompt = f"{PROMPT_SYS}\n\nUser description: {query_cn.strip()}"
        
        # 根據 Google 官方 API 文件組裝 JSON 格式
        payload = {
            "contents": [{
                "parts": [{"text": combined_prompt}]
            }]
        }
        
        # 發送請求給 Google 伺服器 (設定 10 秒超時)
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status() # 如果遇到 HTTP 錯誤會自動跳轉到 except
        
        # 解析回傳的 JSON 並提取文字
        result_json = response.json()
        label = result_json['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # 清洗字串，確保格式乾淨
        label = label.replace(".", "").replace(",", "").replace("  ", " ").strip()
        
        # 兜底：如果 API 回傳空值，就預設回傳 'bottle'
        return (label or "bottle"), "gemini"
        
    except Exception as e:
        print(f"[LLM] Gemini API 呼叫失敗: {e}")
        return "bottle", "fallback"

# ----- 簡單的本機測試區塊 -----
if __name__ == "__main__":
    # 如果你直接執行這個 Python 檔案，它會跑這裡的測試
    test_word = "腳踏車"
    print(f"測試單字: {test_word}")
    result, source = extract_english_label(test_word)
    print(f"Gemini 翻譯結果: {result} (來源: {source})")