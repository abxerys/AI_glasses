# voice_engine.py
import speech_recognition as sr
import pyttsx3
import os

def recognize_audio_from_file(wav_file_path: str) -> str:
    """
    【ASR 語音轉文字】
    讀取 ESP32 傳來的原生 WAV 錄音檔，並透過 Google API 辨識成文字。
    """
    if not os.path.exists(wav_file_path):
        print(f"[ASR] 找不到音檔: {wav_file_path}")
        return ""
        
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_file_path) as source:
            audio_data = recognizer.record(source)
            
        print("[ASR] 正在將語音送往 Google 進行辨識...")
        text = recognizer.recognize_google(audio_data, language="zh-TW")
        print(f"[ASR] 辨識結果: {text}")
        return text
        
    except sr.UnknownValueError:
        print("[ASR] 抱歉，聽不清楚您說什麼。")
        return ""
    except Exception as e:
        print(f"[ASR] 發生未預期錯誤: {e}")
        return ""

def text_to_speech_file(text: str, output_path: str = "stream.wav"):
    """
    【TTS 文字轉語音 (離線版)】
    生成 100% 純正的 PCM WAV 音檔，確保 ESP32 喇叭不會破音。
    """
    if not text:
        return
        
    print(f"[TTS] 正在生成純淨 WAV 語音: '{text}' ...")
    try:
        engine = pyttsx3.init()
        # 稍微調慢語速，讓 ESP32 播放時更清晰 (預設通常是 200)
        engine.setProperty('rate', 160) 
        
        # 存成標準 WAV 檔
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        print(f"[TTS] 語音已成功儲存至 {output_path}")
    except Exception as e:
        print(f"[TTS] 語音生成失敗: {e}")

# ==========================================
# 本機測試區塊
# ==========================================
if __name__ == "__main__":
    test_text = "系統啟動成功，目前位於對話模式，請說出指令。"
    test_output = "test_voice.wav"
    
    # 1. 測試 TTS (生成純正 WAV)
    text_to_speech_file(test_text, test_output)
    
    # 2. 測試 ASR (讀取剛剛生成的 WAV)
    print("\n--- 開始測試聽力 ---")
    recognize_audio_from_file(test_output)