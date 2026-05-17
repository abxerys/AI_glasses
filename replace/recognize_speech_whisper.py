def recognize_speech_whisper(audio_file_path: str) -> str:
    """呼叫 OpenAI Whisper API 將音檔轉文字"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="zh"  # 指定中文以提升導航指令的辨識速度
            )
        return transcription.text
    except Exception as e:
        print(f"[Whisper API 錯誤]: {e}")
        return ""