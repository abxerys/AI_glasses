# Voice Assets

預錄製的中文播報音檔（.wav）放在這個資料夾。

## 為什麼用預錄音檔而非 TTS

- **延遲低**：edge-tts 雲端合成大約 0.5–1.5 秒，預錄音檔 < 50 ms。
- **離線可用**：行動裝置不一定有穩定網路。
- **音質一致**：每次播報的聲音都一樣。

代價是只能播放預錄好的句子。對於動態內容（例如「找到了，**杯子**就在你前方」這種含有任意物品名稱的句子），程式會 fallback 到 edge-tts。

## 檔案佈局

```
assets/voice/
├── map.zh-CN.json          # 事件 → 檔名 對照表（會入 git）
├── 红灯.WAV               # 各種預錄 .wav（不入 git，避免 repo 過大）
├── 绿灯.wav
├── ...
```

`.wav` 檔本身**不入 git**（見 [.gitignore](../../.gitignore)）。
請把您現有的所有 .wav 跟 `map.zh-CN.json` 都複製進這個資料夾。

## map.zh-CN.json 格式

預期是 `{"event_name": "filename.wav", ...}` 的對照。
[`edge/audio/voice_assets.py`](../../edge/audio/voice_assets.py) 會載入這個檔，
程式碼用語意鍵（例如 `red_light_wait`）查表得到實際 .wav 檔名。

實際 JSON 結構等檔案上傳後再由程式碼適應。
