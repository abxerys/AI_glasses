# audio_player.py
# 处理预录音频文件的播放，通过ESP32扬声器输出

import os
import wave
import json
import asyncio
import threading
import queue
import time
from audio_stream import broadcast_pcm16_realtime
from audio_compressor import compressed_audio_cache, AudioCompressor
import uuid
from gtts import gTTS
import pydub
current_dir = os.path.dirname(os.path.abspath(__file__))

# 暴力指定 FFmpeg 執行檔為同一個資料夾下的 ffmpeg.exe 與 ffprobe.exe
pydub.AudioSegment.converter = os.path.join(current_dir, "ffmpeg.exe")
pydub.AudioSegment.ffprobe   = os.path.join(current_dir, "ffprobe.exe")
# 导入录制器（避免循环导入，在需要时动态导入）
_recorder_imported = False
_sync_recorder = None

def _get_recorder():
    """延迟导入录制器"""
    global _recorder_imported, _sync_recorder
    if not _recorder_imported:
        try:
            import sync_recorder as sr
            _sync_recorder = sr
            _recorder_imported = True
        except Exception as e:
            print(f"[AUDIO] 无法导入录制器: {e}")
            _recorder_imported = True  # 标记已尝试，避免重复
    return _sync_recorder

# 兼容旧工程中的示例音频（保留）
AUDIO_BASE_DIR = r"C:\Users\Administrator\Desktop\rebuild1002\music"

# 新增：voice 目录与映射表
# 使用脚本所在目录的 voice 文件夹，避免工作目录问题
VOICE_DIR = os.getenv("VOICE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice"))
VOICE_MAP_FILE = os.path.join(VOICE_DIR, "map.zh-CN.json")

# 音频文件映射（将合并 voice 映射）
AUDIO_MAP = {
    "检测到物体": os.path.join(AUDIO_BASE_DIR, "音频1.wav"),
    "向上": os.path.join(AUDIO_BASE_DIR, "音频2.wav"),
    "向下": os.path.join(AUDIO_BASE_DIR, "音频3.wav"),
    "向左": os.path.join(AUDIO_BASE_DIR, "音频4.wav"),
    "向右": os.path.join(AUDIO_BASE_DIR, "音频5.wav"),
    "OK": os.path.join(AUDIO_BASE_DIR, "音频6.wav"),
    "向前": os.path.join(AUDIO_BASE_DIR, "音频7.wav"),
    "后退": os.path.join(AUDIO_BASE_DIR, "音频8.wav"),
    "拿到物体": os.path.join(AUDIO_BASE_DIR, "音频9.wav"),
}

# 音频缓存，避免重复读取
_audio_cache = {}

# 音频播放队列和工作线程 - 使用优先级队列
_audio_queue = queue.PriorityQueue(maxsize=10)
_audio_priority = 0  # 递增的优先级计数器
_worker_thread = None
_worker_loop = None
_is_playing = False  # 标记是否正在播放音频
_playing_lock = threading.Lock()  # 播放锁
_initialized = False
_last_play_ts = 0.0  # 记录上次播放结束时间，用于决定预热静音长度

def load_wav_file(filepath):
    """加载WAV文件并返回PCM数据（自动转换为8kHz）"""
    if filepath in _audio_cache:
        return _audio_cache[filepath]
    
    # 使用压缩缓存
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        compressed_data = compressed_audio_cache.load_and_compress(filepath)
        if compressed_data:
            # 存储压缩后的数据
            _audio_cache[filepath] = compressed_data
            return compressed_data
    
    # 原始加载方式（不压缩）
    try:
        with wave.open(filepath, 'rb') as wav:
            # 检查音频格式
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            
            if channels != 1:
                print(f"[AUDIO] 警告: {filepath} 不是单声道，将只使用第一个声道")
            if sampwidth != 2:
                print(f"[AUDIO] 警告: {filepath} 不是16位音频")
            
            # 读取所有帧
            frames = wav.readframes(wav.getnframes())
            
            # 如果是立体声，只取左声道
            if channels == 2:
                import audioop
                frames = audioop.tomono(frames, sampwidth, 1, 0)
            
            # 统一转换为8kHz（使用ratecv保证音调和速度不变）
            if framerate != 8000:
                import audioop
                frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                print(f"[AUDIO] 重采样: {filepath} {framerate}Hz -> 8000Hz")
            
            _audio_cache[filepath] = frames
            return frames
            
    except Exception as e:
        print(f"[AUDIO] 加载音频文件失败 {filepath}: {e}")
        return None

def _merge_voice_map():
    """读取 voice/map.zh-CN.json 并合并到 AUDIO_MAP"""
    try:
        if not os.path.exists(VOICE_MAP_FILE):
            print(f"[AUDIO] 未找到映射文件: {VOICE_MAP_FILE}")
            return
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            m = json.load(f)
        added = 0
        for text, info in (m or {}).items():
            files = (info or {}).get("files") or []
            if not files:
                continue
            fname = files[0]
            fpath = os.path.join(VOICE_DIR, fname)
            if os.path.exists(fpath):
                AUDIO_MAP[text] = fpath
                added += 1
            else:
                print(f"[AUDIO] 映射文件缺失: {fpath}")
        print(f"[AUDIO] 已合并 voice 映射 {added} 条")
    except Exception as e:
        print(f"[AUDIO] 读取 voice 映射失败: {e}")

def preload_all_audio():
    """预加载所有音频文件到内存"""
    print("[AUDIO] 开始预加载音频文件...")
    loaded_count = 0
    
    # 【暂时禁用变速】因为需要修改缓存机制
    # 需要加速的音频列表（斑马线相关）
    # speedup_keywords = ["斑马线", "画面"]
    # speedup_factor = 1.3  # 加速30%
    
    for audio_key, filepath in AUDIO_MAP.items():
        if os.path.exists(filepath):
            # 【修复】暂时使用默认速度加载
            # need_speedup = any(keyword in audio_key for keyword in speedup_keywords)
            # speed = speedup_factor if need_speedup else 1.0
            
            data = load_wav_file(filepath)  # 使用默认参数
            if data:
                loaded_count += 1
                # if need_speedup:
                #     print(f"[AUDIO] 加载（加速{speedup_factor}x）: {audio_key}")
        else:
            # 降低噪声输出
            pass
    print(f"[AUDIO] 预加载完成，共加载 {loaded_count} 个音频文件")

def _audio_worker():
    """音频播放工作线程"""
    global _worker_loop
    
    # 尝试设置线程优先级（Windows特定）
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            # 设置线程为高优先级
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                1  # THREAD_PRIORITY_ABOVE_NORMAL
            )
            print("[AUDIO] 设置音频线程为高优先级")
    except Exception as e:
        print(f"[AUDIO] 设置线程优先级失败: {e}")
    
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)
    
    async def process_queue():
        while True:
            try:
                # 从优先级队列获取数据
                priority_data = await asyncio.get_event_loop().run_in_executor(None, _audio_queue.get, True)
                if priority_data is None:
                    break
                # 解包优先级和实际音频数据
                if isinstance(priority_data, tuple) and len(priority_data) == 2:
                    _, audio_data = priority_data
                else:
                    audio_data = priority_data
                await _broadcast_audio_optimized(audio_data)
            except Exception as e:
                print(f"[AUDIO] 工作线程错误: {e}")
    
    _worker_loop.run_until_complete(process_queue())

async def _broadcast_audio_optimized(pcm_data: bytes):
    """优化的音频广播：单次调用由底层按20ms节拍发送，移除重复节拍和Python层sleep"""
    global _last_play_ts, _is_playing
    try:
        # 设置播放标志
        with _playing_lock:
            _is_playing = True
        # 此时 pcm_data 应该已经是解压后的16位PCM数据了（8kHz）
        now = time.monotonic()
        idle_sec = now - (_last_play_ts or now)
        # 首次或长时间空闲后，预热更长静音；否则小静音
        lead_ms = 160 if idle_sec > 3.0 else 60
        tail_ms = 40

        lead_silence = b'\x00' * (lead_ms * 8000 * 2 // 1000)  # 8k * 2B
        tail_silence = b'\x00' * (tail_ms * 8000 * 2 // 1000)

        # 完整音频数据（包含静音）
        full_audio = lead_silence + pcm_data + tail_silence
        
        # 注意：录制在 broadcast_pcm16_realtime 中统一完成，避免重复

        # 单次调用交给底层 pacing（20ms节拍在 broadcast_pcm16_realtime 内部实现）
        await broadcast_pcm16_realtime(full_audio)

        _last_play_ts = time.monotonic()
    except Exception as e:
        print(f"[AUDIO] 广播音频失败: {e}")
    finally:
        # 清除播放标志
        with _playing_lock:
            _is_playing = False

def initialize_audio_system():
    """初始化音频系统"""
    global _initialized, _worker_thread, _last_play_ts
    
    if _initialized:
        return
    
    # 先合并 voice 映射，再预加载
    _merge_voice_map()
    preload_all_audio()
    
    _worker_thread = threading.Thread(target=_audio_worker, daemon=True)
    _worker_thread.start()
    _initialized = True
    _last_play_ts = 0.0
    
    # 显示压缩统计
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        stats = compressed_audio_cache.get_compression_stats()
        print(f"[AUDIO] 音频压缩统计:")
        print(f"  - 文件数: {stats['files_cached']}")
        print(f"  - 原始大小: {stats['total_original_size'] / 1024:.1f} KB")
        print(f"  - 压缩后: {stats['total_compressed_size'] / 1024:.1f} KB")
        print(f"  - 压缩率: {stats['compression_ratio']:.1%}")
        print(f"  - 节省: {stats['bytes_saved'] / 1024:.1f} KB")
    
    print("[AUDIO] 音频系统初始化完成（预加载+工作线程）")

def play_audio_threadsafe(audio_key):
    """线程安全的音频播放函数"""
    global _audio_queue, _audio_priority
    
    if not _initialized:
        initialize_audio_system()
    
    if audio_key not in AUDIO_MAP:
        print(f"[AUDIO] 未知的音频键: {audio_key}")
        return
    
    filepath = AUDIO_MAP[audio_key]
    pcm_data = _audio_cache.get(filepath)
    if pcm_data is None:
        print(f"[AUDIO] 音频未在缓存中: {audio_key}")
        return
    
    # 如果是压缩的数据，先解压
    if pcm_data and len(pcm_data) > 5 and pcm_data[0] in [0x01, 0x02]:
        pcm_data = compressed_audio_cache.decompress(pcm_data)
        if not pcm_data:
            print(f"[AUDIO] 解压失败: {audio_key}")
            return
    
    # 【优化】实时播报策略：保持队列最小化，避免积压延迟
    queue_size = _audio_queue.qsize()
    
    # 检查是否正在播放
    with _playing_lock:
        currently_playing = _is_playing
    
    # 实时策略：只允许1个积压，超过立即清空
    if queue_size > 0 and not currently_playing:
        # 未播放时立即清空，播放最新语音
        print(f"[AUDIO] 清空队列（当前{queue_size}个），播放最新语音")
        _audio_queue = queue.PriorityQueue(maxsize=10)
    elif queue_size > 1 and currently_playing:
        # 正在播放时，如果积压>1个则清空（保持实时性）
        print(f"[AUDIO] 队列积压({queue_size}个)，清空以保持实时")
        _audio_queue = queue.PriorityQueue(maxsize=10)
    try:
        # 使用优先级队列，确保音频按顺序播放
        _audio_priority += 1
        _audio_queue.put_nowait((_audio_priority, pcm_data))
        if queue_size >= 1:
            print(f"[AUDIO] 播放队列当前大小: {queue_size + 1}")
    except queue.Full:
        # 播放队列满则丢弃，保持实时性
        print(f"[AUDIO] 队列满，丢弃: {audio_key}")
        pass

# 全局语音节流
_last_voice_time = 0
_last_voice_text = ""
_voice_cooldown = 4.0  # 相同语音至少间隔1秒

# 语音优先级定义
VOICE_PRIORITY = {
    'obstacle': 100,     # 障碍物 - 最高优先级
    'direction': 50,     # 转向/平移 - 中等优先级  
    'straight': 10,      # 保持直行 - 最低优先级
    'other': 30          # 其他 - 默认优先级
}


TTS_TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_tts")
if not os.path.exists(TTS_TEMP_DIR):
    os.makedirs(TTS_TEMP_DIR)
# 新增：根据中文提示文案直接播放（会做轻度规范化与降级）
#change
def play_voice_text(text: str):
    global _last_voice_time, _last_voice_text
    
    if not text:
        return
    if not _initialized:
        initialize_audio_system()
    
    current_time = time.time()
    # 嚴格節流：相同的字句，4秒內絕對不重複播放
    if text == _last_voice_text and (current_time - _last_voice_time) < _voice_cooldown:
        return

    _last_voice_text = text
    _last_voice_time = current_time

    # 【實作理由 1：優先調用本地資源 (Cache Hit)】
    # 如果要播放的文字 (例如 "向前", "向左") 已經存在於 AUDIO_MAP，
    # 直接呼叫 play_audio_threadsafe 使用快取。這會瞬間將原生的 8kHz/16-bit PCM 
    # 推入佇列，完全避開網路延遲與格式轉換問題，大幅提升導航即時性。
    if text in AUDIO_MAP:
        print(f"[AUDIO] 命中本地預錄語音快取: {text}")
        play_audio_threadsafe(text)
        return

    # 【實作理由 2：為主執行緒解鎖 (Thread Non-blocking)】
    # 當遇到無法對應快取的動態文字 (例如 "紅綠燈檢測已啟動") 時，
    # 必須將同步的 gTTS 網路請求與 pydub 檔案處理放進背景執行緒 (Background Thread)。
    # 這樣才不會卡死 FastAPI 的主非同步迴圈，確保 ESP32 的 HTTP 音訊串流能平順傳輸而不超時。
    def _tts_task():
        try:
            unique_id = uuid.uuid4().hex
            mp3_path = os.path.join(TTS_TEMP_DIR, f"{unique_id}.mp3")
            wav_path = os.path.join(TTS_TEMP_DIR, f"{unique_id}.wav")

            print(f"[AUDIO] 正在生成 TTS 語音 (背景): {text}")
            tts = gTTS(text=text, lang='zh-tw')
            tts.save(mp3_path)

            audio = pydub.AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(8000)
            audio = audio.set_channels(1)
            audio = audio.set_sample_width(2)
            audio.export(wav_path, format="wav")

            audio += 2
            pcm_data = audio.raw_data

            if pcm_data:
                global _audio_priority, _audio_queue
                _audio_priority += 1
                
                # 清空目前佇列以確保即時性
                with _audio_queue.mutex:
                    _audio_queue.queue.clear()
                
                _audio_queue.put_nowait((_audio_priority, pcm_data))
                print(f"[AUDIO] AI 語音播放推送成功: {text}")

            try:
                if os.path.exists(mp3_path): os.remove(mp3_path)
                if os.path.exists(wav_path): os.remove(wav_path)
            except Exception:
                pass
                
        except Exception as e:
            print(f"[AUDIO] AI 語音生成/播放失敗: {e}")

    # 啟動背景執行緒去處理語音生成，讓主迴圈立刻放行
    threading.Thread(target=_tts_task, daemon=True).start()

# 兼容旧接口
play_audio_on_esp32 = play_audio_threadsafe