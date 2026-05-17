# ---------- WebSocket：ESP32 相机入口（JPEG 二进制） ----------
@app.on_event("startup")
async def start_video_simulation():
    print("啟動本地影片測試迴圈...")
    asyncio.create_task(video_simulation_loop())

async def video_simulation_loop():
    global blind_path_navigator, cross_street_navigator, orchestrator
    global yolo_seg_model, obstacle_detector
    global yolomedia_running, yolomedia_sending_frames, camera_viewers

    # 1. 初始化各項導航器 (這部分邏輯與原本的 ESP32 連線時相同)
    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        print("[NAVIGATION] 盲道導航器已初始化 (影片測試)")

    if cross_street_navigator is None and yolo_seg_model is not None:
        cross_street_navigator = CrossStreetNavigator(
            seg_model=yolo_seg_model,
            coco_model=None,
            obs_model=None
        )
        print("[CROSS_STREET] 過馬路導航器已初始化 (影片測試)")

    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        print("[NAV MASTER] 統領狀態機已初始化 (影片測試)")

    # 2. 開啟本地影片檔 (請確保 test_video.mp4 放在 app_main.py 同一個資料夾)
    video_path = "videos/blind_path.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[影片錯誤] 無法開啟測試影片：{video_path}。請確認檔名與路徑。")
        return

    print("[影片測試] 成功開啟影片，開始模擬推流...")
    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            
            # 如果影片播完，重新從頭播放
            if not ret:
                print("[影片測試] 影片播放完畢，重新開始循環。")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_counter += 1

            # 將影片幀轉為 bytes 格式 (模擬 ESP32 傳來的資料格式)
            ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            data = enc.tobytes()

            # 推送到 bridge_io (供尋物模式 yolomedia 使用)
            try:
                bridge_io.push_raw_jpeg(data)
            except Exception:
                pass

            # -------------------------------------------------------------
            # 【核心辨識邏輯】與原本的處理流程相同
            # -------------------------------------------------------------
            bgr = frame
            
            # 【修改】尋物模式邏輯保留，以防影片測試時手動切換到找物品
            if orchestrator and not yolomedia_running and bgr is not None:
                current_state = orchestrator.get_state()
                
                if current_state == "ITEM_SEARCH":
                    if not yolomedia_sending_frames and camera_viewers:
                        # 找物品模式：直接送出原始畫面，不跑導航模型
                        ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for d in dead:
                                camera_viewers.discard(d)
                    
                    await asyncio.sleep(0.033)
                    continue 
                
                # --- 上帝模式疊加開始 ---
                out_img = bgr.copy() # 複製原始畫面作為底圖，開始逐層疊加
                
                try:
                    # 第一層：紅綠燈辨識
                    import trafficlight_detection
                    tl_result = trafficlight_detection.process_single_frame(out_img, ui_broadcast_callback=None)
                    if tl_result['vis_image'] is not None:
                        out_img = tl_result['vis_image']

                    # 第二層：斑馬線辨識
                    cross_guidance = ""
                    if cross_street_navigator:
                        cross_res = cross_street_navigator.process_frame(out_img)
                        if cross_res.annotated_image is not None:
                            out_img = cross_res.annotated_image
                        cross_guidance = cross_res.guidance_text

                    # 第三層：盲道與障礙物辨識
                    blind_guidance = ""
                    if blind_path_navigator:
                        blind_res = blind_path_navigator.process_frame(out_img)
                        if blind_res.annotated_image is not None:
                            out_img = blind_res.annotated_image
                        blind_guidance = blind_res.guidance_text
                            
                    # 整合語音輸出：將各模組想說的話印在終端機上
                    if tl_result.get('color') not in ["unknown", None]:
                        print(f"[紅綠燈狀態] 偵測到: {tl_result.get('color')}")
                    if cross_guidance:
                        print(f"[過馬路語音] {cross_guidance}")
                    if blind_guidance:
                        print(f"[盲道避障語音] {blind_guidance}")

                except Exception as e:
                    if frame_counter % 30 == 0:
                        print(f"[上帝模式] 處理影片幀時出錯: {e}")

                # --- 廣播結果到網頁端 ---
                if camera_viewers and out_img is not None:
                    ok, enc = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        jpeg_data = enc.tobytes()
                        dead = []
                        for viewer_ws in list(camera_viewers):
                            try:
                                await viewer_ws.send_bytes(jpeg_data)
                            except Exception:
                                dead.append(viewer_ws)
                        for d in dead:
                            camera_viewers.discard(d)
                            
            # 強制暫停 0.033 秒 (模擬真實的 30 FPS 速率)
            await asyncio.sleep(0.033)

    except Exception as e:
        print(f"[影片測試錯誤] 迴圈發生異常: {e}")
    finally:
        cap.release()