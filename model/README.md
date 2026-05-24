# Models

放兩個 YOLOv8s 權重檔，皆**不入 git**（見 [.gitignore](../.gitignore)）。

| 檔名 | 用途 | 來源 |
|---|---|---|
| `yolov8s.pt` | 物品搜尋（COCO 80 類） | 首次執行時由 ultralytics 自動下載到這裡 |
| `crossing_yolov8s.pt` | 斑馬線 + 紅綠燈辨識 | 使用者自備 fine-tune 後上傳到此資料夾 |

## 取得 `yolov8s.pt`

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
# 預設下載到 ~/.cache 或當前資料夾。若已下載過會直接複用。
mv yolov8s.pt models/   # 若下載到別處可移過來
```

## `crossing_yolov8s.pt` 類別命名

預期由 [`edge/config.py`](../edge/config.py) 中的 `CROSSING_CLASSES` 對應。
若實際模型用了不同的類別名稱（例如 `red`、`green` 而非 `red_light`、`green_light`），
請修改 `edge/config.py` 的對應表，不要改模型。
