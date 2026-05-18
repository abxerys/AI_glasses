COCO_ZH = {
    "person": "人",
    "bicycle": "腳踏車",
    "car": "汽車",
    "motorcycle": "機車",
    "airplane": "飛機",
    "bus": "公車",
    "train": "火車",
    "truck": "卡車",
    "boat": "船",
    "traffic light": "紅綠燈",
    "fire hydrant": "消防栓",
    "stop sign": "停止標誌",
    "parking meter": "停車計時器",
    "bench": "長椅",
    "bird": "鳥",
    "cat": "貓",
    "dog": "狗",
    "horse": "馬",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑馬",
    "giraffe": "長頸鹿",
    "backpack": "背包",
    "umbrella": "雨傘",
    "handbag": "手提包",
    "tie": "領帶",
    "suitcase": "行李箱",
    "frisbee": "飛盤",
    "skis": "滑雪板",
    "snowboard": "雪板",
    "sports ball": "球",
    "kite": "風箏",
    "baseball bat": "球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "衝浪板",
    "tennis racket": "網球拍",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "湯匙",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "蘋果",
    "sandwich": "三明治",
    "orange": "橘子",
    "broccoli": "花椰菜",
    "carrot": "紅蘿蔔",
    "hot dog": "熱狗",
    "pizza": "披薩",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙發",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "馬桶",
    "tv": "電視",
    "laptop": "筆電",
    "mouse": "滑鼠",
    "remote": "遙控器",
    "keyboard": "鍵盤",
    "cell phone": "手機",
    "microwave": "微波爐",
    "oven": "烤箱",
    "toaster": "烤麵包機",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "書",
    "clock": "時鐘",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹風機",
    "toothbrush": "牙刷",
    # ── shoppingbest5.pt custom classes ──
    "AD_milk": "AD鈣奶",
    "Red_Bull": "紅牛",
}

ZH_TO_COCO = {}
for _en, _zh in COCO_ZH.items():
    ZH_TO_COCO[_zh] = _en

_ZH_ALIAS = {
    "手機": "cell phone",
    "電話": "cell phone",
    "行動電話": "cell phone",
    "鑰匙": "backpack",
    "包包": "backpack",
    "後背包": "backpack",
    "錢包": "handbag",
    "皮夾": "handbag",
    "水壺": "bottle",
    "瓶子": "bottle",
    "水瓶": "bottle",
    "馬克杯": "cup",
    "杯子": "cup",
    "電腦": "laptop",
    "筆電": "laptop",
    "電視": "tv",
    "遙控": "remote",
    "遙控器": "remote",
    "雨傘": "umbrella",
    "傘": "umbrella",
    "書本": "book",
    "書": "book",
    "椅子": "chair",
    "盆栽": "potted plant",
    "花": "potted plant",
    # ── shoppingbest5.pt custom items ──
    "AD鈣奶": "AD_milk",
    "AD钙奶": "AD_milk",
    "鈣奶": "AD_milk",
    "钙奶": "AD_milk",
    "牛奶": "AD_milk",
    "紅牛": "Red_Bull",
    "红牛": "Red_Bull",
    "能量飲料": "Red_Bull",
    "能量饮料": "Red_Bull",
    "提神飲料": "Red_Bull",
}
ZH_TO_COCO.update(_ZH_ALIAS)


def en_to_zh(name: str) -> str:
    return COCO_ZH.get(name, name)


def zh_to_coco(zh_text: str) -> str | None:
    """Match a fragment of Chinese text against a known item alias.

    Returns the COCO English class name, or None if no alias appears in the text.
    Longest alias wins to avoid '水瓶' getting matched as '瓶'.
    """
    candidates = [k for k in ZH_TO_COCO.keys() if k in zh_text]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return ZH_TO_COCO[best]
