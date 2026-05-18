from edge.intent import parse


def test_voice_search_phone():
    i = parse("幫我找手機")
    assert i.name == "voice_search"
    assert i.target == "cell phone"


def test_voice_search_cup():
    i = parse("找杯子")
    assert i.name == "voice_search"
    assert i.target == "cup"


def test_voice_search_bottle_aliases():
    assert parse("尋水壺").target == "bottle"
    assert parse("找水瓶").target == "bottle"


def test_voice_search_shopping_drinks():
    assert parse("找紅牛").target == "Red_Bull"
    assert parse("找AD鈣奶").target == "AD_milk"
    assert parse("尋鈣奶").target == "AD_milk"


def test_crossing():
    assert parse("我要過馬路").name == "crossing"
    assert parse("斑馬線").name == "crossing"
    assert parse("看一下紅綠燈").name == "crossing"


def test_blind_path():
    assert parse("盲道").name == "blind_path"
    assert parse("盲道導航").name == "blind_path"
    assert parse("回到盲道").name == "blind_path"
    assert parse("导盲").name == "blind_path"


def test_cancel_takes_priority():
    assert parse("停").name == "cancel"
    assert parse("取消").name == "cancel"
    assert parse("結束尋找手機").name == "cancel"


def test_unknown():
    assert parse("").name == "unknown"
    assert parse("今天天氣不錯").name == "unknown"
    assert parse("找個不存在的東西").name == "unknown"
