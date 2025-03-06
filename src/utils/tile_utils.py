from .constants import *

def init_tile_mapping():
    """初始化牌面到id的映射，使用条筒万的顺序"""
    tile_to_id = {}
    id_to_tile = {}
    index_symble_list = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    # 条子: 1-9 (现在是0-8)
    for i in range(9):
        tile_name = f"{index_symble_list[i]}条"
        tile_to_id[tile_name] = i
        id_to_tile[i] = tile_name
    
    # 饼子: 1-9 (现在是9-17)
    for i in range(9):
        tile_name = f"{index_symble_list[i]}筒"
        tile_to_id[tile_name] = i + 9
        id_to_tile[i + 9] = tile_name
    
    # 万子: 1-9 (现在是18-26)
    for i in range(9):
        tile_name = f"{index_symble_list[i]}万"
        tile_to_id[tile_name] = i + 18
        id_to_tile[i + 18] = tile_name
    
    # 风牌: 东南西北 (现在是27-30)
    winds = ["东", "南", "西", "北"]
    for i, wind in enumerate(winds):
        tile_name = f"{wind}风"
        tile_to_id[tile_name] = i + 27
        id_to_tile[i + 27] = tile_name
    
    # 箭牌: 中发白 (现在是31-33)
    dragons = ["红中", "发财", "白板"]
    for i, dragon in enumerate(dragons):
        tile_to_id[dragon] = i + 31
        id_to_tile[i + 31] = dragon
    
    return tile_to_id, id_to_tile

def is_valid_chi_combination(rush_card, card1, card2):
    """检查三张牌是否能构成有效的顺子"""
    suits = ['万', '条', '饼']
    
    # 提取花色
    rush_suit = next((s for s in suits if s in rush_card), None)
    card1_suit = next((s for s in suits if s in card1), None)
    card2_suit = next((s for s in suits if s in card2), None)
    
    # 如果任何一张牌没有有效花色，或者花色不一致，则无法组成顺子
    if not (rush_suit and card1_suit and card2_suit and rush_suit == card1_suit == card2_suit):
        return False
    
    # 提取数字
    try:
        rush_num = int(''.join(filter(str.isdigit, rush_card)))
        card1_num = int(''.join(filter(str.isdigit, card1)))
        card2_num = int(''.join(filter(str.isdigit, card2)))
    except:
        return False
    
    # 检查三个数字是否构成顺子
    numbers = [rush_num, card1_num, card2_num]
    numbers.sort()
    return numbers[0] + 1 == numbers[1] and numbers[1] + 1 == numbers[2]

def chinese_tile_to_id(tile_name):
    """
    将中文麻将牌名称转换为数字ID
    使用条-筒-万的顺序
    
    参数:
    tile_name: 中文麻将牌名称，如"一万"、"东"等
    
    返回:
    对应的牌ID (0-33)，如果无法识别则返回-1
    """
    # 条子: 0-8 (一条到九条)
    if "条" in tile_name or "索" in tile_name:
        for i, num in enumerate(["一", "二", "三", "四", "五", "六", "七", "八", "九"]):
            if num in tile_name:
                return i
    
    # 饼子: 9-17 (一饼到九饼)
    if "饼" in tile_name or "筒" in tile_name:
        for i, num in enumerate(["一", "二", "三", "四", "五", "六", "七", "八", "九"]):
            if num in tile_name:
                return i + 9
    
    # 万子: 18-26 (一万到九万)
    if "万" in tile_name:
        for i, num in enumerate(["一", "二", "三", "四", "五", "六", "七", "八", "九"]):
            if num in tile_name:
                return i + 18
    
    # 风牌: 27-30 (东南西北)
    wind_map = {"东风": 27, "南风": 28, "西风": 29, "北风": 30}
    for wind, idx in wind_map.items():
        if wind in tile_name:
            return idx
    
    # 箭牌: 31-33 (中发白)
    dragon_map = {"红中": 31, "发财": 32, "白板": 33}
    for dragon, idx in dragon_map.items():
        if dragon in tile_name:
            return idx
    
    # 未能识别
    return -1

def tile_id_to_chinese(tile_id):
    """
    将牌ID转换为中文麻将牌名称
    使用条-筒-万的顺序
    
    参数:
    tile_id: 麻将牌ID，范围为0-33
    
    返回:
    对应的中文麻将牌名称
    """
    # 条子: 0-8
    if 0 <= tile_id <= 8:
        numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
        return f"{numbers[tile_id]}条"
    
    # 饼子/筒子: 9-17
    elif 9 <= tile_id <= 17:
        numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
        return f"{numbers[tile_id-9]}筒"
    
    # 万子: 18-26
    elif 18 <= tile_id <= 26:
        numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
        return f"{numbers[tile_id-18]}万"
    
    # 风牌: 27-30
    elif 27 <= tile_id <= 30:
        winds = ["东", "南", "西", "北"]
        return f"{winds[tile_id-27]}风"
    
    # 箭牌: 31-33
    elif 31 <= tile_id <= 33:
        dragons = ["红中", "发财", "白板"]
        return dragons[tile_id-31]
    
    # 未能识别
    return "未知牌"

def format_hand_chinese(hand):
    """将手牌数组转换为中文描述"""
    valid_tiles = [tile for tile in hand if tile != NUM_TILE_TYPES and tile >= 0]
    return " ".join([tile_id_to_chinese(tile) for tile in valid_tiles])

def parse_chinese_hand(hand_text):
    """
    解析中文手牌文本
    
    参数:
    hand_text: 中文手牌文本，如"一万 三条 东风"等
    
    返回:
    数字ID列表
    """
    # 尝试多种可能的分隔符
    for sep in [' ', ',', '，', '、', '|']:
        if sep in hand_text:
            tiles = hand_text.split(sep)
            tile_ids = []
            for tile in tiles:
                tile = tile.strip()
                if tile:  # 非空
                    tile_id = chinese_tile_to_id(tile)
                    if tile_id >= 0:  # 有效牌
                        tile_ids.append(tile_id)
            if tile_ids:  # 如果成功解析出牌
                return tile_ids
    
    # 如果没有常见分隔符，尝试逐字解析
    # 假设每个牌名是2-3个字符
    i = 0
    tile_ids = []
    while i < len(hand_text):
        # 尝试2字符和3字符的牌名
        for length in [2, 3]:
            if i + length <= len(hand_text):
                tile_name = hand_text[i:i+length]
                tile_id = chinese_tile_to_id(tile_name)
                if tile_id >= 0:  # 有效牌
                    tile_ids.append(tile_id)
                    i += length
                    break
        else:
            # 如果2字符和3字符都不匹配，跳过当前字符
            i += 1
    
    return tile_ids

def sort_hand_cards(hand_cards):
    hand_cards_id = [chinese_tile_to_id(card) for card in hand_cards]
    hand_cards_id.sort()
    hand_cards = [tile_id_to_chinese(tile_id) for tile_id in hand_cards_id]
    return hand_cards