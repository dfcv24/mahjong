def get_chi_tiles_by_type(hand, rush_tile, chi_type):
    """
    根据吃牌类型和rush牌，从手牌中选择构成顺子的两张牌
    
    参数:
    hand: 手牌列表
    rush_tile: rush牌ID
    chi_type: 吃牌类型 (0=前吃, 1=中吃, 2=后吃)
    
    返回:
    选择的两张手牌的索引
    """
    # 获取rush牌的点数和花色
    r_num = rush_tile % 9  # 0-8
    suit = rush_tile // 9  # 0=万, 1=条, 2=筒
    
    # 根据吃牌类型确定需要的两张牌
    if chi_type == 0:  # 前吃
        needed_tiles = [suit * 9 + (r_num - 2), suit * 9 + (r_num - 1)]
    elif chi_type == 1:  # 中吃
        needed_tiles = [suit * 9 + (r_num - 1), suit * 9 + (r_num + 1)]
    else:  # 后吃
        needed_tiles = [suit * 9 + (r_num + 1), suit * 9 + (r_num + 2)]
    
    # 从手牌中找到这两张牌的索引
    indices = []
    for tile in needed_tiles:
        for i, h in enumerate(hand):
            if h == tile and i not in indices:
                indices.append(i)
                break
    
    return indices
