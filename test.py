from mpmath import mp, mpf, sqrt

# 设置小数精度（比如100位）
mp.dps = 999999999999999  # Decimal Places

def liuhui_pi(iterations=20):
    """
    用刘徽割圆术（单位圆）高精度逼近π
    :param iterations: 割圆次数，每次边数翻倍
    :return: 逼近的π值（面积）
    """
    r = mpf(1)            # 单位圆半径
    n = 6                 # 从正六边形开始
    s = mpf(1)            # 初始边长（可以设为单位边长近似）

    # 实际从角度计算初始边长更合理
    s = 2 * r * mp.sin(mp.pi / n)

    for _ in range(iterations):
        # 割圆术边长递推公式：
        # 新边长 s' = sqrt(2 - sqrt(4 - s^2))
        s = sqrt(2 - sqrt(4 - s**2))
        n *= 2

    perimeter = n * s
    area = (perimeter * r) / 2
    return area

# 调用
pi_approx = liuhui_pi(iterations=999)

print(f"用割圆术计算得到的π值（精度100位）：\n{pi_approx}")