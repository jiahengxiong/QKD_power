import matplotlib.pyplot as plt

# 数据集1：Protocol: BB84, Bypass: True, Detector: APD, Map: Paris
data_bypass_true = {
    5000: 283.6363636363637, 10000: 283.6363636363637,
    15000: 312.00000000000006, 20000: 312.00000000000006,
    25000: 368.7272727272728, 30000: 453.8181818181819,
    35000: 453.8181818181819, 40000: 510.5454545454547,
    45000: 567.2727272727274, 50000: 595.6363636363637,
    55000: 595.6363636363637, 60000: 652.3636363636365,
    65000: 709.0909090909092, 70000: 737.4545454545456,
    75000: 794.1818181818184, 80000: 794.1818181818184,
    85000: 794.1818181818184, 90000: 822.5454545454547,
    95000: 850.9090909090911, 100000: 907.6363636363639,
    105000: 992.727272727273
}

# 数据集2：Protocol: BB84, Bypass: False, Detector: APD, Map: Paris
data_bypass_false = {
    5000: 283.6363636363637, 10000: 283.6363636363637,
    15000: 312.00000000000006, 20000: 312.00000000000006,
    25000: 340.36363636363643, 30000: 340.36363636363643,
    35000: 397.0909090909092, 40000: 397.0909090909092,
    45000: 425.45454545454555, 50000: 425.45454545454555,
    55000: 482.1818181818183, 60000: 482.1818181818183,
    65000: 538.909090909091, 70000: 538.909090909091,
    75000: 567.2727272727274, 80000: 595.6363636363637,
    85000: 652.3636363636365, 90000: 709.0909090909092,
    95000: 709.0909090909092, 100000: 794.1818181818184,
    105000: 822.5454545454547
}

# 保证 x 坐标（traffic）顺序正确
traffic_true = sorted(data_bypass_true.keys())
key_rate_true = [data_bypass_true[k] for k in traffic_true]

traffic_false = sorted(data_bypass_false.keys())
key_rate_false = [data_bypass_false[k] for k in traffic_false]

# 创建图形并绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(traffic_true, key_rate_true, marker='o', label='BB84, Bypass: True, APD, Paris')
plt.plot(traffic_false, key_rate_false, marker='s', label='BB84, Bypass: False, APD, Paris')

plt.xlabel("Traffic bit/s")
plt.ylabel("Average Power")
plt.title("Secret Key Rate vs Traffic for BB84 Protocol (APD Detector)")
plt.xlim(5000, 105000)
plt.grid(True)
plt.legend()

# 保存图像到 PNG 文件
plt.savefig("Key_Rate_vs_Traffic.png", dpi=300)

# 显示图形
plt.show()