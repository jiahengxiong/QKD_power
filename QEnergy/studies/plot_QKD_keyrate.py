from plotly.data import experiment

# 导入 QEnergy 包的组件和实验类
from QEnergy.qenergy import components as comp
from QEnergy.qenergy.experiments_dv import (
    BB84Experiment,
    EntanglementBasedExperiment,
    MDIQKDExperiment,
)
from QEnergy.qenergy.experiments_cv import CVQKDProtocol
from QEnergy.studies import FIGSIZE_HALF, EXPORT_DIR

import matplotlib.pyplot as plt
import numpy as np


# 修改后的 compute_key_rate 函数，distance 参数为单个数值，内部转换为列表传递给实验类
def compute_key_rate(distance, protocol, receiver=None):
    # 将单个距离值封装成列表（实验类要求传入列表）
    dist = [distance]
    petabit = 1e15
    pcoupling = 0.9
    sourcerate = 80 * 10 ** 6
    mu = 0.1
    muE91 = 0.1
    QBER = 0.01

    # 激光器和探测器的设定
    wavelength = 1550
    laser = comp.LaserNKTkoheras1550()
    laserE91 = comp.LaserMira780Pulsed()

    # 对于BB84和E91，使用APD探测器；仅当 receiver 参数为 'APD' 时设置APD探测器
    if receiver == 'APD':
        detector = comp.DetectorInGAs1550()
    elif receiver == 'SNSPD':
        detector = comp.DetectorSNSPD1550()
    else:
        # 如果未指定或不适用，则后续协议内部自行设定（例如CV-QKD不使用此探测器）
        detector = None

    pbsm = 0.5

    # 其他组件配置
    othercomponentBB84 = [
        comp.MotorizedWavePlate(),
        comp.MotorizedWavePlate(),
        comp.Computer(),
        comp.Computer(),
        comp.SwitchIntensityModulator(),
        comp.TimeTagger(),
    ]
    othercomponentE91 = [
        comp.MotorizedWavePlate(),
        comp.MotorizedWavePlate(),
        comp.Computer(),
        comp.Computer(),
        comp.Oven(),
        comp.TimeTagger(),
        comp.TimeTagger(),
    ]
    othercomponentMDI = [
        comp.MotorizedWavePlate(),
        comp.MotorizedWavePlate(),
        comp.Computer(),
        comp.Computer(),
        comp.Computer(),
        comp.SwitchIntensityModulator(),
        comp.TimeTagger(),
    ]

    # 根据选择的协议进行计算
    if protocol == 'BB84':
        Experiment = BB84Experiment(
            sourcerate,
            pcoupling,
            mu,
            wavelength,
            QBER,
            laser,
            detector,
            dist,
            othercomponentBB84,
        )
        rate = Experiment.compute_secret_key_rate()
        key_rate = rate[0]
    elif protocol == 'E91':
        Experiment = EntanglementBasedExperiment(
            sourcerate,
            pcoupling,
            muE91,
            wavelength,
            QBER,
            laserE91,
            detector,
            detector,  # 双探测器配置相同
            dist,
            othercomponentE91,
        )
        rate = Experiment.compute_secret_key_rate()
        key_rate = rate[0]
    elif protocol == 'CV-QKD':
        # 此处CV-QKD的参数与探测器配置保持不变
        gigabit = 1e9  # 目标秘钥比特数量
        MJ = 1e6  # 转换单位到兆焦

        eta = 0.7  # 探测器效率
        Vel = 0.01  # 电噪声
        beta_PSK = 0.95  # PSK信息校正效率（未使用）
        beta_GM = 0.95  # GM信息校正效率
        sourcerate_cv = 1e8  # 激光频率
        xi = 0.005  # 额外噪声

        # 源组件（对所有协议通用）
        source = [
            comp.LaserCVPPCL590(),
            comp.MBC(),
            comp.DAC(),
            comp.ThorlabsPowerMeter(),
            comp.Computer(),
        ]

        # 同相干探测组件（保留原有设置）
        DetectorHeterodyne1P = [
            "Heterodyne 1P",
            comp.ADC(),
            comp.LaserCVPPCL590(),
            comp.Computer(),
            comp.SwitchCVQKD(),
            comp.PolarizationController(),
            comp.ThorlabsPDB(),
            comp.ThorlabsPDB(),  # 作为探测器的部分
        ]
        CVQKD_experiment = CVQKDProtocol(
            eta, Vel, beta_GM, sourcerate_cv, source, DetectorHeterodyne1P, xi, dist, "Gauss"
        )
        rate = CVQKD_experiment.compute_secret_key_rate()
        key_rate = rate[0]
    else:
        raise ValueError("未识别的协议类型。请选择 'BB84', 'E91' 或 'CV-QKD'。")

    return key_rate


# 定义距离范围 [1, 120] km
distances = np.arange(1, 401)  # 1 到 120 km

# 分别存储不同协议的秘钥率列表
key_rates_BB84 = []
key_rates_E91 = []
key_rates_CVQKD = []

# 计算每个距离下各协议的秘钥率
for d in distances:
    kr_bb84 = compute_key_rate(d, protocol='BB84', receiver='SNSPD')
    key_rates_BB84.append(kr_bb84)

    kr_e91 = compute_key_rate(d, protocol='BB84', receiver='APD')
    key_rates_E91.append(kr_e91)

    # 对于 CV-QKD，不需要传入探测器参数（内部已有固定设置）
    kr_cvqkd = compute_key_rate(d, protocol='CV-QKD')
    key_rates_CVQKD.append(kr_cvqkd)

# 绘图展示各个协议下的秘钥率随距离的变化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(distances, key_rates_BB84, label='BB84 + SNSPD')
plt.plot(distances, key_rates_E91, label='BB84 + APD')
plt.plot(distances, key_rates_CVQKD, label='CV-QKD')

plt.xlabel('Distance [km]', fontsize=20)
plt.ylabel('Secret Key Rate [bit/s]', fontsize=20)
plt.title('Secret Key Rate vs Distance for QKD Protocols', fontsize=20)

# 设置坐标轴刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=20)

# 设置 y 轴为对数坐标
plt.yscale('log')

plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()


# 保存图像（可选）
plt.savefig(EXPORT_DIR / "KeyRate_vs_Distance.png")
plt.show()