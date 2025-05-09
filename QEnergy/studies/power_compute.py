from QEnergy.qenergy import components as comp
from QEnergy.qenergy.experiments_dv import (
    BB84Experiment,
    EntanglementBasedExperiment,
    MDIQKDExperiment,
)
from QEnergy.qenergy.experiments_cv import CVQKDProtocol
from QEnergy.studies import FIGSIZE_FULL, EXPORT_DIR


def compute_power(distance, protocol, receiver):
    global key_rate, power
    dist = [distance]
    petabit = 1e15
    pcoupling = 0.9
    sourcerate = 80 * 10 ** 6
    mu = 0.1
    muE91 = 0.1
    QBER = 0.01

    ## Laser and detectors
    wavelength = 1550
    laser = comp.LaserNKTkoheras1550()
    laserE91 = comp.LaserMira780Pulsed()
    if receiver == 'SNSPD':
        detector = comp.DetectorSNSPD1550()
    if receiver == 'APD':
        detector = comp.DetectorInGAs1550()
    pbsm = 0.5

    # Other components
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

    '''othercomponentMDI = [
        comp.MotorizedWavePlate(),
        comp.MotorizedWavePlate(),
        comp.Computer(),
        comp.Computer(),
        comp.Computer(),
        comp.SwitchIntensityModulator(),
        comp.TimeTagger(),
    ]'''

    # Detector
    '''detectorsnspd = comp.DetectorSNSPD1550()
    detectoringaas = comp.DetectorInGAs1550()

    # Other components
    other = [
        comp.MotorizedWavePlate(),
        comp.MotorizedWavePlate(),
        comp.Computer(),
        comp.Computer(),
        comp.SwitchIntensityModulator(),
        comp.TimeTagger(),
    ]'''

    '''Experiment03 = MDIQKDExperiment(
        sourcerate,
        pcoupling,
        mu,
        wavelength,
        QBER,
        laser,
        laser,
        detector,
        dist,
        pbsm,
        othercomponentMDI,
    )'''
    # Experiment01 = BB84Experiment(sourcerate,pcoupling,mu,wavelength,0.01,laser,detectorsnspd,dist,other)
    # Experiment02 = BB84Experiment(sourcerate,pcoupling,mu,wavelength,0.01,laser,detectoringaas,dist,other)
    # Experiment03 = BB84Experiment(sourcerate,pcoupling,mu,wavelength,0.05,laser,detectoringaas,dist,other)


    # rate01 = Experiment01.compute_secret_key_rate()
    '''rate02 = Experiment02.compute_secret_key_rate()
    power03 = Experiment03.power()
    rate03 = Experiment03.compute_secret_key_rate()'''
    if protocol == 'BB84':
        # key_rate = rate01[0]
        Experiment01 = BB84Experiment(
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
        power01 = Experiment01.power()
        source_component = [laser]
        detector_component = [detector]
        source_power = 0
        detector_power = 0
        other_power = 0
        for source in source_component:
            source_power = source_power + source.power
        for detector in detector_component:
            detector_power = detector_power + detector.power
        for other in othercomponentBB84:
            other_power = other_power + other.power
        power = {'total':power01, 'source':source_power, 'detector':detector_power, 'other':other_power}

    if protocol == 'E91':
        Experiment02 = EntanglementBasedExperiment(
            sourcerate,
            pcoupling,
            muE91,
            wavelength,
            QBER,
            laserE91,
            detector,
            detector,
            dist,
            othercomponentE91,
        )
        power02 = Experiment02.power()
        source_component = [laserE91]
        detector_component = [detector, detector]
        other_component = othercomponentE91
        source_power = 0
        detector_power = 0
        other_power = 0
        for source in source_component:
            source_power = source_power + source.power
        for detector in detector_component:
            detector_power = detector_power + detector.power
        for other in other_component:
            other_power = other_power + other.power
        power = {'source':source_power, 'detector':detector_power, 'other':other_power, 'total':power02}
    if protocol == 'CV-QKD':
        gigabit = 1e9  # Target number of secret bits
        MJ = 1e6  # Change of scale to megajoules

        # Parameters of the implementation
        eta = 0.7  # Detector efficiency
        Vel = 0.01  # Electrical noise
        beta_PSK = 0.95  # Information reconciliation efficiency for PSK
        beta_GM = 0.95  # Information reconciliation efficiency for GM
        sourcerate = 1e8  # Laser frequency
        xi = 0.005  # Excess noise

        #############################################################################
        # Source (same for all protocols)
        source = [
            comp.LaserCVPPCL590(),
            comp.MBC(),
            comp.DAC(),
            comp.ThorlabsPowerMeter(),
            comp.Computer(),
        ]

        # Components, Homodyne detection single polarization


        # Components, Homodyne detection double polarization
        DetectorHeterodyne1P = [
            "Heterodyne 1P",
            comp.ADC(),
            comp.LaserCVPPCL590(),
            comp.Computer(),
            comp.SwitchCVQKD(),
            comp.PolarizationController(),
            comp.ThorlabsPDB(),
            comp.ThorlabsPDB(),  # Detector
        ]

        # Components, Heterodyne detection single polarization

        CVQKD_experiment = CVQKDProtocol(
            eta, Vel, beta_GM, sourcerate, source, DetectorHeterodyne1P, xi, dist, "Gauss"
        )
        source_component = [comp.LaserCVPPCL590()]
        detector_component = [comp.ThorlabsPDB(),
            comp.ThorlabsPDB()]
        other_component = [comp.MBC(),
            comp.DAC(),
            comp.ThorlabsPowerMeter(),
            comp.Computer(),
            comp.ADC(),
            comp.LaserCVPPCL590(),
            comp.Computer(),
            comp.SwitchCVQKD(),
            comp.PolarizationController()]
        source_power = 0
        detector_power = 0
        other_power = 0
        for source in source_component:
            source_power = source_power + source.power
        for detector in detector_component:
            detector_power = detector_power + detector.power
        for other in other_component:
            other_power = other_power + other.power
        total_power = CVQKD_experiment.power()
        power = {'source':source_power, 'detector':detector_power, 'other':other_power, 'total':total_power}


    return power
    """tskr = Experiment01.time_skr(petabit)
    tskr2 = Experiment02.time_skr(petabit)
    tskr3 = Experiment03.time_skr(petabit)

    EnergyBB84 = [Experiment01.total_energy(t) / 1000000 for t in tskr]
    EnergyE91 = [Experiment02.total_energy(t) / 1000000 for t in tskr2]
    EnergyMDI = [Experiment03.total_energy(t) / 1000000 for t in tskr3]

    EEBB84 = [r / power01 for r in rate01]
    EEE91 = [r / power02 for r in rate02]
    EEMDI = [r / power03 for r in rate03]"""


"""fig, ax1 = plt.subplots(1, figsize=FIGSIZE_FULL)
left, bottom, width, height = [0.55, 0.35, 0.38, 0.38]
ax2 = fig.add_axes([left, bottom, width, height])
ax1.plot(dist, EEBB84, label="BB84")
ax1.plot(dist, EEE91, label="E91")
ax1.plot(dist, EEMDI, label="MDI")
ax2.plot(dist, EnergyBB84, label="BB84")
ax2.plot(dist, EnergyE91, label="E91")
ax2.plot(dist, EnergyMDI, label="MDI")

ax1.set(xlabel="Distance [km]", ylabel="Energy efficiency [Rate/Watt]")
ax2.set(xlabel="Distance [km]", ylabel="$E^{\\text{1 Petabit}}$ [MJ]")
ax1.tick_params(axis="both", which="major")

ax1.legend(loc="best")
plt.savefig(EXPORT_DIR / "EE.pdf", format="pdf")
plt.show()"""
if __name__ == "__main__":
    power = compute_power(40, 'E91', 'APD')
    print(power)