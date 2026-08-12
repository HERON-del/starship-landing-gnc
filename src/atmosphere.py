"""
Exponential atmosphere model.

Density falls roughly exponentially with altitude:

    rho(z) = rho_0 exp(-z / H)

with `rho_0 = 1.225 kg/m^3` at sea level and a scale height of 8500 m. Over the
altitude band this project cares about — ground to a few kilometres — that is
accurate to a few percent against the US Standard Atmosphere, and it has the
property the optimiser needs: it is smooth, positive everywhere, and cheap to
differentiate.

Pressure, temperature and speed of sound use the troposphere lapse-rate model,
which is only needed for reporting Mach number. `ambiance` is already a
dependency and is more accurate; this module exists so the aero model has no
runtime dependency beyond NumPy and so the exact same formula appears inside the
optimiser and inside the simulator.
"""

import numpy as np

RHO_0 = 1.225        # sea-level density [kg/m^3]
H_SCALE = 8500.0     # density scale height [m]
T_0 = 288.15         # sea-level temperature [K]
P_0 = 101325.0       # sea-level pressure [Pa]
LAPSE = 0.0065       # troposphere lapse rate [K/m]
R_AIR = 287.05       # specific gas constant for dry air [J/(kg K)]
GAMMA_AIR = 1.4      # ratio of specific heats


def density(z):
    """
    Air density at altitude [kg/m^3].

    Accepts scalars or arrays. Negative altitudes are clamped to zero rather
    than extrapolated: the optimiser occasionally probes slightly below the pad
    during an iteration, and returning a super-sea-level density there produces
    a drag spike that is entirely fictional.
    """
    z = np.maximum(np.asarray(z, dtype=float), 0.0)
    return RHO_0 * np.exp(-z / H_SCALE)


def temperature(z):
    """Air temperature at altitude [K], troposphere lapse rate."""
    z = np.maximum(np.asarray(z, dtype=float), 0.0)
    return T_0 - LAPSE * z


def pressure(z):
    """Air pressure at altitude [Pa]."""
    z = np.maximum(np.asarray(z, dtype=float), 0.0)
    return P_0 * (1.0 - LAPSE * z / T_0) ** 5.2559


def speed_of_sound(z):
    """Speed of sound at altitude [m/s]."""
    return np.sqrt(GAMMA_AIR * R_AIR * temperature(z))


def mach(speed, z):
    """Mach number for a given speed [m/s] at altitude."""
    return np.asarray(speed, dtype=float) / speed_of_sound(z)


if __name__ == "__main__":
    print(f"{'alt [m]':>9} {'rho [kg/m^3]':>14} {'p [kPa]':>10} "
          f"{'T [K]':>8} {'a [m/s]':>9}")
    for z in (0, 500, 1000, 2000, 3000, 5000, 8500, 12000):
        print(f"{z:9,d} {float(density(z)):14.4f} {float(pressure(z))/1e3:10.2f} "
              f"{float(temperature(z)):8.1f} {float(speed_of_sound(z)):9.1f}")
