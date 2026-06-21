import warnings

import matplotlib.pyplot as plt
import numpy as np
import precession


warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

N = 10000
N_RESONANCE = 8000
N_PRECESSION_SAMPLES = 9000

q_value = 0.8
chi1_value = 1.0
chi2_value = 1.0
r_value = 10.0

q = np.full(N, q_value)
chi1 = np.full(N, chi1_value)
chi2 = np.full(N, chi2_value)
r = np.full(N, r_value)

# Monte Carlo sampling:
# theta_i = arccos(2*u_i - 1), with u_i uniform in [0, 1].
u1 = np.random.uniform(0.0, 1.0, N)
u2 = np.random.uniform(0.0, 1.0, N)

theta1 = np.arccos(2.0*u1 - 1.0)
theta2 = np.arccos(2.0*u2 - 1.0)
deltaphi = np.random.uniform(-np.pi, np.pi, N)

cos_theta1 = np.cos(theta1)
cos_theta2 = np.cos(theta2)

deltachi, kappa, chieff = precession.angles_to_conserved(
    theta1,
    theta2,
    deltaphi,
    r,
    q,
    chi1,
    chi2
)

deltachi = np.asarray(deltachi).reshape(-1)
kappa = np.asarray(kappa).reshape(-1)
chieff = np.asarray(chieff).reshape(-1)

quadrants = [
    (
        "up-up",
        (cos_theta1 >= 0.0) & (cos_theta2 >= 0.0),
        "deeppink",
        "deeppink",
        "o",
        0.0,
        0.0,
        45
    ),
    (
        "up-down",
        (cos_theta1 >= 0.0) & (cos_theta2 < 0.0),
        "dodgerblue",
        "dodgerblue",
        "s",
        0.0,
        np.pi,
        45
    ),
    (
        "down-up",
        (cos_theta1 < 0.0) & (cos_theta2 >= 0.0),
        "darkorange",
        "darkorange",
        "^",
        np.pi,
        0.0,
        45
    ),
    (
        "down-down",
        (cos_theta1 < 0.0) & (cos_theta2 < 0.0),
        "limegreen",
        "limegreen",
        "*",
        np.pi,
        np.pi,
        45
    )
]


def as_1d(value):
    return np.asarray(value, dtype=float).reshape(-1)


def averaged_chip(
    theta1_input,
    theta2_input,
    deltaphi_input,
    r_input,
    q_input,
    chi1_input,
    chi2_input
):
    theta1_input = as_1d(theta1_input)
    theta2_input = as_1d(theta2_input)
    deltaphi_input = as_1d(deltaphi_input)
    r_input = as_1d(r_input)
    q_input = as_1d(q_input)
    chi1_input = as_1d(chi1_input)
    chi2_input = as_1d(chi2_input)

    values = np.full(theta1_input.size, np.nan)

    for i in range(theta1_input.size):
        try:
            values[i] = as_1d(
                precession.eval_chip(
                    kappa=as_1d(
                        precession.angles_to_conserved(
                            theta1_input[i],
                            theta2_input[i],
                            deltaphi_input[i],
                            r_input[i],
                            q_input[i],
                            chi1_input[i],
                            chi2_input[i]
                        )[1]
                    )[0],
                    r=float(r_input[i]),
                    chieff=as_1d(
                        precession.eval_chieff(
                            theta1_input[i],
                            theta2_input[i],
                            q_input[i],
                            chi1_input[i],
                            chi2_input[i]
                        )
                    )[0],
                    q=float(q_input[i]),
                    chi1=float(chi1_input[i]),
                    chi2=float(chi2_input[i]),
                    which="averaged",
                    method="montecarlo",
                    Nsamples=N_PRECESSION_SAMPLES
                )
            )[0]
        except Exception:
            pass

    return values

# ------------------------------------------------------------
# Monte Carlo x and y values
# ------------------------------------------------------------

x = averaged_chip(
    theta1,
    theta2,
    deltaphi,
    r,
    q,
    chi1,
    chi2
)

y = as_1d(
    precession.eval_delta_omega(
        kappa,
        r,
        chieff,
        q,
        chi1,
        chi2
    )
)

# ------------------------------------------------------------
# Schnittman spin-orbit resonances
# ------------------------------------------------------------

chieff_limits = precession.chiefflimits(
    q_value,
    chi1_value,
    chi2_value
)

chieff_min = as_1d(chieff_limits[0])[0]
chieff_max = as_1d(chieff_limits[1])[0]

chieff_resonance = np.linspace(
    chieff_min,
    chieff_max,
    N_RESONANCE
)

r_resonance = np.full(N_RESONANCE, r_value)
q_resonance = np.full(N_RESONANCE, q_value)
chi1_resonance = np.full(N_RESONANCE, chi1_value)
chi2_resonance = np.full(N_RESONANCE, chi2_value)

(
    theta1_pi,
    theta2_pi,
    deltaphi_pi,
    theta1_zero,
    theta2_zero,
    deltaphi_zero
) = precession.anglesresonances(
    r_resonance,
    chieff_resonance,
    q_resonance,
    chi1_resonance,
    chi2_resonance
)

x_pi = averaged_chip(
    theta1_pi,
    theta2_pi,
    deltaphi_pi,
    r_resonance,
    q_resonance,
    chi1_resonance,
    chi2_resonance
)

x_zero = averaged_chip(
    theta1_zero,
    theta2_zero,
    deltaphi_zero,
    r_resonance,
    q_resonance,
    chi1_resonance,
    chi2_resonance
)

y_pi = np.zeros(N_RESONANCE)
y_zero = np.zeros(N_RESONANCE)



# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

x = as_1d(x)
y = as_1d(y)

finite = np.isfinite(x) & np.isfinite(y)

valid_y = y[finite]

if valid_y.size == 0:
    raise RuntimeError(
        "No finite Delta Omega values were produced."
    )

# Exclude the bottom 2% and top 2% of Delta Omega.
y_lower, y_upper = np.percentile(
    valid_y,
    [2.0, 98.0]
)

# Protection against invalid or identical limits.
if (
    not np.isfinite(y_lower)
    or not np.isfinite(y_upper)
    or y_lower >= y_upper
):
    y_lower = np.min(valid_y)
    y_upper = np.max(valid_y)

if y_lower >= y_upper:
    padding = max(
        abs(y_lower) * 0.05,
        1.0e-12
    )

    y_lower -= padding
    y_upper += padding

plot_finite = (
    finite
    & (y >= y_lower)
    & (y <= y_upper)
)

plt.figure()

for (
    label,
    quadrant_mask,
    point_color,
    marker_color,
    marker,
    theta1_exact,
    theta2_exact,
    marker_size
) in quadrants:

    mask = plot_finite & quadrant_mask

    plt.scatter(
        x[mask],
        y[mask],
        s=2,
        color=point_color,
        linewidths=0
    )

    try:
        # All exactly aligned cases have averaged chi_p = 0.
        x_exact = 0.0

        # Nutation and precession-frequency variation vanish
        # for exactly aligned configurations.
        y_exact = 0.0

        if np.isfinite(x_exact) and np.isfinite(y_exact):
            plt.scatter(
                [x_exact],
                [y_exact],
                s=marker_size,
                marker=marker,
                facecolors="none",
                edgecolors=marker_color,
                linewidths=1.2,
                label=label,
                zorder=10
            )

    except Exception:
        pass

finite_zero = (
    np.isfinite(x_zero)
    & np.isfinite(y_zero)
)

finite_pi = (
    np.isfinite(x_pi)
    & np.isfinite(y_pi)
)

# DeltaPhi = 0: black solid line
plt.plot(
    x_zero[finite_zero],
    y_zero[finite_zero],
    color="black",
    linestyle="-",
    linewidth=2.0,
    label=r"$\Delta\Phi=0$ resonance",
    zorder=8
)

# DeltaPhi = pi: black dashed line
plt.plot(
    x_pi[finite_pi],
    y_pi[finite_pi],
    color="black",
    linestyle="--",
    linewidth=2.0,
    label=r"$\Delta\Phi=\pi$ resonance",
    zorder=8
)

plt.xlabel(r"$\chi_p$, averaged")
plt.ylabel(r"$\Delta\Omega_L$")
plt.ylim(y_lower, y_upper)
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()