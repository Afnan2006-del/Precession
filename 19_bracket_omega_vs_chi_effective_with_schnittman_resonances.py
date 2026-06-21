import warnings

import matplotlib.pyplot as plt
import numpy as np
import precession


warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

N = 10000
N_RESONANCE = 500
N_PRECESSION_SAMPLES = 1000

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

# ------------------------------------------------------------
# Monte Carlo x and y values
# ------------------------------------------------------------

x = as_1d(
    precession.eval_chieff(
        theta1,
        theta2,
        q,
        chi1,
        chi2
    )
)

y = as_1d(
    precession.eval_bracket_omega(
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

x_pi = chieff_resonance.copy()

x_zero = chieff_resonance.copy()

deltachi_pi, kappa_pi, chieff_pi = precession.angles_to_conserved(
    theta1_pi,
    theta2_pi,
    deltaphi_pi,
    r_resonance,
    q_resonance,
    chi1_resonance,
    chi2_resonance
)

deltachi_zero, kappa_zero, chieff_zero = precession.angles_to_conserved(
    theta1_zero,
    theta2_zero,
    deltaphi_zero,
    r_resonance,
    q_resonance,
    chi1_resonance,
    chi2_resonance
)

deltachi_pi = as_1d(deltachi_pi)
kappa_pi = as_1d(kappa_pi)
chieff_pi = as_1d(chieff_pi)

deltachi_zero = as_1d(deltachi_zero)
kappa_zero = as_1d(kappa_zero)
chieff_zero = as_1d(chieff_zero)

y_pi = as_1d(
    precession.eval_OmegaL(
        deltachi_pi,
        kappa_pi,
        r_resonance,
        chieff_pi,
        q_resonance,
        chi1_resonance,
        chi2_resonance
    )
)

y_zero = as_1d(
    precession.eval_OmegaL(
        deltachi_zero,
        kappa_zero,
        r_resonance,
        chieff_zero,
        q_resonance,
        chi1_resonance,
        chi2_resonance
    )
)



# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

x = as_1d(x)
y = as_1d(y)

finite = np.isfinite(x) & np.isfinite(y)

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

    mask = finite & quadrant_mask

    plt.scatter(
        x[mask],
        y[mask],
        s=2,
        color=point_color,
        linewidths=0
    )

    theta1_exact_array = np.array([theta1_exact])
    theta2_exact_array = np.array([theta2_exact])
    deltaphi_exact_array = np.array([0.0])

    r_exact = np.array([r_value])
    q_exact = np.array([q_value])
    chi1_exact = np.array([chi1_value])
    chi2_exact = np.array([chi2_value])

    try:

        x_exact = as_1d(
            precession.eval_chieff(
                theta1_exact_array,
                theta2_exact_array,
                q_exact,
                chi1_exact,
                chi2_exact
            )
        )[0]


        deltachi_exact, kappa_exact, chieff_exact = (
            precession.angles_to_conserved(
                theta1_exact_array,
                theta2_exact_array,
                deltaphi_exact_array,
                r_exact,
                q_exact,
                chi1_exact,
                chi2_exact
            )
        )

        y_exact = as_1d(
            precession.eval_OmegaL(
                    deltachi_exact,
                    kappa_exact,
                    r_exact,
                    chieff_exact,
                    q_exact,
                    chi1_exact,
                    chi2_exact
                )
        )[0]


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

finite_zero = np.isfinite(x_zero) & np.isfinite(y_zero)
finite_pi = np.isfinite(x_pi) & np.isfinite(y_pi)

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

plt.xlabel(r"$\chi_{\rm eff}$")
plt.ylabel(r"$\langle\Omega_L\rangle$")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()
