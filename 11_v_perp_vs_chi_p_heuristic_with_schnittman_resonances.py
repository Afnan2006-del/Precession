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


def corrected_v_perp(
    theta1_input,
    theta2_input,
    deltaphi_input,
    q_input,
    chi1_input,
    chi2_input
):
    """
    Package v_perp quantity:
        H * eta**2 * Delta_parallel

    This correction is used only in the three v_perp scripts.
    """

    theta1_input = as_1d(theta1_input)
    theta2_input = as_1d(theta2_input)
    deltaphi_input = as_1d(deltaphi_input)
    q_input = as_1d(q_input)
    chi1_input = as_1d(chi1_input)
    chi2_input = as_1d(chi2_input)

    eta = precession.eval_eta(q_input)

    Lvec, S1vec, S2vec = precession.angles_to_Lframe(
        theta1_input,
        theta2_input,
        deltaphi_input,
        np.ones(theta1_input.size),
        q_input,
        chi1_input,
        chi2_input
    )

    hatL = precession.normalize_nested(Lvec)
    hatS1 = precession.normalize_nested(S1vec)
    hatS2 = precession.normalize_nested(S2vec)

    Delta = -precession.scalar_nested(
        1.0/(1.0 + q_input),
        precession.scalar_nested(q_input*chi2_input, hatS2)
        - precession.scalar_nested(chi1_input, hatS1)
    )

    Delta_parallel = precession.dot_nested(Delta, hatL)

    H = 6.9e3

    return H * eta**2 * Delta_parallel / 299792.458

# ------------------------------------------------------------
# Monte Carlo x and y values
# ------------------------------------------------------------

x = as_1d(
    precession.eval_chip(
        theta1=theta1,
        theta2=theta2,
        deltaphi=deltaphi,
        r=r,
        q=q,
        chi1=chi1,
        chi2=chi2,
        which="heuristic"
    )
)

y = corrected_v_perp(
    theta1,
    theta2,
    deltaphi,
    q,
    chi1,
    chi2
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

x_pi = as_1d(
    precession.eval_chip(
        theta1=theta1_pi,
        theta2=theta2_pi,
        deltaphi=deltaphi_pi,
        r=r_resonance,
        q=q_resonance,
        chi1=chi1_resonance,
        chi2=chi2_resonance,
        which="heuristic"
    )
)

x_zero = as_1d(
    precession.eval_chip(
        theta1=theta1_zero,
        theta2=theta2_zero,
        deltaphi=deltaphi_zero,
        r=r_resonance,
        q=q_resonance,
        chi1=chi1_resonance,
        chi2=chi2_resonance,
        which="heuristic"
    )
)

y_pi = corrected_v_perp(
    theta1_pi,
    theta2_pi,
    deltaphi_pi,
    q_resonance,
    chi1_resonance,
    chi2_resonance
)

y_zero = corrected_v_perp(
    theta1_zero,
    theta2_zero,
    deltaphi_zero,
    q_resonance,
    chi1_resonance,
    chi2_resonance
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
            precession.eval_chip(
                theta1=theta1_exact_array,
                theta2=theta2_exact_array,
                deltaphi=deltaphi_exact_array,
                r=r_exact,
                q=q_exact,
                chi1=chi1_exact,
                chi2=chi2_exact,
                which="heuristic"
            )
        )[0]


        y_exact = corrected_v_perp(
            theta1_exact_array,
            theta2_exact_array,
            deltaphi_exact_array,
            q_exact,
            chi1_exact,
            chi2_exact
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

plt.xlabel(r"$\chi_p$, heuristic")
plt.ylabel(r"$v_\perp$")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()
