import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def simulate_fire(l, p=0.01, f=0.2, min_events=300):
    """Simulate forest fire model and return fire sizes using von Neumann neighborhood."""
    S = np.zeros((l, l))
    fire_sizes = []
    fire_count = 0
    while fire_count < min_events:
        # Grow trees
        for i in range(l):
            for j in range(l):
                if S[i, j] == 0 and np.random.rand() < p:
                    S[i, j] = 1
        # Lightning strike
        i_strike = np.random.randint(l)
        j_strike = np.random.randint(l)
        if S[i_strike, j_strike] == 1 and np.random.rand() < f:
            S[i_strike, j_strike] = 3
            fire_size = 0
            while np.any(S == 3):
                burning = np.argwhere(S == 3)
                for i, j in burning:
                    # Spread fire to neighbors (Moore neighborhood, periodic boundaries)
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = (i + di) % l
                            nj = (j + dj) % l
                            if S[ni, nj] == 1:
                                S[ni, nj] = 3
                    S[i, j] = 2  # Burnt
                fire_size += len(burning)
            fire_sizes.append(fire_size)
            fire_count += 1
            S[S == 2] = 0  # Remove burnt trees
    return np.array(fire_sizes)


def ccdf(data, bins):
    """Calculate complementary cumulative distribution function (CCDF)."""
    ccdf_y = []
    for edge in bins[:-1]:
        ccdf_y.append(np.mean(data >= edge))
    return bins[:-1], np.array(ccdf_y)


def power_law(x, alpha, c):
    return c * x ** (1 - alpha)


sizes = [16, 32, 64, 128, 256, 512]
p = 0.01
f = 0.2
repeats = 5
min_events = 300

mean_alphas = []
std_alphas = []
inv_N = []

for l in sizes:
    alphas = []
    plt.figure(figsize=(7, 5))
    lower_bound = 0.01
    upper_bound = 0.2
    for rep in range(repeats):
        fire_sizes = simulate_fire(l, p, f, min_events)
        rel_sizes = fire_sizes / (l * l)
        bins = np.logspace(np.log10(1 / (l * l)), 0, 50)
        x, y = ccdf(rel_sizes, bins)
        # Fit power law to tail (exclude smallest sizes)
        fit_mask = (x > lower_bound) & (x < upper_bound)
        try:
            popt, _ = curve_fit(
                power_law, x[fit_mask], y[fit_mask], bounds=([0, 0], [5, 10])
            )
            alphas.append(popt[0])
            plt.plot(x, y, label=f"Rep {rep+1}, α={popt[0]:.2f}")
        except Exception:
            plt.plot(x, y, label=f"Rep {rep+1}, fit failed")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Relative fire size (n/N²)")
    plt.ylabel("cCDF")
    plt.title(f"Moore Forest Fire Model, N={l}")
    plt.legend()
    mean_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)
    fit_region = f"Fit region: {lower_bound} < x < {upper_bound}"
    textstr = f"Average α: {mean_alpha:.2f} ± {std_alpha:.2f}\n{fit_region}"
    plt.gca().text(
        0.05,
        0.05,  # bottom left
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    plt.tight_layout()
    plt.savefig(f"forest_fire_ccdf_N{l}.png")
    plt.show()
    print(
        f"N={l}: ᾱ ± Δα = {mean_alpha:.2f} ± {std_alpha:.2f} (Fit region: {lower_bound} < x < {upper_bound})"
    )
    mean_alphas.append(mean_alpha)
    std_alphas.append(std_alpha)
    inv_N.append(1 / l)

# Plot ¯αN vs N^{-1} with error bars and trendline
plt.figure(figsize=(6, 4))
plt.errorbar(inv_N, mean_alphas, yerr=std_alphas, fmt="o", capsize=5, label="Data")
coeffs = np.polyfit(inv_N, mean_alphas, 1)
trend_x = np.linspace(min(inv_N), max(inv_N), 100)
trend_y = np.polyval(coeffs, trend_x)
plt.plot(
    trend_x, trend_y, "r--", label=f"Trendline: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}"
)
plt.xlabel(r"$N^{-1}$")
plt.ylabel(r"$\overline{\alpha}_N$")
plt.title(r"Exponent $\overline{\alpha}_N$ vs $N^{-1}$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("alpha_vs_invN.png")
plt.show()
