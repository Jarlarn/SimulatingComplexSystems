import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def simulate_fire(l, p=0.01, f=0.2, min_events=300):
    S = np.zeros((l, l))
    fire_sizes = []
    fire_count = 0
    steps = 0
    while fire_count < min_events:
        # Tree growth
        S[(np.random.rand(l, l) < p) & (S == 0)] = 1
        # Lightning strike
        lightning_location = (np.random.rand(2) * l).astype(int)
        if (
            S[lightning_location[0], lightning_location[1]] == 1
            and np.random.rand() < f
        ):
            S[lightning_location[0], lightning_location[1]] = 3
            fire_size = 0
            while np.any(S == 3):
                fire_indices = np.argwhere(S == 3)
                for i, j in fire_indices:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = (i + di) % l, (j + dj) % l  # periodic boundary
                            if S[ni, nj] == 1:
                                S[ni, nj] = 3
                    S[i, j] = 2
                fire_size += len(fire_indices)
            fire_sizes.append(fire_size)
            fire_count += 1
            S[S == 2] = 0
        steps += 1
    return np.array(fire_sizes)


def ccdf(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins)
    cumsum = np.cumsum(hist[::-1])[::-1]
    return bin_edges[:-1], cumsum / cumsum[0]


def power_law(x, a, c):
    return c * x ** (-a)


sizes = [16, 32, 64, 128, 256, 512]
p = 0.01
f = 0.2
repeats = 5
min_events = 300

for l in sizes:
    all_ccdf_x = []
    all_ccdf_y = []
    alphas = []
    plt.figure(figsize=(7, 5))
    for rep in range(repeats):
        fire_sizes = simulate_fire(l, p, f, min_events)
        rel_sizes = fire_sizes / (l * l)
        bins = np.logspace(np.log10(1 / (l * l)), 0, 50)
        x, y = ccdf(rel_sizes, bins)
        all_ccdf_x.append(x)
        all_ccdf_y.append(y)
        # Fit power law to tail (exclude smallest sizes)
        fit_mask = x > 5 / (l * l)
        try:
            popt, _ = curve_fit(
                power_law, x[fit_mask], y[fit_mask], bounds=([0, 0], [5, 10])
            )
            alphas.append(popt[0])
            plt.plot(x, y, label=f"Rep {rep+1}, α={popt[0]:.2f}")
        except Exception as e:
            plt.plot(x, y, label=f"Rep {rep+1}, fit failed")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Relative fire size (n/N²)")
    plt.ylabel("cCDF")
    plt.title(f"Moore Forest Fire Model, N={l}")
    plt.legend()
    # Add textbox with ᾱ ± Δα and fitting region
    mean_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)
    fit_region = f"Fit region: x > {5/(l*l):.2e}"
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
        f"N={l}: ᾱ ± Δα = {mean_alpha:.2f} ± {std_alpha:.2f} (fit region: x > {5/(l*l):.2e})"
    )
