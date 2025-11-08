import numpy as np
import matplotlib.pyplot as plt
import time

N = 100
H = 0
J = 1
KB = 1
T = 2.3
S1, S2 = 1, 1
dhalf = 3
dhalf_values = [3, 5, 7, 10]
steps = 6000


def calculate_energy(lattice):
    energy = 0.0
    for i in range(N):
        for j in range(N):
            # Periodic boundary conditions
            spin = lattice[i, j]
            neighbors = (
                lattice[(i + 1) % N, j]
                + lattice[(i - 1) % N, j]
                + lattice[i, (j + 1) % N]
                + lattice[i, (j - 1) % N]
            )
            energy += spin * neighbors

    e_tot = -J / 2 * (1 / N**2) * energy
    return e_tot


def monte_carlo_step(lattice):
    beta = 1 / (KB * T)
    for _ in range(N * N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        neighbors = (
            lattice[(i + 1) % N, j]
            + lattice[(i - 1) % N, j]
            + lattice[i, (j + 1) % N]
            + lattice[i, (j - 1) % N]
        )
        # Calculate energies for spin up (+1) and spin down (-1)
        E_plus = -(H + J * neighbors)
        E_minus = -(H - J * neighbors)
        # Probabilities for spin up and down
        p_plus = np.exp(-beta * E_plus)
        p_minus = np.exp(-beta * E_minus)
        prob_up = p_plus / (p_plus + p_minus)
        # Set spin according to probability
        lattice[i, j] = 1 if np.random.rand() < prob_up else -1


lattice_fig, lattice_axes = plt.subplots(1, 4, figsize=(16, 4))
etot_fig, etot_axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, dhalf in enumerate(dhalf_values):
    lattice = np.random.choice([-1, 1], size=(N, N))
    N1 = int(N // 2 - dhalf)
    N2 = int(N // 2 + dhalf)
    lattice[N1, :] = S1
    lattice[N2, :] = S2

    etot_history = []
    start_time = time.time()
    for step in range(steps):
        monte_carlo_step(lattice)
        etot = calculate_energy(lattice)
        etot_history.append(etot)
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time_per_step = elapsed / (step + 1)
            remaining_steps = steps - (step + 1)
            est_remaining = avg_time_per_step * remaining_steps
            progress = (step + 1) / steps * 100
            print(
                f"dhalf={dhalf}: Step {step + 1}/{steps} ({progress:.1f}% complete) | "
                f"Elapsed: {elapsed:.1f}s | Est. time left: {est_remaining:.1f}s"
            )

    # Plot lattice in its own figure
    ax_lattice = lattice_axes[idx]
    ax_lattice.imshow(lattice, cmap="bwr", vmin=-1, vmax=1)
    ax_lattice.set_title(f"dhalf = {dhalf}, S1 = {S1}, S2 = {S2}")
    ax_lattice.axis("off")

    # Plot etot history in its own figure
    ax_etot = etot_axes[idx]
    ax_etot.plot(range(1, steps + 1), etot_history, label="etot")
    equilibrium_etot = float(np.mean(etot_history[-1000:]))
    ax_etot.axhline(
        equilibrium_etot,
        color="red",
        linestyle="--",
        label=f"Equilibrium: {equilibrium_etot:.4f}",
    )
    ax_etot.set_xlabel("Step")
    ax_etot.set_ylabel("etot")
    ax_etot.legend()
    ax_etot.set_title(f"etot vs step (dhalf={dhalf}, S1={S1}, S2={S2})")

lattice_fig.tight_layout()
lattice_fig.savefig("Question2_lattices", dpi=600)
etot_fig.tight_layout()
etot_fig.savefig("Question2_etot_history", dpi=600)
plt.show()
