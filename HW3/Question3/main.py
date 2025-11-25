import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 2000
L = 100
d = 0.9
I0 = 10
steps = 1000  # Max steps to avoid infinite loops

# SIR parameters for both types
params = {"A": {"beta": 0.8, "gamma": 0.02}, "B": {"beta": 0.1, "gamma": 0.01}}


def periodic_boundary(pos, L):
    return pos % L


def run_simulation(beta, gamma, seed=None):
    rng = np.random.default_rng(seed)
    # States: 0 = S, 1 = I, 2 = R
    states = np.zeros(N, dtype=int)
    states[:I0] = 1  # Infect I0 agents
    rng.shuffle(states)

    # Random initial positions
    positions = rng.uniform(0, L, size=(N, 2))

    S_hist, I_hist, R_hist = [], [], []

    for t in range(steps):
        S_hist.append(np.sum(states == 0))
        I_hist.append(np.sum(states == 1))
        R_hist.append(np.sum(states == 2))

        if I_hist[-1] == 0:
            break

        # Move agents (random walk)
        move = rng.uniform(-1, 1, size=(N, 2))
        move = np.where(rng.random((N, 1)) < d, move, 0)
        positions = periodic_boundary(positions + move, L)

        # Infection step
        for i in np.where(states == 1)[0]:
            # Infect susceptible agents within distance 1
            dists = np.linalg.norm(positions - positions[i], axis=1)
            susceptible = np.where((states == 0) & (dists < 1))[0]
            for j in susceptible:
                if rng.random() < beta:
                    states[j] = 1

        # Recovery step
        infected = np.where(states == 1)[0]
        recoveries = rng.random(len(infected)) < gamma
        states[infected[recoveries]] = 2

    return np.array(S_hist), np.array(I_hist), np.array(R_hist)


def run_simulation_heterogeneous(beta1, gamma1, beta2, gamma2, seed=None):
    rng = np.random.default_rng(seed)
    N1 = N // 2
    N2 = N - N1
    # States: 0 = S, 1 = I, 2 = R
    states = np.zeros(N, dtype=int)
    states[:I0] = 1  # Infect I0 agents
    rng.shuffle(states)
    # Types: 0 = type 1, 1 = type 2
    types = np.zeros(N, dtype=int)
    types[N1:] = 1
    rng.shuffle(types)

    positions = rng.uniform(0, L, size=(N, 2))

    S_hist, I_hist, R_hist = [], [], []

    for t in range(steps):
        S_hist.append(np.sum(states == 0))
        I_hist.append(np.sum(states == 1))
        R_hist.append(np.sum(states == 2))

        if I_hist[-1] == 0:
            break

        # Move agents (random walk)
        move = rng.uniform(-1, 1, size=(N, 2))
        move = np.where(rng.random((N, 1)) < d, move, 0)
        positions = periodic_boundary(positions + move, L)

        # Infection step
        for i in np.where(states == 1)[0]:
            dists = np.linalg.norm(positions - positions[i], axis=1)
            susceptible = np.where((states == 0) & (dists < 1))[0]
            for j in susceptible:
                # Use correct beta for the susceptible agent
                beta = beta1 if types[j] == 0 else beta2
                if rng.random() < beta:
                    states[j] = 1

        # Recovery step
        infected = np.where(states == 1)[0]
        for idx in infected:
            gamma = gamma1 if types[idx] == 0 else gamma2
            if rng.random() < gamma:
                states[idx] = 2

    return np.array(S_hist), np.array(I_hist), np.array(R_hist)


def plot_results(results, label, save=False):
    S, I, R = results
    t = np.arange(len(S))
    plt.figure(figsize=(10, 5))
    plt.plot(t, S, label="Susceptible")
    plt.plot(t, I, label="Infected")
    plt.plot(t, R, label="Recovered")
    plt.xlabel("Time step")
    plt.ylabel("Number of agents")
    plt.legend()
    plt.title(f"SIR Model - Case {label}")
    plt.tight_layout()
    if save:
        plt.savefig(f"SIR_case_{label}.png")
    plt.show()


def run_experiment(case_label):
    beta = params[case_label]["beta"]
    gamma = params[case_label]["gamma"]
    S_runs, I_runs, R_runs = [], [], []
    for rep in range(6):
        S, I, R = run_simulation(beta, gamma, seed=rep)
        S_runs.append(S)
        I_runs.append(I)
        R_runs.append(R)
    # Pad with last value for shorter runs
    max_len = max(map(len, S_runs))
    S_runs = [np.pad(S, (0, max_len - len(S)), "edge") for S in S_runs]
    I_runs = [np.pad(I, (0, max_len - len(I)), "edge") for I in I_runs]
    R_runs = [np.pad(R, (0, max_len - len(R)), "edge") for R in R_runs]
    S_mean = np.mean(S_runs, axis=0)
    I_mean = np.mean(I_runs, axis=0)
    R_mean = np.mean(R_runs, axis=0)
    return S_mean, I_mean, R_mean


def run_experiment_heterogeneous():
    beta1, gamma1 = params["A"]["beta"], params["A"]["gamma"]
    beta2, gamma2 = params["B"]["beta"], params["B"]["gamma"]
    S_runs, I_runs, R_runs = [], [], []
    for rep in range(6):
        S, I, R = run_simulation_heterogeneous(beta1, gamma1, beta2, gamma2, seed=rep)
        S_runs.append(S)
        I_runs.append(I)
        R_runs.append(R)
    max_len = max(map(len, S_runs))
    S_runs = [np.pad(S, (0, max_len - len(S)), "edge") for S in S_runs]
    I_runs = [np.pad(I, (0, max_len - len(I)), "edge") for I in I_runs]
    R_runs = [np.pad(R, (0, max_len - len(R)), "edge") for R in R_runs]
    S_mean = np.mean(S_runs, axis=0)
    I_mean = np.mean(I_runs, axis=0)
    R_mean = np.mean(R_runs, axis=0)
    return S_mean, I_mean, R_mean


if __name__ == "__main__":
    # for case in ["A", "B"]:
    #     results = run_experiment(case)
    #     plot_results(results, case, save=True)
    # Part 2: Heterogeneous population
    results_D = run_experiment_heterogeneous()
    plot_results(results_D, "D", save=True)
