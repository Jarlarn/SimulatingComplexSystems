import numpy as np
import matplotlib.pyplot as plt

N = 100
steps = 200
p_values = [0.45, 0.48, 0.50, 0.52, 0.55]
final_states = []
V1_list = []


def count_neighbors(lattice):
    neighbors = (
        np.roll(lattice, 1, axis=0)
        + np.roll(lattice, -1, axis=0)
        + np.roll(lattice, 1, axis=1)
        + np.roll(lattice, -1, axis=1)
        + np.roll(np.roll(lattice, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(lattice, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(lattice, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(lattice, -1, axis=0), -1, axis=1)
    )
    return neighbors


for p in p_values:
    lattice = np.random.choice([0, 1], size=(N, N), p=[1 - p, p])
    for step in range(steps):
        neighbors = count_neighbors(lattice)
        new_lattice = np.where(neighbors <= 3, 0, lattice)
        new_lattice = np.where(neighbors >= 5, 1, new_lattice)  
        lattice = new_lattice
    final_states.append(lattice.copy())
    V1_list.append(np.sum(lattice))

fig, axes = plt.subplots(1, len(p_values), figsize=(15, 3))
for i, (ax, state, p) in enumerate(zip(axes, final_states, p_values)):
    ax.imshow(state, cmap="binary")
    ax.set_title(f"p={p}, V1={V1_list[i]}")
    ax.axis("off")
fig.suptitle("Final State for Each p")
plt.tight_layout()
plt.savefig("Finalstate")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(p_values, V1_list, marker="o")
plt.xlabel("Initial fraction p")
plt.ylabel("Final number of 1s (V1)")
plt.title("V1 vs p")
plt.grid(True)
plt.savefig("V1")
plt.show()
