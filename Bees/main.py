import matplotlib.pyplot as plt
import numpy as np
from bee_movement import Bee
from plant import Plant
from typing import List, Tuple

L: int = 30
min_length: int = -L
max_length: int = L
num_plants: int = 100
num_bees: int = 50
num_hives: int = 1
v_min = 0
v_max = 5
v_mean = 24
v_std = 1.0

# Create Plant objects at random positions
plants: List[Plant] = [
    Plant(
        np.random.uniform(min_length, max_length),
        np.random.uniform(min_length, max_length),
    )
    for _ in range(num_plants)
]

# Create Beehives at random positions
beehives: List[Tuple[float, float]] = [
    (
        np.random.uniform(min_length, max_length),
        np.random.uniform(min_length, max_length),
    )
    for _ in range(num_hives)
]

bees: List[Bee] = [
    Bee(
        np.random.uniform(min_length, max_length),
        np.random.uniform(min_length, max_length),
        np.random.uniform(0, 2 * np.pi),
        np.clip(np.random.normal(v_mean, v_std), v_min, v_max),
    )
    for _ in range(num_bees)
]

# Draw the box
plt.figure(figsize=(10, 10))
plt.plot(
    [min_length, max_length, max_length, min_length, min_length],
    [min_length, min_length, max_length, max_length, min_length],
    "k-",
)

# Plot plants
plant_x: List[float] = [plant.x for plant in plants]
plant_y: List[float] = [plant.y for plant in plants]
plt.scatter(plant_x, plant_y, c="green", s=50, marker=".", label="Plant")

# Plot beehives
beehive_x: List[float] = [hive[0] for hive in beehives]
beehive_y: List[float] = [hive[1] for hive in beehives]
plt.scatter(
    beehive_x,
    beehive_y,
    c="#FCE205",
    s=100,
    edgecolors="black",
    marker="*",
    label="Bee Base",
)

# Plot bees
bee_x: List[float] = [bee.x for bee in bees]
bee_y: List[float] = [bee.y for bee in bees]
plt.scatter(
    bee_x, bee_y, c="#FCE205", s=70, marker="h", edgecolors="black", label="Bee"
)

# Plot velocity vectors for each bee
bee_u: List[float] = [bee.velocity * np.cos(bee.orientaion) for bee in bees]
bee_v: List[float] = [bee.velocity * np.sin(bee.orientaion) for bee in bees]
plt.quiver(
    bee_x,
    bee_y,
    bee_u,
    bee_v,
    angles="xy",
    scale_units="xy",
    scale=1,
    color="black",
    width=0.003,
    label="Velocity",
)

plt.xlim(min_length, max_length)
plt.ylim(min_length, max_length)
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Box with Randomly Placed Plants And Beehives")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
