import matplotlib.pyplot as plt
import numpy as np
from bee_movement import Bee
from plant import Plant
from typing import List, Tuple

L: int = 500
min_length: int = -L
max_length: int = L
num_plants: int = 100
num_bees: int = 50
num_hives: int = 1
v_mean = 24  # Avg bee speed
v_std = 1.0


def create_plants(num_plants: int, min_length: int, max_length: int) -> List[Plant]:
    """Create Plant objects at random positions."""
    return [
        Plant(
            np.random.uniform(min_length, max_length),
            np.random.uniform(min_length, max_length),
        )
        for _ in range(num_plants)
    ]


def create_beehives(
    num_hives: int, min_length: int, max_length: int
) -> List[Tuple[float, float]]:
    """Create beehive positions at random locations."""
    return [
        (
            np.random.uniform(min_length, max_length),
            np.random.uniform(min_length, max_length),
        )
        for _ in range(num_hives)
    ]


def create_bees(
    num_bees: int, min_length: int, max_length: int, v_mean: float, v_std: float
) -> List[Bee]:
    """Create Bee objects at random positions, orientations, and velocities."""
    return [
        Bee(
            np.random.uniform(min_length, max_length),
            np.random.uniform(min_length, max_length),
            np.random.uniform(0, 2 * np.pi),
            np.random.normal(v_mean, v_std),
        )
        for _ in range(num_bees)
    ]


def plot_simulation(
    plants: List[Plant],
    beehives: List[Tuple[float, float]],
    bees: List[Bee],
    min_length: int,
    max_length: int,
) -> None:
    """Plot the simulation box, plants, beehives, bees, and their velocity vectors."""
    plt.figure(figsize=(10, 10))
    plt.plot(
        [min_length, max_length, max_length, min_length, min_length],
        [min_length, min_length, max_length, max_length, min_length],
        "k-",
    )

    # Plot plants
    plant_x = [plant.x for plant in plants]
    plant_y = [plant.y for plant in plants]
    plt.scatter(plant_x, plant_y, c="green", s=100, marker=".", label="Plant")

    # Plot beehives
    beehive_x = [hive[0] for hive in beehives]
    beehive_y = [hive[1] for hive in beehives]
    plt.scatter(
        beehive_x,
        beehive_y,
        c="#FCE205",
        s=250,
        edgecolors="black",
        marker="*",
        label="Bee Base",
    )

    # Plot bees
    bee_x = [bee.x for bee in bees]
    bee_y = [bee.y for bee in bees]
    plt.scatter(
        bee_x, bee_y, c="#FCE205", s=25, marker="h", edgecolors="black", label="Bee"
    )

    # Plot velocity vectors for each bee
    bee_u = [bee.velocity * np.cos(bee.orientaion) for bee in bees]
    bee_v = [bee.velocity * np.sin(bee.orientaion) for bee in bees]
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


def main() -> None:
    plants = create_plants(num_plants, min_length, max_length)
    beehives = create_beehives(num_hives, min_length, max_length)
    bees = create_bees(num_bees, min_length, max_length, v_mean, v_std)
    plot_simulation(plants, beehives, bees, min_length, max_length)


if __name__ == "__main__":
    main()
