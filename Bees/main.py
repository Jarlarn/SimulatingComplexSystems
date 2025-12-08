import matplotlib.pyplot as plt
import numpy as np
from bee import Bee
from plant import Plant
from typing import List, Tuple, Dict
import matplotlib.animation as animation
import json
import os
from datetime import datetime

# Simulation parameters
L: int = 500
min_length: int = -L
max_length: int = L
num_plants: int = 50
num_bees: int = 200
num_hives: int = 2
v_mean = 24  # Avg bee speed
v_std = 1.0

# Simulation settings
dt = 0.05  # Time step
max_steps = 1000
attraction_strength = 0.4

# Results folder
RESULTS_FOLDER = "results"


def ensure_results_folder() -> None:
    """Ensure results folder exists."""
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print(f"Created '{RESULTS_FOLDER}' folder")


def get_simulation_parameters() -> Dict:
    """Get all current simulation parameters."""
    return {
        "simulation_bounds": {"min": min_length, "max": max_length},
        "num_plants": num_plants,
        "num_bees": num_bees,
        "num_hives": num_hives,
        "bee_velocity_mean": v_mean,
        "bee_velocity_std": v_std,
        "dt": dt,
        "max_steps": max_steps,
        "attraction_strength": attraction_strength,
    }


def save_run_results(
    metrics: Dict,
    run_name: str = None,
    parameters: Dict = None,
) -> str:
    """Save simulation results to a JSON file in the results folder."""
    ensure_results_folder()

    if parameters is None:
        parameters = get_simulation_parameters()

    # Generate run name with timestamp if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    # Combine parameters and metrics
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "parameters": parameters,
        "metrics": metrics,
    }

    # Save as JSON
    filepath = os.path.join(RESULTS_FOLDER, f"{run_name}.json")
    with open(filepath, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")

    return filepath


def create_results_summary() -> str:
    """Create a summary CSV of all results for easy comparison."""
    ensure_results_folder()

    summary_path = os.path.join(RESULTS_FOLDER, "summary.csv")

    # Collect all run files
    run_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith(".json")]

    if not run_files:
        print("No results to summarize")
        return summary_path

    # Extract data from each run
    summary_data = []
    for filename in sorted(run_files):
        filepath = os.path.join(RESULTS_FOLDER, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Flatten parameters and metrics for CSV
            row = {
                "run_name": data.get("run_name", ""),
                "timestamp": data.get("timestamp", ""),
            }

            # Add parameters
            if "parameters" in data:
                for key, value in data["parameters"].items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            row[f"param_{key}_{subkey}"] = subvalue
                    else:
                        row[f"param_{key}"] = value

            # Add metrics
            if "metrics" in data:
                for key, value in data["metrics"].items():
                    row[f"metric_{key}"] = value

            summary_data.append(row)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filename}")
            continue

    if summary_data:
        # Write CSV
        import csv

        with open(summary_path, "w", newline="") as f:
            fieldnames = sorted(set(k for d in summary_data for k in d.keys()))
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)

        print(f"✓ Summary saved to: {summary_path}")

    return summary_path


def display_saved_results() -> None:
    """Display all saved results with their key metrics."""
    ensure_results_folder()

    run_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith(".json")]

    if not run_files:
        print("No saved results found")
        return

    print("\n" + "=" * 80)
    print("SAVED SIMULATION RESULTS")
    print("=" * 80)

    for filename in sorted(run_files):
        filepath = os.path.join(RESULTS_FOLDER, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            print(f"\n📊 {data['run_name']}")
            print(f"   Timestamp: {data['timestamp']}")

            params = data.get("parameters", {})
            print(f"   Parameters:")
            print(
                f"      Bees: {params.get('num_bees', 'N/A')} | Plants: {params.get('num_plants', 'N/A')} | Hives: {params.get('num_hives', 'N/A')}"
            )
            print(
                f"      Attraction Strength: {params.get('attraction_strength', 'N/A')}"
            )

            metrics = data.get("metrics", {})
            print(f"   Results:")
            print(f"      Total Trips: {metrics.get('total_trips', 'N/A')}")
            print(
                f"      Pollen Collected: {metrics.get('total_pollen_collected', 'N/A'):.0f}"
            )
            print(
                f"      Plants Visited: {metrics.get('plants_visited', 'N/A')}/{params.get('num_plants', 'N/A')} ({metrics.get('plant_visitation_rate', 'N/A')*100 if isinstance(metrics.get('plant_visitation_rate'), (int, float)) else 'N/A':.1f}%)"
            )
            print(
                f"      Pollen per Distance: {metrics.get('pollen_per_distance', 'N/A'):.4f}"
            )
        except json.JSONDecodeError:
            print(f"   ⚠ Warning: Could not parse {filename}")

    print("\n" + "=" * 80 + "\n")


def create_plants(
    num_plants: int,
    min_length: int,
    max_length: int,
    max_pollen: int = 100,
    attraction_radius: float = 60.0,
) -> List[Plant]:
    """Create Plant objects at random positions."""
    return [
        Plant(
            np.random.uniform(min_length, max_length),
            np.random.uniform(min_length, max_length),
            max_pollen=max_pollen,
            attraction_radius=attraction_radius,
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
    num_bees: int,
    min_length: int,
    max_length: int,
    beehives: List[Tuple[float, float]],
    v_mean: float,
    v_std: float,
    detection_range: float = 75.0,
) -> List[Bee]:
    """Create Bee objects starting at hives."""
    bees = []
    for i in range(num_bees):
        # Assign bees to hives (distribute evenly)
        hive = beehives[i % len(beehives)]

        # Start near the hive with some random offset
        offset = np.random.uniform(-20, 20, 2)

        bees.append(
            Bee(
                hive[0] + offset[0],
                hive[1] + offset[1],
                np.random.uniform(0, 2 * np.pi),
                np.random.normal(v_mean, v_std),
                hive_pos=hive,
                detection_range=detection_range,
            )
        )
    return bees


def simulation_step(
    bees: List[Bee],
    plants: List[Plant],
    dt: float,
    min_length: int,
    max_length: int,
    attraction_strength: float = 0.3,
) -> None:
    """Execute one step of the simulation."""

    for bee in bees:
        if bee.has_pollen:
            # Return to hive with pollen
            bee.move_towards_target(bee.hive_x, bee.hive_y, dt)

            # Deposit pollen if at hive
            if bee.is_at_hive():
                bee.deposit_pollen()
                bee.last_plant_visited = None  # Reset memory
        else:
            # Look for plants
            nearby_plants = bee.detect_plants(plants)

            if nearby_plants:
                # Move towards closest plant with pollen
                target_plant, dist = nearby_plants[0]

                if dist < bee.collection_radius:
                    # Collect pollen
                    bee.collect_pollen(target_plant)
                else:
                    # Move towards plant
                    bee.move_towards_target(target_plant.x, target_plant.y, dt)
            else:
                # Explore with Lévy walk and plant attraction
                bee.move_with_attraction(plants, dt, attraction_strength)

        # Keep bee within boundaries
        bee.enforce_boundaries(min_length, max_length, min_length, max_length)


def calculate_pollination_metrics(
    bees: List[Bee], plants: List[Plant], steps: int
) -> Dict:
    """Calculate efficiency metrics for the pollination simulation."""

    # Bee metrics
    total_trips = sum(bee.trips_completed for bee in bees)
    total_pollen_by_bees = sum(bee.total_pollen_collected for bee in bees)
    avg_distance_per_bee = np.mean([bee.distance_traveled for bee in bees])
    avg_trips_per_bee = np.mean([bee.trips_completed for bee in bees])

    # Plant metrics
    total_visits = sum(plant.times_visited for plant in plants)
    plants_visited = sum(1 for plant in plants if plant.times_visited > 0)
    visitation_rate = plants_visited / len(plants)
    avg_pollen_level = np.mean([plant.get_pollen_ratio() for plant in plants])

    # Efficiency metrics
    pollen_per_distance = (
        total_pollen_by_bees / (avg_distance_per_bee * len(bees))
        if avg_distance_per_bee > 0
        else 0
    )
    pollen_per_bee_per_step = total_pollen_by_bees / (len(bees) * steps)

    return {
        "total_trips": total_trips,
        "total_pollen_collected": total_pollen_by_bees,
        "avg_distance_per_bee": avg_distance_per_bee,
        "avg_trips_per_bee": avg_trips_per_bee,
        "total_plant_visits": total_visits,
        "plants_visited": plants_visited,
        "plant_visitation_rate": visitation_rate,
        "avg_pollen_level": avg_pollen_level,
        "pollen_per_distance": pollen_per_distance,
        "pollen_per_bee_per_step": pollen_per_bee_per_step,
    }


def plot_simulation(
    plants: List[Plant],
    beehives: List[Tuple[float, float]],
    bees: List[Bee],
    min_length: int,
    max_length: int,
    step: int = 0,
) -> None:
    """Plot the simulation box, plants, beehives, bees, and their velocity vectors."""
    plt.figure(figsize=(12, 10))

    # Plot boundary
    plt.plot(
        [min_length, max_length, max_length, min_length, min_length],
        [min_length, min_length, max_length, max_length, min_length],
        "k-",
        linewidth=2,
    )

    # Plot plants with size based on pollen level
    plant_x = [plant.x for plant in plants]
    plant_y = [plant.y for plant in plants]
    plant_pollen = [plant.get_pollen_ratio() for plant in plants]

    # Color plants by pollen level
    scatter_plants = plt.scatter(
        plant_x,
        plant_y,
        c=plant_pollen,
        s=100,
        marker=".",
        cmap="Greens",
        vmin=0,
        vmax=1,
        label="Plant",
        edgecolors="darkgreen",
        linewidth=0.5,
    )
    plt.colorbar(scatter_plants, label="Pollen Level", shrink=0.8)

    # Plot beehives
    beehive_x = [hive[0] for hive in beehives]
    beehive_y = [hive[1] for hive in beehives]
    plt.scatter(
        beehive_x,
        beehive_y,
        c="#FCE205",
        s=400,
        edgecolors="black",
        marker="*",
        label="Hive",
        linewidth=2,
    )

    # Plot bees - different colors for those with/without pollen
    bees_with_pollen = [bee for bee in bees if bee.has_pollen]
    bees_without_pollen = [bee for bee in bees if not bee.has_pollen]

    if bees_without_pollen:
        bee_x = [bee.x for bee in bees_without_pollen]
        bee_y = [bee.y for bee in bees_without_pollen]
        plt.scatter(
            bee_x,
            bee_y,
            c="#FCE205",
            s=40,
            marker="h",
            edgecolors="black",
            label="Bee (foraging)",
            linewidth=0.5,
        )

        # Plot velocity vectors
        bee_u = [bee.velocity * np.cos(bee.orientation) for bee in bees_without_pollen]
        bee_v = [bee.velocity * np.sin(bee.orientation) for bee in bees_without_pollen]
        plt.quiver(
            bee_x,
            bee_y,
            bee_u,
            bee_v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="orange",
            width=0.003,
            alpha=0.6,
        )

    if bees_with_pollen:
        bee_x = [bee.x for bee in bees_with_pollen]
        bee_y = [bee.y for bee in bees_with_pollen]
        plt.scatter(
            bee_x,
            bee_y,
            c="orange",
            s=40,
            marker="h",
            edgecolors="black",
            label="Bee (with pollen)",
            linewidth=0.5,
        )

        # Plot velocity vectors
        bee_u = [bee.velocity * np.cos(bee.orientation) for bee in bees_with_pollen]
        bee_v = [bee.velocity * np.sin(bee.orientation) for bee in bees_with_pollen]
        plt.quiver(
            bee_x,
            bee_y,
            bee_u,
            bee_v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
            width=0.003,
            alpha=0.6,
        )

    plt.xlim(min_length, max_length)
    plt.ylim(min_length, max_length)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Bee Pollination Simulation - Step {step}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def run_simulation(
    num_steps: int = 1000,
    visualize_interval: int = 100,
    print_metrics: bool = True,
) -> Tuple[List[Bee], List[Plant], Dict]:
    """Run the complete simulation."""

    # Initialize
    plants = create_plants(num_plants, min_length, max_length)
    beehives = create_beehives(num_hives, min_length, max_length)
    bees = create_bees(num_bees, min_length, max_length, beehives, v_mean, v_std)

    print(f"Starting simulation with {num_bees} bees and {num_plants} plants...")
    print(
        f"Simulation bounds: [{min_length}, {max_length}] x [{min_length}, {max_length}]"
    )
    print(f"Running {num_steps} steps with dt={dt}\n")

    # Run simulation
    for step in range(num_steps):
        simulation_step(bees, plants, dt, min_length, max_length, attraction_strength)

        # Visualize at intervals
        if visualize_interval and step % visualize_interval == 0:
            print(f"Step {step}/{num_steps}")
            if step > 0:  # Skip initial state
                metrics = calculate_pollination_metrics(bees, plants, step)
                print(f"  Trips completed: {metrics['total_trips']}")
                print(f"  Pollen collected: {metrics['total_pollen_collected']:.1f}")
                print(
                    f"  Plants visited: {metrics['plants_visited']}/{len(plants)} ({metrics['plant_visitation_rate']*100:.1f}%)"
                )
                print()

    # Final metrics
    metrics = calculate_pollination_metrics(bees, plants, num_steps)

    if print_metrics:
        print("\n" + "=" * 60)
        print("FINAL SIMULATION METRICS")
        print("=" * 60)
        print(f"\nBee Performance:")
        print(f"  Total trips completed: {metrics['total_trips']}")
        print(f"  Average trips per bee: {metrics['avg_trips_per_bee']:.2f}")
        print(f"  Total pollen collected: {metrics['total_pollen_collected']:.1f}")
        print(f"  Average distance per bee: {metrics['avg_distance_per_bee']:.1f}")

        print(f"\nPlant Coverage:")
        print(f"  Total plant visits: {metrics['total_plant_visits']}")
        print(f"  Plants visited: {metrics['plants_visited']}/{len(plants)}")
        print(f"  Visitation rate: {metrics['plant_visitation_rate']*100:.1f}%")
        print(f"  Average pollen level: {metrics['avg_pollen_level']*100:.1f}%")

        print(f"\nEfficiency Metrics:")
        print(f"  Pollen per distance: {metrics['pollen_per_distance']:.4f}")
        print(f"  Pollen per bee per step: {metrics['pollen_per_bee_per_step']:.4f}")
        print("=" * 60 + "\n")

    return bees, plants, metrics


def main() -> None:
    """Main entry point."""
    # Run simulation
    bees, plants, metrics = run_simulation(
        num_steps=max_steps,
        visualize_interval=200,
        print_metrics=True,
    )

    # Create beehives list for visualization
    beehives = list(set([(bee.hive_x, bee.hive_y) for bee in bees]))

    # Save results with current parameters
    run_name = f"run_{num_bees}bees_{num_plants}plants_{num_hives}hives"
    save_run_results(metrics, run_name=run_name)

    # Update summary CSV
    create_results_summary()

    # Display all saved results
    display_saved_results()

    # Final visualization
    plot_simulation(plants, beehives, bees, min_length, max_length, step=max_steps)
    plt.show()


if __name__ == "__main__":
    main()
