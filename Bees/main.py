import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bee import Bee
from plant import Plant
from typing import List, Tuple, Dict, Optional
import matplotlib.animation as animation
import json
import os
from datetime import datetime

# Each bee can visit 75 plants before reaching capacity.
# Each plant holds 3 mg pollen
# Each visit collects 0.7 mg pollen
# Simulation parameters
L: int = 500  # 1 Acre
min_length: int = -L
max_length: int = L
num_plants: int = 10
num_hives: int = 1
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


def get_next_run_folder() -> str:
    """Create and return the next run folder path (results/runN)."""
    ensure_results_folder()
    existing = [
        d
        for d in os.listdir(RESULTS_FOLDER)
        if os.path.isdir(os.path.join(RESULTS_FOLDER, d)) and d.startswith("run")
    ]
    indices = []
    for name in existing:
        try:
            indices.append(int(name[3:]))
        except ValueError:
            continue
    next_idx = (max(indices) + 1) if indices else 1
    run_dir = os.path.join(RESULTS_FOLDER, f"run{next_idx}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"✓ Sweep directory: {run_dir}")
    return run_dir


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
    run_name: Optional[str] = None,
    parameters: Optional[Dict] = None,
    base_dir: Optional[str] = None,
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
    target_dir = base_dir if base_dir else RESULTS_FOLDER
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, f"{run_name}.json")
    with open(filepath, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")

    return filepath


def create_results_summary(base_dir: Optional[str] = None) -> str:
    """Create a summary CSV of JSON results for easy comparison.
    If base_dir is provided, summarize that folder; otherwise summarize top-level results.
    """
    ensure_results_folder()

    target_dir = base_dir if base_dir else RESULTS_FOLDER
    summary_path = os.path.join(target_dir, "summary.csv")

    # Collect all run files
    run_files = [f for f in os.listdir(target_dir) if f.endswith(".json")]

    if not run_files:
        print("No results to summarize")
        return summary_path

    # Extract data from each run
    summary_data = []
    for filename in sorted(run_files):
        filepath = os.path.join(target_dir, filename)
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
    bees = create_bees(num_bees, beehives, v_mean, v_std)

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


def run_param_sweep(
    bees_list: List[int],
    plants_list: List[int],
    steps: int = 1000,
    visualize_interval: int = 0,
    repeats: int = 1,
) -> Tuple[pd.DataFrame, str]:
    """Run multiple simulations over combinations and return aggregated results plus sweep dir."""
    ensure_results_folder()
    sweep_dir = get_next_run_folder()
    records = []

    global num_bees, num_plants

    for plants in plants_list:
        for bees in bees_list:
            for r in range(repeats):
                num_plants = plants
                num_bees = bees

                _, _, metrics = run_simulation(
                    num_steps=steps,
                    visualize_interval=visualize_interval,
                    print_metrics=False,
                )

                run_name = f"sweep_{bees}bees_{plants}plants_r{r}"
                save_run_results(metrics, run_name=run_name, base_dir=sweep_dir)

                records.append(
                    {
                        "bees": bees,
                        "plants": plants,
                        "repeat": r,
                        "total_trips": metrics["total_trips"],
                        "total_pollen_collected": metrics["total_pollen_collected"],
                        "avg_distance_per_bee": metrics["avg_distance_per_bee"],
                        "avg_trips_per_bee": metrics["avg_trips_per_bee"],
                        "total_plant_visits": metrics["total_plant_visits"],
                        "plants_visited": metrics["plants_visited"],
                        "visitation_rate": metrics["plant_visitation_rate"],
                        "avg_pollen_level": metrics["avg_pollen_level"],
                        "pollen_per_distance": metrics["pollen_per_distance"],
                        "pollen_per_bee_per_step": metrics["pollen_per_bee_per_step"],
                    }
                )

    df = pd.DataFrame.from_records(records)
    sweep_csv = os.path.join(sweep_dir, "param_sweep.csv")
    df.to_csv(sweep_csv, index=False)
    print(f"✓ Parameter sweep saved: {sweep_csv}")

    return df, sweep_dir


def plot_param_sweep(df: pd.DataFrame, base_dir: Optional[str] = None) -> None:
    """Create summary plots across parameter combinations and optionally save to base_dir."""
    if df.empty:
        print("No data to plot.")
        return

    # Aggregate over repeats
    agg = df.groupby(["bees", "plants"], as_index=False).mean(numeric_only=True)

    # Prepare pivot tables for heatmaps
    pivot_visit = agg.pivot(index="plants", columns="bees", values="visitation_rate")
    pivot_pollen = agg.pivot(
        index="plants", columns="bees", values="total_pollen_collected"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Heatmap: Visitation rate
    im0 = axes[0, 0].imshow(
        pivot_visit.values, aspect="auto", cmap="viridis", vmin=0, vmax=1
    )
    axes[0, 0].set_title("Visitation Rate")
    axes[0, 0].set_xlabel("Bees")
    axes[0, 0].set_ylabel("Plants")
    axes[0, 0].set_xticks(range(len(pivot_visit.columns)))
    axes[0, 0].set_xticklabels(pivot_visit.columns)
    axes[0, 0].set_yticks(range(len(pivot_visit.index)))
    axes[0, 0].set_yticklabels(pivot_visit.index)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Heatmap: Total pollen collected
    im1 = axes[0, 1].imshow(pivot_pollen.values, aspect="auto", cmap="magma")
    axes[0, 1].set_title("Total Pollen Collected")
    axes[0, 1].set_xlabel("Bees")
    axes[0, 1].set_ylabel("Plants")
    axes[0, 1].set_xticks(range(len(pivot_pollen.columns)))
    axes[0, 1].set_xticklabels(pivot_pollen.columns)
    axes[0, 1].set_yticks(range(len(pivot_pollen.index)))
    axes[0, 1].set_yticklabels(pivot_pollen.index)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Line plot: Efficiency vs bees (per plants bucket)
    for plants_val in sorted(agg["plants"].unique()):
        sub = agg[agg["plants"] == plants_val].sort_values("bees")
        axes[1, 0].plot(
            sub["bees"], sub["pollen_per_bee_per_step"], label=f"plants={plants_val}"
        )
    axes[1, 0].set_title("Efficiency (pollen per bee per step)")
    axes[1, 0].set_xlabel("Bees")
    axes[1, 0].set_ylabel("Efficiency")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # Line plot: Visitation vs plants (per bees bucket)
    for bees_val in sorted(agg["bees"].unique()):
        sub = agg[agg["bees"] == bees_val].sort_values("plants")
        axes[1, 1].plot(sub["plants"], sub["visitation_rate"], label=f"bees={bees_val}")
    axes[1, 1].set_title("Visitation Rate by Plants")
    axes[1, 1].set_xlabel("Plants")
    axes[1, 1].set_ylabel("Visitation rate")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()

    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
        fig_path = os.path.join(base_dir, "param_sweep_plot.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved to: {fig_path}")

    plt.show()


def main() -> None:
    """Main entry point."""
    # Define parameter grid
    bees_list = [20, 30, 40, 60, 70, 80, 120, 160, 200]
    plants_list = [10, 25, 50, 75, 100]

    # Run sweep
    df, sweep_dir = run_param_sweep(
        bees_list=bees_list,
        plants_list=plants_list,
        steps=max_steps,
        visualize_interval=0,  # disable per-run prints for speed
        repeats=10,  # averaging over multiple runs to smooth randomness
    )

    # Update summary CSV within the latest sweep folder as well as top-level
    # Top-level summary remains for all-time results; per-sweep summary lives inside runN
    # Determine latest sweep dir for convenience
    run_dirs = [
        d
        for d in os.listdir(RESULTS_FOLDER)
        if os.path.isdir(os.path.join(RESULTS_FOLDER, d)) and d.startswith("run")
    ]
    if run_dirs:
        indices = []
        for name in run_dirs:
            try:
                indices.append((int(name[3:]), name))
            except ValueError:
                continue
        if indices:
            latest_name = max(indices)[1]
            create_results_summary(os.path.join(RESULTS_FOLDER, latest_name))
    create_results_summary()

    # Plot
    plot_param_sweep(df, base_dir=sweep_dir)


if __name__ == "__main__":
    main()
