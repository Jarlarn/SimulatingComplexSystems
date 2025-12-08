import numpy as np
from typing import List, Tuple, Optional


class Bee:
    def __init__(
        self,
        x: float,
        y: float,
        orientation: float,
        velocity: float,
        hive_pos: Tuple[float, float],
        detection_range: float = 75.0,
        collection_radius: float = 5.0,
    ) -> None:
        self.x: float = x
        self.y: float = y
        self.orientation: float = orientation
        self.velocity: float = velocity
        self.hive_x: float = hive_pos[0]
        self.hive_y: float = hive_pos[1]
        self.detection_range: float = detection_range
        self.collection_radius: float = collection_radius

        # Pollen state
        self.has_pollen: bool = False
        self.pollen_amount: float = 0.0
        self.pollen_capacity: float = 20.0

        # Memory of last plant visited
        self.last_plant_visited: Optional[object] = None

        # Statistics
        self.trips_completed: int = 0
        self.total_pollen_collected: float = 0.0
        self.distance_traveled: float = 0.0

    def move(self, dx: float, dy: float) -> None:
        """Move the bee by dx and dy."""
        self.x += dx
        self.y += dy

    def set_position(self, x: float, y: float) -> None:
        """Set the bee's position directly."""
        self.x = x
        self.y = y

    def collect_pollen(self) -> None:
        """Bee collects pollen."""
        self.has_pollen = True

    def deposit_pollen(self) -> None:
        """Bee deposits pollen at the hive."""
        self.has_pollen = False

    def update_orientation(self, theta: float) -> None:
        "Update the orientation"
        self.orientation = theta

    def update_velocity(self, velocity: float) -> None:
        "Update the velocity"
        self.velocity = velocity

    def detect_plants(self, plants, detection_range):
        """Find plants within detection range"""
        nearby = []
        for plant in plants:
            dist = np.sqrt((self.x - plant.x) ** 2 + (self.y - plant.y) ** 2)
            if dist <= detection_range:
                nearby.append((plant, dist))
        return nearby

    def levy_walk(self, dt):
        """Lévy flight movement pattern"""
        # Short steps frequently, long steps rarely
        step_length = np.random.pareto(1.5) * self.velocity * dt
        self.orientation += np.random.normal(0, 0.3)  # Add turning
        self.x += step_length * np.cos(self.orientation)
        self.y += step_length * np.sin(self.orientation)
