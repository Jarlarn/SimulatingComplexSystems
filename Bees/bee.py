import numpy as np
from typing import List, Tuple, Optional
from plant import Plant


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
        self.pollen_capacity: float = 35.0  # mg

        # Memory of last plant visited
        self.last_plant_visited: Optional[object] = None

        # Statistics
        self.trips_completed: int = 0
        self.total_pollen_collected: float = 0.0
        self.distance_traveled: float = 0.0

    def move(self, dx: float, dy: float) -> None:
        """Move the bee by dx and dy."""
        dist = np.sqrt(dx**2 + dy**2)
        self.distance_traveled += dist
        self.x += dx
        self.y += dy

    def set_position(self, x: float, y: float) -> None:
        """Set the bee's position directly."""
        self.x = x
        self.y = y

    def collect_pollen(self, plant: Plant) -> bool:
        """Bee collects up to remaining capacity from a plant."""
        if not plant.has_pollen():
            return False

        remaining = max(self.pollen_capacity - self.pollen_amount, 0.0)
        if remaining <= 0:
            return False

        amount = plant.collect_pollen(remaining)
        if amount > 0:
            self.pollen_amount += amount
            self.has_pollen = self.pollen_amount > 0
            self.last_plant_visited = plant
            self.total_pollen_collected += amount
            return True
        return False

    def deposit_pollen(self) -> float:
        """Bee deposits pollen at the hive."""
        if self.has_pollen:
            amount = self.pollen_amount
            self.has_pollen = False
            self.pollen_amount = 0.0
            self.trips_completed += 1
            return amount
        return 0.0

    def detect_plants(self, plants):
        """Find plants within detection range"""
        nearby = []
        for plant in plants:
            # Don't return to the same plant we just visited
            if plant == self.last_plant_visited:
                continue

            dist = np.sqrt((self.x - plant.x) ** 2 + (self.y - plant.y) ** 2)
            if dist <= self.detection_range and plant.has_pollen():
                nearby.append((plant, dist))
                nearby.sort(key=lambda x: x[1])
        return nearby

    def distance_to_hive(self) -> float:
        """Calculate distance to hive."""
        return np.sqrt((self.x - self.hive_x) ** 2 + (self.y - self.hive_y) ** 2)

    def is_at_hive(self, threshold: float = 10.0) -> bool:
        """Check if bee is at the hive."""
        return self.distance_to_hive() < threshold

    def move_towards_target(self, target_x: float, target_y: float, dt: float) -> None:
        """Move directly towards a target position."""
        dx = target_x - self.x
        dy = target_y - self.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 0:
            # Update orientation to face target
            self.orientation = np.arctan2(dy, dx)

            # Move towards target
            step = min(self.velocity * dt, dist)
            self.x += (dx / dist) * step
            self.y += (dy / dist) * step
            self.distance_traveled += step

    def levy_walk(self, dt):
        """Lévy flight movement pattern"""
        # Short steps frequently, long steps rarely
        step_length = np.random.pareto(1.5) * self.velocity * dt
        self.orientation += np.random.normal(0, 0.3)  # Add turning
        self.x += step_length * np.cos(self.orientation)
        self.y += step_length * np.sin(self.orientation)

    def move_with_attraction(
        self, plants: List, dt: float, attraction_strength: float = 0.3
    ) -> None:
        """Move with Lévy walk but influenced by nearby plant attraction."""
        # Base Lévy movement
        step_length = np.random.pareto(1.5) * self.velocity * dt

        # Calculate attraction from nearby plants
        attraction_x = 0.0
        attraction_y = 0.0

        for plant in plants:
            if plant == self.last_plant_visited or not plant.has_pollen():
                continue

            dx = plant.x - self.x
            dy = plant.y - self.y
            dist = np.sqrt(dx**2 + dy**2)

            # Plants within their attraction radius influence the bee
            if dist < plant.attraction_radius and dist > 0:
                # Attraction falls off with distance
                attraction_force = (1 - dist / plant.attraction_radius) ** 2
                attraction_x += (dx / dist) * attraction_force
                attraction_y += (dy / dist) * attraction_force

        # Normalize attraction vector
        attraction_magnitude = np.sqrt(attraction_x**2 + attraction_y**2)
        if attraction_magnitude > 0:
            attraction_x /= attraction_magnitude
            attraction_y /= attraction_magnitude

            # Blend random walk with attraction
            target_orientation = np.arctan2(attraction_y, attraction_x)

            # Weighted average between current orientation and attraction
            self.orientation = (
                1 - attraction_strength
            ) * self.orientation + attraction_strength * target_orientation

            # Add some random noise
            self.orientation += np.random.normal(0, 0.2)
        else:
            # Pure random walk if no attraction
            self.orientation += np.random.normal(0, 0.3)

        # Move in the resulting direction
        dx = step_length * np.cos(self.orientation)
        dy = step_length * np.sin(self.orientation)

        self.x += dx
        self.y += dy
        self.distance_traveled += step_length

    def enforce_boundaries(
        self, min_x: float, max_x: float, min_y: float, max_y: float
    ) -> None:
        """Keep bee within boundaries (bounce off walls)."""
        if self.x < min_x:
            self.x = min_x
            self.orientation = np.pi - self.orientation
        elif self.x > max_x:
            self.x = max_x
            self.orientation = np.pi - self.orientation

        if self.y < min_y:
            self.y = min_y
            self.orientation = -self.orientation
        elif self.y > max_y:
            self.y = max_y
            self.orientation = -self.orientation

    def is_full(self) -> bool:
        """Return True if the bee has reached or exceeded capacity."""
        return self.pollen_amount >= self.pollen_capacity
