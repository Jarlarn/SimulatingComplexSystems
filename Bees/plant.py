class Plant:
    def __init__(
        self,
        x: float,
        y: float,
        max_pollen: int = 3,
        attraction_radius: float = 50.0,
    ) -> None:
        self.x: float = x
        self.y: float = y
        self.max_pollen: int = max_pollen
        self.current_pollen: float = max_pollen
        self.attraction_radius: float = attraction_radius
        self.times_visited: int = 0
        self.total_pollen_collected: float = 0.0

    def set_position(self, x: float, y: float) -> None:
        """Set the plant's position directly."""
        self.x = x
        self.y = y

    def has_pollen(self) -> bool:
        """Check if plant has pollen available."""
        return self.current_pollen > 0

    def collect_pollen(self, amount: float = 0.70) -> float:
        """Collect pollen from the plant. Returns amount actually collected."""
        collected = min(amount, self.current_pollen)
        self.current_pollen -= collected
        self.times_visited += 1
        self.total_pollen_collected += collected
        return collected

    def get_pollen_ratio(self) -> float:
        """Get current pollen as ratio of max pollen."""
        return self.current_pollen / self.max_pollen
