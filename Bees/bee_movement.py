class Bee:
    def __init__(self, x: float, y: float, orientation: float, velocity: float) -> None:
        self.x: float = x
        self.y: float = y
        self.orientaion: float = orientation
        self.velocity: float = velocity
        self.has_pollen: bool = False

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
        self.orientaion = theta

    def update_velocity(self, velocity: float) -> None:
        self.velocity = velocity
