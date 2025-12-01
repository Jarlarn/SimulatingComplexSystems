class Plant:
    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def set_position(self, x: float, y: float) -> None:
        """Set the plant's position directly."""
        self.x = x
        self.y = y
