"""CelebA class indices used by the three-condition CCDM extension."""

STYLE = {"Straight_Hair": 0, "Wavy_Hair": 1}
COLOR = {"Brown_Hair": 0, "Blond_Hair": 1, "Gray_Hair": 2, "Black_Hair": 3}
SEX = {"Male": 0, "Female": 1}


def parse_condition(name):
    """Parse a directory name in thesis order: hairstyle color sex."""
    parts = name.split(" ")
    if len(parts) != 3:
        raise ValueError(f"Expected 'Style Color Sex', got {name!r}")
    return STYLE[parts[0]], COLOR[parts[1]], SEX[parts[2]]
