"""Shared fixed geometry for non-inventory scene objects, in metres."""

# A delivery table does not store inventory.  Its dimensions are global scene
# geometry, while each database row stores only its named map-frame pose.
DELIVERY_TABLE_LENGTH = 1.20  # local +X
DELIVERY_TABLE_WIDTH = 0.80   # local +Y
DELIVERY_TABLE_HEIGHT = 0.75  # floor to tabletop centre
DELIVERY_TABLE_TOP_THICKNESS = 0.03


def delivery_table_spec() -> dict[str, float]:
    """Return the JSON-safe delivery-table dimensions consumed by the web UI."""
    return {
        "length": DELIVERY_TABLE_LENGTH,
        "width": DELIVERY_TABLE_WIDTH,
        "height": DELIVERY_TABLE_HEIGHT,
        "top_thickness": DELIVERY_TABLE_TOP_THICKNESS,
    }
