# Supermarket Shelf Manager

Three.js warehouse visual manager backed by `shelf_inventory.db`.

## Coordinate contract

The database uses a ROS `map` compatible, right-handed Z-up coordinate system:
`+X × +Y = +Z`. `world_x` and `world_y` identify each shelf's local origin,
not a screen-relative corner. At `yaw = 0`, a shelf extends along local `+X`
(width) and `+Y` (length); `yaw` rotates this local frame counter-clockwise
around `+Z` in radians.

## Run locally

Python is managed with `uv`:

```bash
uv sync
uv run python api_server.py
```

In another terminal, start the web application:

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite development server forwards `/api` calls
to the local API at port `8000`.

## Interaction model

- Click a shelf to edit its name, position, yaw, type, or remove it.
- Click a colored product block to edit its SKU and slot coordinates, or remove it.
- Use the plus button in the overview to add a shelf; use `Add product` from a
  selected shelf to create a slot.
- Product blocks are intentionally generic cuboids. Their centers come from the
  database's existing coordinate conversion, not from new visual-only data.
