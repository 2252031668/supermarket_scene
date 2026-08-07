# Fixed Shelf Slot Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing inventory rows into stable fixed shelf positions with `expected_sku`/`actual_sku` state, synchronized JSON v2, explicit status APIs, and Web support for normal, shortage, and misplaced positions.

**Architecture:** SQLite remains the only writable business source. `shelf_inventory.slot_id` is a server-generated immutable text primary key; status is derived from the two SKU fields. Per-shelf calibration JSON is rebuilt as a read cache after successful database writes, while its image/layer/bbox calibration data is preserved. CV upload and recognition remain out of scope.

**Tech Stack:** Python `sqlite3` and `json`, existing `ThreadingHTTPServer` API, React + TypeScript + Vite, existing React Three Fiber scene, standard-library `unittest` checks.

---

## Files And Responsibilities

- Modify `shelf_database.py`: new fixed-slot schema, slot dataclass, stable ID generation, derived status, state queries, actual/expected SKU queries, world-position lookup, and action methods.
- Modify `api_server.py`: new slot payloads, stable-ID routes, shortage/misplacement queries, JSON sync calls, and snapshot fields.
- Modify `calibration_manager.py`: JSON v2 `slots` projection and atomic writes that preserve calibration fields.
- Modify `mujoco/generate_scene_from_database.py`: instantiate only non-null `actual_sku` rows.
- Modify `web/src/types.ts`: expected/actual/status and fixed slot ID types.
- Modify `web/src/App.tsx`: stable-ID editor, state filters, shortage/misplacement lists, and slot actions.
- Modify `web/src/ManualImportPage.tsx`: empty-position mode and separate expected/actual review fields.
- Modify `web/src/WarehouseScene.tsx`: render clickable empty-position markers and status styling.
- Modify `web/src/styles.css`: status badges, filters, empty markers, and review-table layout.
- Modify `init_database.py`: initialize the new schema and seed normal slots with expected and actual SKU equal.
- Modify `docs/database_schema_and_api.md` and `README.md`: document the new model and current scope.
- Create temporarily `scripts/migrate_fixed_slots_once.py`: one-time database/JSON migration; run, verify, then delete from the repository.
- Create `tests/test_fixed_slots.py`: focused database, status, action, migration/projection, and world-position checks.

The worktree already contains user changes. Do not reset, checkout, or overwrite unrelated changes; inspect overlapping hunks before editing.

### Task 1: Add Failing Contract Checks

**Files:**
- Create: `tests/test_fixed_slots.py`

- [ ] **Step 1: Write the failing test file**

Use an in-memory `ShelfDatabase`, create one shelf type, one shelf, and three SKUs. Cover the contract before implementation:

```python
import unittest

from shelf_database import DEFAULT_SHELF_HEIGHT, DEFAULT_SHELF_LENGTH, DEFAULT_SHELF_WIDTH
from shelf_database import DEFAULT_BOTTOM_CLEARANCE, DEFAULT_LEVEL_SPACING
from shelf_database import DEFAULT_PANEL_THICK, DEFAULT_BACK_THICK
from shelf_database import DEFAULT_SHELF_DEPTH_NORMAL, DEFAULT_SHELF_DEPTH_BOTTOM
from shelf_database import ShelfDatabase


class FixedSlotTests(unittest.TestCase):
    def setUp(self):
        self.db = ShelfDatabase(":memory:")
        type_id = self.db.add_shelf_type(
            "test", DEFAULT_SHELF_LENGTH, DEFAULT_SHELF_WIDTH, DEFAULT_SHELF_HEIGHT,
            5, DEFAULT_BOTTOM_CLEARANCE, DEFAULT_LEVEL_SPACING, DEFAULT_PANEL_THICK,
            DEFAULT_BACK_THICK, DEFAULT_SHELF_DEPTH_NORMAL, DEFAULT_SHELF_DEPTH_BOTTOM,
        )
        self.shelf_id = self.db.add_shelf_group("Shelf", shelf_type_id=type_id)
        self.db.register_skus_batch([{"sku": "cola"}, {"sku": "sprite"}, {"sku": "tea"}])

    def tearDown(self):
        self.db.close()

    def test_slot_id_is_stable_and_status_is_derived(self):
        self.db.create_slot(self.shelf_id, 0, 2, 43, "sprite", "sprite")
        slot = self.db.get_slot_by_id("1-0-2-43")
        self.assertEqual(slot.slot_id, "1-0-2-43")
        self.assertEqual(slot.status, "正常")

        self.db.set_actual_sku(slot.slot_id, None)
        self.assertEqual(self.db.get_slot_by_id(slot.slot_id).status, "缺货")

        self.db.set_actual_sku(slot.slot_id, "cola")
        self.assertEqual(self.db.get_slot_by_id(slot.slot_id).status, "摆放错误")

    def test_take_and_restock_keep_slot_id(self):
        self.db.create_slot(self.shelf_id, 0, 2, 50, "sprite", "sprite")
        self.db.take_slot("1-0-2-50")
        self.assertIsNone(self.db.get_slot_by_id("1-0-2-50").actual_sku)
        self.db.restock_slot("1-0-2-50")
        slot = self.db.get_slot_by_id("1-0-2-50")
        self.assertEqual(slot.actual_sku, "sprite")
        self.assertEqual(slot.slot_id, "1-0-2-50")

    def test_slot_location_is_not_updated(self):
        self.db.create_slot(self.shelf_id, 0, 2, 65, "tea", "tea")
        with self.assertRaises(ValueError):
            self.db.update_slot("1-0-2-65", shelf_id=self.shelf_id, face=1)

    def test_missing_slots_are_not_scene_products(self):
        self.db.create_slot(self.shelf_id, 0, 2, 80, "tea", None)
        world_rows = self.db.get_all_slots_world()
        self.assertEqual(world_rows[0]["actual_sku"], None)


if __name__ == "__main__":
    unittest.main()
```

Adjust only the test fixture's shelf ID assumption if SQLite starts IDs at a different value; the production contract remains the generated `slot_id` string.

- [ ] **Step 2: Run the contract checks and confirm they fail**

Run:

```bash
python -m unittest tests/test_fixed_slots.py -v
```

Expected: failure because `create_slot`, `get_slot_by_id`, `set_actual_sku`, `take_slot`, `restock_slot`, and the new fields do not exist yet.

- [ ] **Step 3: Commit the failing checks**

```bash
git add tests/test_fixed_slots.py
git commit -m "test: define fixed shelf slot contracts"
```

### Task 2: Implement The Fixed-Slot Database Model

**Files:**
- Modify: `shelf_database.py`
- Modify: `init_database.py`
- Test: `tests/test_fixed_slots.py`

- [ ] **Step 1: Replace the inventory dataclass fields**

Change `ShelfSlot` to expose `slot_id`, `shelf_id`, `face`, `level`, `y_cm`, `expected_sku`, `actual_sku`, dimensions, image directory, and derived `status`. Keep a small shared status helper so all queries use the same rule:

```python
def slot_status(expected_sku: str, actual_sku: str | None) -> str:
    if actual_sku is None:
        return "缺货"
    return "正常" if actual_sku == expected_sku else "摆放错误"
```

- [ ] **Step 2: Replace schema creation with text `slot_id` primary key**

Use the schema in the design document. Do not leave a production migration branch for the old `sku`/integer-ID schema. Add indexes on `expected_sku` and `actual_sku`; keep the existing position uniqueness constraint.

- [ ] **Step 3: Add canonical ID and row conversion helpers**

Keep `format_slot_id(shelf_id, face, level, y_cm)` as the only formatter and use it at creation and import time. Add a row-to-dataclass helper that computes `status` but never stores it. Add `get_slot_by_id(slot_id)` and make `slot_id_str_to_tuple` validate the canonical format.

- [ ] **Step 4: Implement fixed-slot write methods**

Define a module-level `_UNSET = object()` sentinel so `actual_sku: null` can be distinguished from an omitted update field.

Implement the minimum methods used by API and tests:

```python
create_slot(shelf_id, face, level, y_cm, expected_sku, actual_sku=None, ...)
update_slot(slot_id, *, expected_sku=None, actual_sku=_UNSET, width_cm=_UNSET, height_cm=_UNSET, image_dir=_UNSET)
set_actual_sku(slot_id, actual_sku)
take_slot(slot_id)
restock_slot(slot_id)
delete_slot(slot_id)
```

`create_slot` generates the ID and rejects caller-provided IDs, invalid SKU references, out-of-bounds positions, and duplicate coordinates. `update_slot` rejects all location fields. `take_slot` and `restock_slot` are idempotent.

- [ ] **Step 5: Update all database reads**

Update `get_slot`, `get_shelf_inventory`, `get_all_slots_world`, `get_shelf_group_all_slots_world`, SKU summaries, and SKU location queries to select the new fields. Add `get_shortage_slots()` and `get_misplaced_slots()`. Actual SKU summaries and actual-location queries ignore null `actual_sku`; expected-location queries use `expected_sku`.

- [ ] **Step 6: Update scene consumers**

Change `get_all_slots_world()` to return both SKU fields and status. Update `mujoco/generate_scene_from_database.py` in Task 5 to skip rows whose `actual_sku` is null.

- [ ] **Step 7: Update initialization code**

Change `init_database.py` and the module demo calls from `set_slot(..., sku=...)` to `create_slot(..., expected_sku=..., actual_sku=...)`. Normal seeded products set both values equal.

- [ ] **Step 8: Run the database checks**

Run:

```bash
python -m unittest tests/test_fixed_slots.py -v
```

Expected: all database contract tests pass. Do not run the migration against the real database until Task 3 has a verified backup and dry-run path.

- [ ] **Step 9: Commit the database model**

```bash
git add shelf_database.py init_database.py tests/test_fixed_slots.py
git commit -m "feat: model fixed shelf slots and derived status"
```

### Task 3: Migrate The Existing Database And JSON Once

**Files:**
- Create temporarily: `scripts/migrate_fixed_slots_once.py`
- Modify during migration: `shelf_inventory.db`, `data/shelf_calibration/*.json`
- Delete after successful verification: `scripts/migrate_fixed_slots_once.py`
- Test: `tests/test_fixed_slots.py`

- [ ] **Step 1: Write a dry-run capable migration script**

The script must:

```python
copy2(db_path, backup_db_path)
copytree(calibration_dir, backup_calibration_dir)
read old shelf_inventory rows
generate slot_id with ShelfDatabase.format_slot_id
reject duplicate slot IDs or coordinates
create a temporary new table with text slot_id primary key
copy old sku into expected_sku and actual_sku
read each shelf JSON
reject duplicate or orphaned products
convert products to slots while preserving bbox/layers/image metadata
write JSON through temporary files
```

Use an explicit `--dry-run` mode that performs validation and reports counts without replacing files, and a normal mode that replaces the database only after all JSON conversions validate.

- [ ] **Step 2: Run the dry run against the current data**

Run:

```bash
python scripts/migrate_fixed_slots_once.py --dry-run --db shelf_inventory.db --calibration-dir data/shelf_calibration
```

Expected: a report containing old row count, new slot count, converted JSON face count, orphan count `0`, duplicate count `0`, and no file changes.

- [ ] **Step 3: Run the migration with backups**

Run:

```bash
python scripts/migrate_fixed_slots_once.py --db shelf_inventory.db --calibration-dir data/shelf_calibration
```

Expected: backup paths printed, SQLite table replaced, all JSON files have `schema_version: 2` and `faces[*].slots`, and no JSON has a `products` key.

- [ ] **Step 4: Validate the migrated artifacts**

Run:

```bash
python -m unittest tests/test_fixed_slots.py -v
python - <<'PY'
import json
import pathlib
import sqlite3

db = sqlite3.connect("shelf_inventory.db")
columns = {row[1] for row in db.execute("PRAGMA table_info(shelf_inventory)")}
assert "slot_id" in columns and "expected_sku" in columns and "actual_sku" in columns
assert "sku" not in columns and "id" not in columns
for path in pathlib.Path("data/shelf_calibration").glob("*.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert all("products" not in face for face in data["faces"].values())
print("migration validation passed")
PY
```

- [ ] **Step 5: Remove the one-time script and compatibility code**

After validation, delete `scripts/migrate_fixed_slots_once.py` with `apply_patch`. Remove any old-schema production branches and old `products` conversion code from runtime modules. Keep the generated backups outside the repository or in the user-selected backup location.

- [ ] **Step 6: Commit the migration result**

```bash
git add shelf_inventory.db data/shelf_calibration shelf_database.py
git commit -m "chore: migrate inventory data to fixed slots"
```

### Task 4: Implement JSON v2 Projection And API Synchronization

**Files:**
- Modify: `calibration_manager.py`
- Modify: `api_server.py`
- Modify: `tests/test_fixed_slots.py`

- [ ] **Step 1: Add JSON projection helpers and a focused test**

Add a function with one responsibility:

```python
sync_shelf_slots(shelf_id: int, slots: list[dict]) -> None
```

It loads the existing shelf JSON, groups `slots` under their face, preserves `image_file`, `image_hash`, `layers`, and each existing bbox by `slot_id`, replaces only `faces[*].slots`, sets `schema_version` to `2`, writes a sibling temporary file, and calls `os.replace`.

The test must create a temporary calibration JSON with layers and bbox, sync new expected/actual values, reload it, and assert that the layers and bbox survive. A second test must make the target directory unwritable or inject a write failure at the helper boundary and assert that the original JSON remains valid.

- [ ] **Step 2: Add one API-level sync helper**

In `api_server.py`, centralize the sequence `database mutation -> read affected shelf slots -> JSON projection`. Do not call the old per-product JSON updater from multiple routes. If JSON write fails after SQLite commits, return an explicit server error and leave SQLite as the source of truth.

- [ ] **Step 3: Replace the slot creation route**

Make `POST /api/slots` accept coordinates and expected/actual fields, generate `slot_id` server-side, optionally accept bbox data for the calibration projection, return the created `slot_id`, and synchronize the affected shelf JSON.

- [ ] **Step 4: Replace the slot edit/delete routes**

Make `PUT /api/slots/{slot_id}` reject location fields and update only expected/actual/dimensions/bbox. Make `DELETE /api/slots/{slot_id}` delete the fixed position and its JSON slot. Remove old routes that identify a slot by a coordinate payload.

- [ ] **Step 5: Add action routes and list routes**

Implement:

```text
POST /api/slots/{slot_id}/take
POST /api/slots/{slot_id}/restock
GET  /api/shortages
GET  /api/misplacements
```

All action responses return the updated slot and refreshed state. List routes return `slot_id`, location fields, expected/actual SKU, and derived status.

- [ ] **Step 6: Update the state snapshot**

Return fixed-position rows with `slot_id`, expected/actual SKU, status, and world coordinates needed by the existing 3D scene. Add `total_positions`, `actual_items`, `shortages`, and `misplacements`; stop treating every database row as an actual item.

- [ ] **Step 7: Run API and projection checks**

Start the server on an unused local port with a temporary database, then use standard-library `urllib.request` or `curl` to verify:

```text
POST normal slot -> stable slot_id and status 正常
POST empty slot -> status 缺货
PUT actual SKU -> status 摆放错误
POST take -> actual_sku null and status 缺货
POST restock -> actual_sku expected_sku and status 正常
GET shortages/misplacements -> correct slot IDs
GET world-position -> same stable slot ID
```

- [ ] **Step 8: Commit the API and JSON layer**

```bash
git add api_server.py calibration_manager.py tests/test_fixed_slots.py
git commit -m "feat: synchronize fixed slot state to JSON and API"
```

### Task 5: Update Scene Generation And Shared Documentation

**Files:**
- Modify: `mujoco/generate_scene_from_database.py`
- Modify: `docs/database_schema_and_api.md`
- Modify: `README.md`
- Test: `tests/test_fixed_slots.py`

- [ ] **Step 1: Make MuJoCo use actual SKU only**

In `place_products_from_database`, skip rows where `actual_sku is None` and use `actual_sku` for mesh lookup. Keep the fixed slot world position available for callers even when no product is generated.

- [ ] **Step 2: Update database/API documentation**

Replace the old “one slot = one product instance” wording, field tables, write methods, query methods, response examples, and statistics with the fixed-slot model. Document that `slot_id` is a persisted text primary key and position fields are immutable after creation.

- [ ] **Step 3: Update README scope**

Document SQLite as the source of truth, JSON v2 as the CV read cache, current Web manual state management, and the fact that CV upload/recognition is not yet part of Web.

- [ ] **Step 4: Run scene and documentation checks**

Run:

```bash
python -m unittest tests/test_fixed_slots.py -v
uv run python -m mujoco.generate_scene_from_database
```

Expected: no product body for missing slots, no import errors, and generated XML contains only non-null actual products.

- [ ] **Step 5: Commit scene/documentation changes**

```bash
git add mujoco/generate_scene_from_database.py docs/database_schema_and_api.md README.md tests/test_fixed_slots.py
git commit -m "docs: describe fixed slot state and scene behavior"
```

### Task 6: Update Web Types And State Editor

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`

- [ ] **Step 1: Update TypeScript contracts**

Change `Slot` and `SlotDraft` to use:

```ts
type SlotStatus = '正常' | '缺货' | '摆放错误'

type Slot = {
  slot_id: string
  shelf_id: number
  face: number
  level: number
  y_cm: number
  expected_sku: string
  actual_sku: string | null
  status: SlotStatus
  width_cm: number | null
  height_cm: number | null
  image_dir: string
  world_x: number
  world_y: number
  world_z: number
  yaw: number
}
```

Remove `sku` from the frontend state contract.

- [ ] **Step 2: Replace the slot editor fields**

Show stable `slot_id` and location as read-only. Add separate expected-SKU and actual-SKU controls, with an explicit empty option for `actual_sku`. Render a read-only status badge and wire save to `PUT /api/slots/{slot_id}`.

- [ ] **Step 3: Add action buttons**

Use icon buttons with tooltips for take, restock, delete, and world-position detail. Wire them to the stable-ID action routes and refresh state from the response.

- [ ] **Step 4: Add status filters and shortage/misplacement lists**

Add all/shortage/misplacement filters. Render `slot_id`, expected SKU, actual SKU, readable shelf location, and the applicable action. Clicking a row selects the same slot used by the 3D scene.

- [ ] **Step 5: Update statistics and notices**

Use the new state counts and show JSON synchronization errors returned by the API. Do not add a normal-use manual “sync JSON” control.

- [ ] **Step 6: Run the TypeScript compiler**

Run:

```bash
npm --prefix web run build
```

Expected: TypeScript and Vite build succeed with no references to `slot.sku` or coordinate-based slot mutations in the editor.

- [ ] **Step 7: Commit the state editor**

```bash
git add web/src/types.ts web/src/App.tsx web/src/styles.css
git commit -m "feat: manage fixed slot status in Web"
```

### Task 7: Update Manual Import And 3D Empty-Position Rendering

**Files:**
- Modify: `web/src/ManualImportPage.tsx`
- Modify: `web/src/WarehouseScene.tsx`
- Modify: `web/src/styles.css`
- Test: `tests/test_fixed_slots.py`

- [ ] **Step 1: Add the empty-position capture mode**

Keep the existing box geometry and layer/Y calculation. Add a boolean `empty` (or equivalent local field) to each imported item. When enabled, require an expected SKU and send `actual_sku: null`; otherwise default actual to expected.

- [ ] **Step 2: Change review rows to expected/actual controls**

Replace the single SKU select with expected SKU and actual SKU selects. Include an empty option only in actual SKU. Derive the visible status from these values. Keep duplicate `slot_id`, invalid layer, invalid dimensions, missing expected SKU, and missing crop validation before enabling import.

- [ ] **Step 3: Update the manual import payload**

Send each item with coordinates, expected/actual SKU, bbox, dimensions, and crop data. Do not send a client-generated `slot_id`; display the canonical ID returned by the server after import.

- [ ] **Step 4: Render fixed empty slots**

In `WarehouseScene.tsx`, render actual product meshes only for non-null `actual_sku`. Render an empty clickable marker using the slot geometry for null actual SKU rows. Add a status marker/outline for misplaced rows while retaining the actual SKU mesh.

- [ ] **Step 5: Verify manual import behavior**

Run the Web build, then use the existing local API and browser workflow to verify:

```text
normal box -> expected = actual
empty box -> expected set, actual null
wrong product -> expected and actual differ
duplicate position -> import disabled/error
empty slot -> visible and selectable in 3D
```

- [ ] **Step 6: Commit the import and scene UI**

```bash
git add web/src/ManualImportPage.tsx web/src/WarehouseScene.tsx web/src/styles.css
git commit -m "feat: import empty slots and show shelf status"
```

### Task 8: End-to-End Verification And Handoff

**Files:**
- Modify only if verification finds a defect: files from Tasks 2-7

- [ ] **Step 1: Run the focused Python tests**

```bash
python -m unittest tests/test_fixed_slots.py -v
```

Expected: all fixed-slot, status, action, projection, migration, and world-position checks pass.

- [ ] **Step 2: Run the production smoke checks**

```bash
uv run python -m mujoco.generate_scene_from_database
npm --prefix web run build
```

Expected: scene generation and Web build both succeed.

- [ ] **Step 3: Start the local API and Web server**

```bash
uv run python api_server.py
npm --prefix web run dev -- --host 127.0.0.1
```

Use the existing local browser workflow to check a desktop and mobile-width viewport. Verify that JSON updates after each state mutation and that the UI never exposes coordinate editing for an existing slot.

- [ ] **Step 4: Check repository state**

```bash
git status --short
git diff --check
```

Confirm the one-time migration script is absent, no old `products` runtime schema remains, and unrelated user changes are still present.

- [ ] **Step 5: Commit verification fixes only when needed**

```bash
git diff --name-only
git add shelf_database.py api_server.py calibration_manager.py \
  mujoco/generate_scene_from_database.py web/src/types.ts web/src/App.tsx \
  web/src/ManualImportPage.tsx web/src/WarehouseScene.tsx web/src/styles.css \
  tests/test_fixed_slots.py docs/database_schema_and_api.md README.md
git commit -m "test: verify fixed shelf slot workflow"
```

Only stage files that contain fixes from this implementation; if verification is clean, make no empty commit.

Use the actual changed file list; do not stage the whole repository.
