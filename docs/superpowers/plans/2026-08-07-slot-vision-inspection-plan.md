# Slot Vision Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local-photo shelf inspection workflow that detects changed fixed slots, identifies actual SKU with DINOv2, optionally uses the existing Ark VLM as fallback, and applies only user-selected results to SQLite and the JSON cache.

**Architecture:** vision/cv_restock_position.py remains the inspection engine. It aligns the uploaded image to the calibrated shelf face, maps difference regions to JSON v2 slot bounding boxes, ranks SKU reference crops, and writes an immutable per-run report. The API owns upload, configuration, report access, and explicit apply; the Web page reviews and selects report rows. SQLite remains authoritative; applying results updates SQLite first, then projects affected shelves into JSON.

**Tech Stack:** Existing Python/OpenCV/NumPy/DINOv2 dependencies, existing Ark helpers, config.local.yaml, Python http.server API, React/TypeScript, unittest.

---

### Task 1: Define inspection decisions and configuration

**Files:**
- Modify: vision/config.py
- Modify: vision/config.example.yaml
- Modify: tests/test_fixed_slots.py

- [ ] **Step 1: Write failing decision tests**

Add tests for this exact table:

    low DINO + fallback off       -> actual_sku null, reason low_confidence
    low DINO + Ark returns SKU    -> that SKU, source ark
    low DINO + Ark empty           -> actual_sku null
    low DINO + Ark error/unparsed  -> actual_sku null
    high DINO                     -> top SKU, source dino
    top1/top2 within margin        -> low-confidence path

Call a pure function named classify_candidate so these rules are testable without model or network calls.

- [ ] **Step 2: Run the focused test and verify failure**

Run: uv run python -m unittest tests.test_fixed_slots.FixedSlotTests.test_inspection_decision_without_vlm_turns_low_confidence_into_shortage -v

Expected: ImportError or NameError because the contract is not implemented.

- [ ] **Step 3: Implement the smallest contract**

Extend vision/config.py with defaults:

    inspection:
      dino_confidence_threshold: 0.72
      ambiguity_margin: 0.05
      vlm_fallback: false
      vlm_top_k: 4
      save_debug: true

The loader must preserve api_keys, validate a YAML mapping, and clear its cache after saving. Never return api_keys to the Web client.

Add classify_candidate(...) to vision/cv_restock_position.py. A score below threshold or a top1/top2 difference no greater than ambiguity_margin is low confidence. Without fallback it becomes shortage. With fallback, a valid Ark SKU is used; empty, not_product, timeout, API error, or unparseable output becomes shortage. High-confidence DINO uses top1.

- [ ] **Step 4: Run regression tests**

Run: uv run python -m unittest tests.test_fixed_slots -v

Expected: all existing tests and the new decision tests pass.

- [ ] **Step 5: Commit**

    git add vision/config.py vision/config.example.yaml vision/cv_restock_position.py tests/test_fixed_slots.py
    git commit -m "feat: define slot inspection decisions"

### Task 2: Upgrade the CV runner

**Files:**
- Modify: vision/cv_restock_position.py
- Reuse: vision/sku_query.py, vision/vlm_sku_query.py, vision/reference_photo_align.py, calibration_manager.py
- Modify: tests/test_fixed_slots.py

- [ ] **Step 1: Write a deterministic report-shape test**

Use temporary calibration JSON and synthetic images. Mock DINO and Ark calls. Assert run_inspection(...) returns run_id, shelf_id, face, and rows containing slot_id, expected_sku, actual_sku, status, source, confidence, reason, and selected. Assert the slot IDs are read from JSON, never regenerated.

- [ ] **Step 2: Run the test and verify failure**

Run: uv run python -m unittest tests.test_fixed_slots.FixedSlotTests.test_inspection_report_uses_stable_slot_ids -v

Expected: failure because run_inspection does not exist.

- [ ] **Step 3: Implement run_inspection**

Add:

    def run_inspection(current_path: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        """Run one inspection and write result.json plus optional debug artifacts."""

Reuse find_best_match, align_and_crop, and compute_difference_regions. The implementation must:

- load the current image and find the best calibrated shelf face;
- read faces[str(face)]["slots"] from JSON v2;
- transform every slot bbox into cropped current-image coordinates;
- mark only slots overlapped by a difference region as candidates;
- gather reference images from data/item_images/<slot_id>/0.png, grouped by SKU;
- include expected_sku as the expected label and all available known SKU reference crops as candidate actual SKUs;
- reuse the existing DINOv2 loader/scoring path from vision/sku_query.py;
- send only low-confidence or ambiguous crops to Ark when vlm_fallback is true;
- use actual_sku null for unresolved or empty VLM results;
- derive status from expected_sku and actual_sku exactly as the database does;
- keep unchanged rows in the report but mark them unchanged and unselected;
- write result.json, run_config.yaml, and optional aligned, diff, candidate-box, VLM composite, and final overlay artifacts below vision/output/slot_inspection/<run_id>/.

One crop or VLM failure must become a row-level reason rather than aborting the run. Only invalid input, no calibrated face, or failed alignment quality gates may fail the whole run.

- [ ] **Step 4: Verify decisions with mocked dependencies**

Run: uv run python -m unittest tests.test_fixed_slots -v

Expected: high DINO, low-without-fallback, low-with-Ark-SKU, and low-with-Ark-error cases all pass.

- [ ] **Step 5: Preserve CLI behavior**

Retain --current and existing alignment flags. Add --dino-confidence-threshold, --ambiguity-margin, --vlm-fallback, --save-debug, and --no-save-debug. YAML values are defaults and CLI flags take precedence.

Run: uv run python -m vision.cv_restock_position --help

Expected: the new options appear and the module imports successfully.

- [ ] **Step 6: Commit**

    git add vision/cv_restock_position.py tests/test_fixed_slots.py README.md
    git commit -m "feat: inspect fixed slots with dino and ark fallback"

### Task 3: Add API config, run, artifact, and apply endpoints

**Files:**
- Modify: api_server.py
- Modify: shelf_database.py
- Modify: tests/test_fixed_slots.py

- [ ] **Step 1: Write HTTP contract tests**

Extend the existing temporary HTTP server tests with:

    GET  /api/vision/config
    PUT  /api/vision/config
    POST /api/vision/inspect
    GET  /api/vision/runs/{run_id}/result
    GET  /api/vision/runs/{run_id}/artifact/{name}
    POST /api/vision/runs/{run_id}/apply

Use JSON image data for inspect and JSON {slot_ids: ["1-0-2-43"]} for apply. Assert inspect leaves the database unchanged; applying one selected row changes only that slot and updates its JSON projection.

- [ ] **Step 2: Run the API test and verify failure**

Run: uv run python -m unittest tests.test_fixed_slots -v

Expected: 404 responses for the new endpoints.

- [ ] **Step 3: Implement config and inspection endpoints**

GET/PUT /api/vision/config must use vision.config, validate threshold in [0, 1], margin >= 0, top_k in [1, 9], and boolean flags. Do not expose API keys.

POST /api/vision/inspect must validate the data URL, write a temporary input file, merge request overrides over saved config, invoke run_inspection, and return the report metadata and run ID. Store the exact effective config in the run directory.

The artifact endpoint must resolve the run directory and reject path traversal and unknown files. Return image MIME types for debug images and JSON for report files.

- [ ] **Step 4: Implement atomic selected application**

Add a transaction method to shelf_database.py:

    def set_actual_sku_batch(self, changes: list[tuple[str, str | None]]) -> list[ShelfSlot]:
        """Update selected fixed slots atomically and return refreshed rows."""

The apply endpoint must load server-side result.json, accept only slot IDs in that report, reject missing/deleted slots, update only requested IDs, and record applied_slot_ids. After SQLite commit, call the existing JSON projection under JSON_SYNC_LOCK for each affected shelf. A JSON failure must not roll back SQLite; return a clear synchronization error.

- [ ] **Step 5: Run the API and regression tests**

Run: uv run python -m unittest tests.test_fixed_slots -v

Expected: all tests pass, including read-only inspection, selected-only application, and JSON synchronization.

- [ ] **Step 6: Commit**

    git add api_server.py shelf_database.py tests/test_fixed_slots.py
    git commit -m "feat: review and apply vision slot results"

### Task 4: Add the Web inspection page

**Files:**
- Create: web/src/VisionInspectionPage.tsx
- Modify: web/src/App.tsx
- Modify: web/src/types.ts
- Modify: web/src/styles.css

- [ ] **Step 1: Define TypeScript response types**

Add VisionConfig, VisionInspectionReport, and VisionInspectionRow. A row contains slot_id, expected_sku, actual_sku, status, source, confidence, reason, selected, and optional artifact names.

- [ ] **Step 2: Implement the page controls**

VisionInspectionPage must provide:

- native image upload;
- DINO confidence and ambiguity sliders with numeric values;
- VLM fallback checkbox;
- debug-artifact checkbox;
- 保存配置, 运行识别, 全选, 取消全选, 应用修改, 返回;
- result table with stable slot ID, expected SKU, detected SKU, status, confidence, source, and reason;
- per-row checkbox disabled for unchanged rows;
- debug images only when present in the report.

运行识别 replaces page-local report state and never refreshes or modifies warehouse data. 应用修改 sends only checked slot IDs, shows the confirmed count, then calls the existing /api/state refresh.

- [ ] **Step 3: Add navigation beside manual import**

Add 巡检识别 beside the existing manual-entry button in App.tsx. Reuse the current view-toggle pattern; do not add a router or UI dependency.

- [ ] **Step 4: Style and build**

Add responsive toolbar/table styles, status colors, selected-row styling, horizontal scrolling on narrow screens, and accessible labels/titles for icon-only controls.

Run: npm --prefix web run build

Expected: TypeScript and Vite build pass.

- [ ] **Step 5: Commit**

    git add web/src/App.tsx web/src/VisionInspectionPage.tsx web/src/types.ts web/src/styles.css
    git commit -m "feat: add vision inspection review page"

### Task 5: Document and verify the complete workflow

**Files:**
- Modify: README.md
- Modify: docs/database_schema_and_api.md
- Modify: tests/test_fixed_slots.py only if final regression coverage is missing

- [ ] **Step 1: Document the safety boundary**

Document that recognition is read-only until 应用修改, low confidence follows the VLM fallback setting, VLM failure becomes shortage, and applying a SKU/null actual value automatically derives status and synchronizes JSON.

- [ ] **Step 2: Document config, artifacts, and API payloads**

Document vision/config.local.yaml, Ark-only first release, vision/output/slot_inspection/<run_id>/, CLI usage, endpoint payloads, and the fact that API keys never reach the Web client.

- [ ] **Step 3: Run final verification**

    uv run python -m unittest tests.test_fixed_slots -v
    uv run python -m py_compile api_server.py shelf_database.py calibration_manager.py vision/*.py
    npm --prefix web run build
    git diff --check

Expected: tests pass, Python compilation succeeds, Web build succeeds, and git diff --check has no output.

- [ ] **Step 4: Commit**

    git add README.md docs/database_schema_and_api.md tests/test_fixed_slots.py
    git commit -m "docs: document vision inspection workflow"

## Self-review

- Coverage: DINO classification, Ark fallback, unresolved-to-shortage behavior, config persistence, read-only runs, selected application, JSON synchronization, Web review controls, artifacts, tests, and documentation are covered.
- Data boundary: no new business table or status column is added; actual_sku remains the only mutable observation and status remains derived.
- Stability: every result and API operation uses immutable slot_id; shelf/face geometry comes from JSON v2.
- Simplification: no background queue or router is added for the first release. A synchronous local run is sufficient until profiling proves otherwise.
