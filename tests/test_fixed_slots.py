import unittest

from shelf_database import (
    DEFAULT_BACK_THICK,
    DEFAULT_BOTTOM_CLEARANCE,
    DEFAULT_LEVEL_SPACING,
    DEFAULT_PANEL_THICK,
    DEFAULT_SHELF_DEPTH_BOTTOM,
    DEFAULT_SHELF_DEPTH_NORMAL,
    DEFAULT_SHELF_HEIGHT,
    DEFAULT_SHELF_LENGTH,
    DEFAULT_SHELF_WIDTH,
    ShelfDatabase,
)


class FixedSlotTests(unittest.TestCase):
    def setUp(self):
        self.db = ShelfDatabase(":memory:")
        type_id = self.db.add_shelf_type(
            "test",
            DEFAULT_SHELF_LENGTH,
            DEFAULT_SHELF_WIDTH,
            DEFAULT_SHELF_HEIGHT,
            5,
            DEFAULT_BOTTOM_CLEARANCE,
            DEFAULT_LEVEL_SPACING,
            DEFAULT_PANEL_THICK,
            DEFAULT_BACK_THICK,
            DEFAULT_SHELF_DEPTH_NORMAL,
            DEFAULT_SHELF_DEPTH_BOTTOM,
        )
        self.shelf_id = self.db.add_shelf_group("Shelf", shelf_type_id=type_id)
        self.db.register_skus_batch(
            [{"sku": "cola"}, {"sku": "sprite"}, {"sku": "tea"}]
        )

    def tearDown(self):
        self.db.close()

    def test_slot_id_is_stable_and_status_is_derived(self):
        slot_id = self.db.create_slot(
            self.shelf_id, 0, 2, 43, "sprite", "sprite"
        )
        slot = self.db.get_slot_by_id(slot_id)
        self.assertEqual(slot.slot_id, "1-0-2-43")
        self.assertEqual(slot.status, "正常")

        self.db.set_actual_sku(slot.slot_id, None)
        self.assertEqual(self.db.get_slot_by_id(slot.slot_id).status, "缺货")

        self.db.set_actual_sku(slot.slot_id, "cola")
        self.assertEqual(self.db.get_slot_by_id(slot.slot_id).status, "摆放错误")

    def test_take_and_restock_keep_slot_id(self):
        slot_id = self.db.create_slot(
            self.shelf_id, 0, 2, 50, "sprite", "sprite"
        )
        self.db.take_slot(slot_id)
        self.assertIsNone(self.db.get_slot_by_id(slot_id).actual_sku)

        self.db.restock_slot(slot_id)
        slot = self.db.get_slot_by_id(slot_id)
        self.assertEqual(slot.actual_sku, "sprite")
        self.assertEqual(slot.slot_id, "1-0-2-50")

    def test_slot_location_is_not_updated(self):
        slot_id = self.db.create_slot(self.shelf_id, 0, 2, 65, "tea", "tea")
        with self.assertRaises(ValueError):
            self.db.update_slot(slot_id, shelf_id=self.shelf_id, face=1)

    def test_missing_slots_remain_queryable_without_actual_product(self):
        slot_id = self.db.create_slot(self.shelf_id, 0, 2, 80, "tea", None)
        slot = self.db.get_slot_by_id(slot_id)
        world_rows = self.db.get_all_slots_world()

        self.assertEqual(slot.status, "缺货")
        self.assertEqual(world_rows[0]["slot_id"], slot_id)
        self.assertIsNone(world_rows[0]["actual_sku"])


if __name__ == "__main__":
    unittest.main()
