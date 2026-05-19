import json
import os
import unittest

# Ensure the database is loaded in-memory before importing project module
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from project.project import (
    init_database,
    DB_ENGINE,
    create_transaction,
    get_inventory_items_by_name,
    get_stock_level, search_quote_history
)


class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Change the working directory to the project directory
        # so that read_csv can find "quotes.csv" and "quote_requests.csv" during init_database()
        project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project")
        os.chdir(project_dir)

    def setUp(self):
        # Initialize the database with fresh schema/data before each test
        init_database(DB_ENGINE)

    def test_create_transaction_success(self):
        # Test a transaction is created successfully
        transaction = create_transaction("Test item", "stock_orders", 10, 100.0, "2026-01-01")
        self.assertIsNotNone(transaction.id)
        self.assertEqual(transaction.item_name, "Test item")
        self.assertEqual(transaction.transaction_type, "stock_orders")
        self.assertEqual(transaction.units, 10)
        self.assertEqual(transaction.price, 100)
        self.assertEqual(transaction.transaction_date, "2026-01-01")

    def test_create_unknown_category_transaction_fails(self):
        # Test a transaction with an unknown category fails
        with self.assertRaises(ValueError):
            create_transaction("Test item", "random type", 10, 100.0, "2026-01-01")

    def test_get_inventory_items_single_item(self):
        result = get_inventory_items_by_name(["paper plates"])
        self.assertIsNotNone(result, "Inventory must not be empty")
        self.assertEqual(len(result), 1)
        item = result[0]
        self.assertEqual(item.item_name.lower(), "paper plates")
        self.assertEqual(item.current_stock, 748)

    def test_get_inventory_items_multiple_items(self):
        result = get_inventory_items_by_name(["paper plates", "GLOSSY paper"])
        self.assertIsNotNone(result, "Inventory must not be empty")
        self.assertEqual(len(result), 2)

        item1 = result[0]
        self.assertEqual(item1.item_name.lower(), "paper plates")
        self.assertEqual(item1.category, "product")
        self.assertEqual(item1.current_stock, 748)
        self.assertEqual(item1.min_stock_level, 144)

        item2 = result[1]
        self.assertEqual(item2.item_name.lower(), "glossy paper")
        self.assertEqual(item2.category, "paper")
        self.assertEqual(item2.current_stock, 587)
        self.assertEqual(item2.min_stock_level, 147)

    def test_get_inexistent_inventory_item_returns_none(self):
        self.assertIsNone(get_inventory_items_by_name(["non existent item"]))

    def test_get_stock_level(self):
        stock_level = get_stock_level("PAPER plaTes", "2026-01-01")
        self.assertIn("Paper plates", stock_level.items)
        self.assertEqual(stock_level.items["Paper plates"], 748)

    def test_get_stock_level_of_inexistent_item_returns_none(self):
        stock_level = get_stock_level("non existent item", "2026-01-01")
        self.assertIsNone(stock_level)

    def test_get_quote_history(self):
        # bulk_search_result = search_quote_history(["bulk"])
        # print(f"bulk search result:\n{json.dumps(bulk_search_result, indent=2)}\n")
        #
        # high_quality_result = search_quote_history(["high-quality"])
        # print(f"high-quality search result:\n{json.dumps(high_quality_result, indent=2)}\n")

        a4_paper_result = search_quote_history(["a4 paper"])
        print(f"a4 paper search result:\n{json.dumps(a4_paper_result, indent=2)}\n")


if __name__ == '__main__':
    unittest.main()
