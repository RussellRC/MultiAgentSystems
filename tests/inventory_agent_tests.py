import json
import os
import unittest
from dataclasses import dataclass
from typing import List

from tests.test_utils import eval_report_cases

# Use a file-based SQLite DB with check_same_thread=False so the agent thread can share the same database.
_test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_inv_agent.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}?check_same_thread=False"

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from project.project import (
    init_database,
    DB_ENGINE,
    new_inventory_agent,
    InventoryAgentOutput,
    FinancialTransaction,
    get_inventory_items_by_name,
    get_stock_level,
    InventorySnapshot, get_supplier_delivery_date
)


def _task(query: str) -> InventoryAgentOutput:
    """Task function: run the inventory agent with the given query."""
    result = new_inventory_agent().run_sync(query)
    output: InventoryAgentOutput = result.output
    print(output.model_dump_json(indent=2))
    return output


@dataclass
class HasStockOrdersTransactionsOnly(Evaluator):
    def evaluate(self, ctx: EvaluatorContext[str, InventoryAgentOutput]):
        return {
            "has_stock_orders_transactions_only": all(
                transaction.transaction_type == "stock_orders" for transaction in ctx.output.placed_transactions
            )
        }


@dataclass
class HasPlacedTransactionsSize(Evaluator):
    size: int

    def evaluate(self, ctx: EvaluatorContext[str, InventoryAgentOutput]):
        return {
            "has_placed_transactions_size": self.size == len(ctx.output.placed_transactions)
        }


def compare_transactions(expected: FinancialTransaction, actual: FinancialTransaction, ) -> bool:
    if expected.item_name.lower() != actual.item_name.lower():
        return False
    if expected.transaction_type != actual.transaction_type:
        return False
    if expected.units != actual.units:
        return False
    if expected.price != actual.price:
        return False
    if expected.transaction_date != actual.transaction_date:
        return False

    return True


def check_all_exist(subset_list, main_list):
    # all() returns True if the inner condition is met for every element in subset_list
    return all(
        any(compare_transactions(sub_item, main_item) for main_item in main_list)
        for sub_item in subset_list
    )


@dataclass
class HasPlacedTransactions(Evaluator):
    """Custom evaluator: assert that the agent's output contains a placed
    stock_orders transaction for the given item."""

    expected_transactions: List[FinancialTransaction]

    def evaluate(self, ctx: EvaluatorContext[str, InventoryAgentOutput]):
        return {
            "has_placed_transactions": check_all_exist(self.expected_transactions, ctx.output.placed_transactions)
        }


@dataclass
class HasMessage(Evaluator):
    """Custom evaluator: assert that the agent output contains the expected message."""

    expected_message: str

    def evaluate(self, ctx: EvaluatorContext[str, InventoryAgentOutput]):
        return {
            "message_found": any(self.expected_message in msg for msg in ctx.output.messages)
        }


class TestInventoryAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Change the working directory to the project directory
        # so that read_csv can find "quotes.csv" and "quote_requests.csv" during init_database()
        project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project")
        os.chdir(project_dir)

    def setUp(self):
        # Remove the temp DB file so init_database starts clean
        if os.path.exists(_test_db_path):
            os.remove(_test_db_path)
        init_database(DB_ENGINE)

    def tearDown(self):
        # Dispose the engine to close any open file handles so SQLite doesn't lock it
        DB_ENGINE.dispose()
        # Remove the temp DB file so init_database starts clean
        if os.path.exists(_test_db_path):
            os.remove(_test_db_path)

    def test_quote_request_1(self):
        delivery_date = "2025-04-15"
        request_date = "2025-04-01"
        items = [
            {
                "item_name": "A4 glossy paper",
                "quantity": 200,
                "unit_price": 0.2,
                "total_amount": 36.0
            },
            {
                "item_name": "heavy cardstock",
                "quantity": 100,
                "unit_price": 0.15,
                "total_amount": 13.5
            },
            {
                "item_name": "colored paper",
                "quantity": 100,
                "unit_price": 0.1,
                "total_amount": 10.0
            }
        ]
        prompt = (
            "A customer has placed an order and we have generated the quote below. Please ensure the order can be fulfilled and replenish the inventory as needed following your rules.\n"
            f"- Desired delivery date: {delivery_date}\n"
            f"- Order Request Date: {request_date}\n"
            f"- Total base amount: 65.0\n"
            "- Total quoted amount: 59.5\n"
            "- Individual item quotes (in JSON format):\n"
            f"{json.dumps(items)}"
        )

        dataset = Dataset(
            name="test_quote_request_1",
            cases=[
                Case(
                    name="test_quote_request_1",
                    inputs=prompt
                ),
            ],
            evaluators=[
                HasStockOrdersTransactionsOnly(),
                # THERE MUST BE ONLY 2 TRANSACTIONS.
                # The current stock of 'colored paper' is 788, so there is enough to fulfill the order.
                HasPlacedTransactionsSize(2),
                HasPlacedTransactions([
                    # A4 glossy paper doesn't exist in inventory, so full amount should be ordered from supplier
                    FinancialTransaction(
                        item_name="A4 glossy paper",
                        transaction_type="stock_orders",
                        units=200,
                        price=200 * 0.2,
                        transaction_date=delivery_date
                    ),
                    # heavy cardstock doesn't exist in inventory, so full amount should be ordered from supplier
                    FinancialTransaction(
                        item_name="heavy cardstock",
                        transaction_type="stock_orders",
                        units=100,
                        price=100 * 0.15,
                        transaction_date=delivery_date
                    ),
                ])
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_replenish_to_minimum(self):
        """Test that Inventory Agent autonomously replenishes an item to min stock level"""
        item_name = "Kraft paper"
        request_date = "2025-04-01"
        inventory_items = get_inventory_items_by_name(["Kraft paper"])
        self.assertEqual(len(inventory_items), 1)
        self.assertEqual(inventory_items[0].unit_price, 0.1)
        self.assertEqual(inventory_items[0].min_stock_level, 64)

        # get stock level
        inventory_snapshot: InventorySnapshot = get_stock_level(item_name, request_date)
        self.assertIsNotNone(inventory_snapshot)
        self.assertEqual(len(inventory_snapshot.items), 1)

        stock = inventory_snapshot.items[item_name]
        self.assertEqual(stock, 493)

        # order quantity & date to get stock below min level
        quantity = 439 # This will bring level to 54
        delivery_date = get_supplier_delivery_date(request_date, quantity)
        quoted_price = round(quantity * inventory_items[0].unit_price * .9, 2) # 10% discount

        items = [
            {
                "item_name": item_name,
                "quantity": quantity,
                "unit_price": inventory_items[0].unit_price,
                "total_amount": quoted_price
            }
        ]
        prompt = (
            "A customer has placed an order and we have generated the quote below. Please ensure the order can be fulfilled and replenish the inventory as needed following your rules.\n"
            f"- Desired delivery date: {delivery_date}\n"
            f"- Order Request Date: {request_date}\n"
            f"- Total base amount: {quantity * inventory_items[0].unit_price}\n"
            f"- Total quoted amount: {quoted_price}\n"
            "- Individual item quotes (in JSON format):\n"
            f"{json.dumps(items)}"
        )

        dataset = Dataset(
            name="replenish_to_minimum",
            cases=[
                Case(
                    name="replenish_to_minimum",
                    inputs=prompt
                ),
            ],
            evaluators=[
                HasStockOrdersTransactionsOnly(),
                # THERE MUST BE 1 TRANSACTION.
                # There's enough stock to fulfill the order, but will bring the stock level down to 54 (10 below min)
                HasPlacedTransactionsSize(1),
                HasPlacedTransactions([
                    FinancialTransaction(
                        item_name=item_name,
                        transaction_type="stock_orders",
                        units=10,
                        price=10 * inventory_items[0].unit_price,
                        transaction_date=delivery_date
                    ),
                ])
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)


if __name__ == "__main__":
    unittest.main()
