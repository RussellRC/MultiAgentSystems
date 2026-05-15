import json
import os
import unittest
from dataclasses import dataclass

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
    InventoryAgentOutput
)


def _task(query: str) -> InventoryAgentOutput:
    """Task function: run the inventory agent with the given query."""
    result = new_inventory_agent().run_sync(query)
    output: InventoryAgentOutput = result.output
    print(output.model_dump_json(indent=2))
    return output


@dataclass
class HasInventoryItem(Evaluator):
    """Custom evaluator: assert that the agent output contains an inventory item
    matching the given item_name."""

    item_name: str

    def evaluate(self, ctx: EvaluatorContext[str, InventoryAgentOutput]):
        for item in ctx.output.inventory_items:
            if item.item_name.lower() == self.item_name.lower():
                return {
                    "item_found": True,
                    "category_matches": item.category is not None,
                    "has_unit_price": item.unit_price is not None,
                    "has_min_stock_level": item.min_stock_level is not None,
                }
        return {"item_found": False}


@dataclass
class HasCalculatedStockLevel(Evaluator):
    """Custom evaluator: assert that the agent output contains a calculated stock level
    for the given item_name with the expected stock value."""

    item_name: str
    expected_stock: int

    def evaluate(self, ctx: EvaluatorContext[str, InventoryAgentOutput]):
        if not ctx.output.calculated_stock_levels or not ctx.output.calculated_stock_levels.items:
            return {"stock_level_correct": False}
        if ctx.output.calculated_stock_levels.items.get(self.item_name) == self.expected_stock:
            return {"stock_level_correct": True}
        return {"stock_level_correct": False}


@dataclass
class HasPlacedTransaction(Evaluator):
    """Custom evaluator: assert that the agent's output contains a placed
    stock_orders transaction for the given item."""

    item_name: str | None = None
    units: int | None = None
    price: float | None = None
    transaction_date: str | None = None
    empty_transactions_expected: bool = False

    def evaluate(self, ctx: EvaluatorContext[str, InventoryAgentOutput]):
        if self.empty_transactions_expected:
            if not ctx.output.placed_transactions:
                return {"expected_empty_transactions": True}
            return {"expected_empty_transactions": False}

        for placed_transaction in ctx.output.placed_transactions:
            # Check transaction type
            if placed_transaction.transaction_type != "stock_orders":
                continue

            # Check item_name (if specified)
            if self.item_name and placed_transaction.item_name.lower() != self.item_name.lower():
                continue

            # Check units (if specified)
            if self.units is not None and placed_transaction.units != self.units:
                continue

            # Check price (if specified)
            if self.price is not None and placed_transaction.price != self.price:
                continue

            # Check transaction_date (if specified)
            if self.transaction_date is not None and placed_transaction.transaction_date != self.transaction_date:
                continue

            return {"expected_transaction_found": True}

        return {"expected_transaction_found": False}


@dataclass
class HasQueriedItems(Evaluator):
    """Custom evaluator: assert that the agent output contains at least
    the expected number of inventory items (showing it queried inventory)."""

    min_items: int

    def evaluate(self, ctx: EvaluatorContext[str, InventoryAgentOutput]):
        count = len(ctx.output.inventory_items)
        return {"queried_enough_items": count >= self.min_items}


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

    @unittest.skip("TODO")
    def test_prompt_from_orchestrator(self):
        delivery_date = "2025-01-08"
        initial_date = "2025-01-01"
        items = {
            "A4 paper": 5000
        }
        prompt = (
            "A customer has placed an order with the details below. Please check the inventory stock levels for each item and replenish as needed following your rules.\n"
            f"- Desired delivery date: {delivery_date}\n"
            f"- Order date: {initial_date}\n"
            f"- Items (in JSON format):\n"
            f"{json.dumps(items)}"
        )

        dataset = Dataset(
            name="prompt from orchestrator: A4 5000",
            cases=[
                Case(
                    name="prompt from orchestrator: A4 5000",
                    inputs=prompt,
                    expected_output=None,
                ),
            ],
            evaluators=[
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    @unittest.skip("INVENTORY AGENT NO LONGER RESPONDS TO THIS KIND OF REQUEST")
    def test_stock_level_at_date(self):
        """The agent should correctly compute stock levels as of a given date."""
        dataset = Dataset(
            name="stock_level_at_date",
            cases=[
                Case(
                    name="stock_level_2026",
                    inputs="What is the stock level of Paper plates as of 2026-01-01?",
                    expected_output=None,
                ),
            ],
            evaluators=[
                HasCalculatedStockLevel(item_name="Paper plates", expected_stock=748),
                HasPlacedTransaction(empty_transactions_expected=True)
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)


    def test_restock_order_happy_path(self):
        """Test that the agent can create a restock order when all necessary fields are provided."""
        dataset = Dataset(
            name="restock_order_happy_path",
            cases=[
                Case(
                    name="reorder_with_all_fields",
                    inputs="Please reorder 200 units of Paper plates for delivery with order date of 2026-01-01.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                HasPlacedTransaction(item_name="Paper plates", units=200, price=20.0, transaction_date="2026-01-05"),
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)


if __name__ == "__main__":
    unittest.main()
