import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import date, timedelta
from unittest.mock import patch
from dotenv import load_dotenv

from sqlalchemy import text

# Use a file-based SQLite DB with check_same_thread=False so the agent thread can share the same database.
_test_db_path = os.path.join(tempfile.gettempdir(), "test_inv_agent.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}?check_same_thread=False"

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

from project.project import (
    init_database,
    DB_ENGINE,
    inventory_agent,
    InventoryAgentOutput,
    FinancialTransaction
)


def _task(query: str) -> InventoryAgentOutput:
    """Task function: run the inventory agent with the given query."""
    result = inventory_agent.run_sync(query)
    output: InventoryAgentOutput = result.output
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


def _assert_eval_results(case_result):
    """Assert all scores and assertions on a single case result passed."""
    for score_name, score in case_result.scores.items():
        assert score.value, f"{score_name} failed for {case_result.name}"
    for assertion_name, assertion in case_result.assertions.items():
        assert assertion.value, f"{assertion_name} failed for {case_result.name}"


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

    def test_inventory_query_returns_item_details(self):
        """The agent should fetch and return details about a specific inventory item."""
        dataset = Dataset(
            name="inventory_query",
            cases=[
                Case(
                    name="query_paper_plates",
                    inputs="What is the current stock of Paper plates?",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="InventoryAgentOutput"),
                HasInventoryItem(item_name="Paper plates"),
                HasCalculatedStockLevel(item_name="Paper plates", expected_stock=748),
                HasPlacedTransaction(empty_transactions_expected=True)
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        for case_result in report.cases:
            _assert_eval_results(case_result)


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
                IsInstance(type_name="InventoryAgentOutput"),
                HasCalculatedStockLevel(item_name="Paper plates", expected_stock=748),
                HasPlacedTransaction(empty_transactions_expected=True)
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        for case_result in report.cases:
            _assert_eval_results(case_result)


    def test_restock_order_happy_path(self):
        """Test that the agent can create a restock order when all necessary fields are provided."""
        dataset = Dataset(
            name="restock_order_happy_path",
            cases=[
                Case(
                    name="reorder_with_all_fields",
                    inputs="Please reorder 200 units of Paper plates for delivery starting from 2026-01-01.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="InventoryAgentOutput"),
                HasPlacedTransaction(item_name="Paper plates", units=200, price=20.0, transaction_date="2026-01-05"),
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        for case_result in report.cases:
            _assert_eval_results(case_result)


    def test_restock_order_double_min_quantity(self):
        """The agent should create a restock order for Paper plates to reach 2x its minimum quantity
        after a sales transaction reduces its stock below the minimum."""

        # First, get the current min_stock_level for "Paper plates"
        with DB_ENGINE.connect() as connection:
            result = connection.execute(
                text("SELECT min_stock_level FROM inventory WHERE item_name = :item_name"),
                {"item_name": "Paper plates"}
            ).fetchone()
            min_stock_level = result[0] if result else 0

            # Insert a sales transaction to bring the stock below min_stock_level
            # Assuming current stock is 748 (from other tests), let's sell enough to go below min_stock_level
            current_stock = 748 # Based on initial data in init_database
            quantity_to_sell = current_stock - min_stock_level + 10 # Ensure it goes below min_stock_level

            connection.execute(
                text(
                    """
                    INSERT INTO transactions (id, item_name, transaction_type, units, price, transaction_date)
                    VALUES (:id, :item_name, :transaction_type, :units, :price, :transaction_date)
                    """
                ),
                {
                    "id": "test-sale-123",
                    "transaction_type": "sales",
                    "item_name": "Paper plates",
                    "units": quantity_to_sell,
                    "price": 0.1 * float(quantity_to_sell),
                    "transaction_date": "2025-01-02"
                }
            )
            connection.commit()

        expected_date = (date.today() + timedelta(days=4)).isoformat()

        # Now, run the agent to trigger the restock
        dataset = Dataset(
            name="restock_paper_plates_double_min",
            cases=[
                Case(
                    name="restock_paper_plates",
                    inputs="Restock the inventory of Paper plates.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="InventoryAgentOutput"),
                HasPlacedTransaction(item_name="Paper plates", units=154, price=15.4, transaction_date=expected_date)
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        for case_result in report.cases:
            _assert_eval_results(case_result)


if __name__ == "__main__":
    unittest.main()