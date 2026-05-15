import os
import unittest
from dataclasses import dataclass
from datetime import date, timedelta

from tests.test_utils import eval_report_cases

# Use a file-based SQLite DB with check_same_thread=False so the agent thread can share the same database.
_test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sales_agent.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}?check_same_thread=False"

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from project.project import (
    init_database,
    DB_ENGINE,
    new_sales_agent,
    SalesAgentOutput,
    get_supplier_delivery_date,
)

def _task(query: str) -> SalesAgentOutput:
    result = new_sales_agent().run_sync(query)
    output: SalesAgentOutput = result.output
    print(output.model_dump_json(indent=2))
    return output

@dataclass
class HasRequestStatus(Evaluator):

    status: str

    def evaluate(self, ctx: EvaluatorContext[str, SalesAgentOutput]):
        if ctx.output.request_status == self.status:
            return {"request_status_verified": True}
        return {"request_status_verified": False}

@dataclass
class HasPlacedSalesTransaction(Evaluator):
    """Custom evaluator: checks if the agent created the correct 'sales' transaction."""
    item_name: str
    units: int
    price: float
    expected_date: str | None = None

    def evaluate(self, ctx: EvaluatorContext[str, SalesAgentOutput]):
        for placed_transaction in ctx.output.placed_transactions:
            if (placed_transaction.transaction_type == "sales" and
                    placed_transaction.item_name.lower() == self.item_name.lower() and
                    placed_transaction.units == self.units and
                    abs(placed_transaction.price - self.price) < 0.01):

                # Check date if specified
                if self.expected_date and placed_transaction.transaction_date != self.expected_date:
                    continue
                return {"transaction_verified": True}
        return {"transaction_verified": False}

@dataclass
class HasEmptySalesTransactions(Evaluator):

    def evaluate(self, ctx: EvaluatorContext[str, SalesAgentOutput]):
        if not ctx.output.placed_transactions:
            return {"empty_transactions_verified": True}
        return {"empty_transaction_verified": False}


class TestSalesAgent(unittest.TestCase):

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

    def test_request_1(self):
        item_name = "A4 paper"
        quantity = 500
        total_amount = 22.5
        delivery_date = "2025-01-05"

        dataset = Dataset(
            name="test_request_1",
            cases=[
                Case(
                    name="test_request_1",
                    inputs=(
                        "The customer wants to finalize an order. Record the sale transaction for the following item:\n"
                        f"- Item name: {item_name}\n"
                        f"- Ordered quantity: {quantity} units\n"
                        f"- Quoted total price ${total_amount:.2f}.\n"
                        f"- Delivery date: {delivery_date}."
                    ),
                    expected_output=None,
                ),
            ],
            evaluators=[
                HasRequestStatus(status="ACCEPTED"),
                HasPlacedSalesTransaction(
                    item_name=item_name,
                    units=quantity,
                    price=total_amount,
                    expected_date="2025-01-05"
                )
            ],
        )

        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

if __name__ == "__main__":
    unittest.main()
