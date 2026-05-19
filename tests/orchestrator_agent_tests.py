import os
import unittest
from dataclasses import dataclass

from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from tests.test_utils import eval_report_cases

# Use a file-based SQLite DB with check_same_thread=False so the agent thread can share the same database.
_test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_orchestrator_agent.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}?check_same_thread=False"

from project.project import OrchestratorAgent, OrchestratorAgentOutput, init_database, DB_ENGINE


def _task(inputs: dict) -> OrchestratorAgentOutput:
    """Task function: run the orchestrator agent with the given query."""
    orchestrator_agent = OrchestratorAgent()
    result = orchestrator_agent.process_customer_order(inputs["customer_request"])
    output = result.output
    print(output.model_dump_json(indent=2))
    return output

@dataclass
class HasReasonableTotal(Evaluator):
    """Custom evaluator: assert that the quote total is within reasonable bounds."""

    min_amount: float
    max_amount: float

    def evaluate(self, ctx: EvaluatorContext[str, OrchestratorAgentOutput]):
        if ctx.output.total_amount is None:
            return {"total_reasonable": False}
        return {
            "total_reasonable": self.min_amount <= ctx.output.total_amount <= self.max_amount
        }

@dataclass
class HasDeliveryDate(Evaluator):
    """Custom evaluator: assert that the final answer meets expected delivery date"""

    expected_date: str = None

    def evaluate(self, ctx: EvaluatorContext[str, OrchestratorAgentOutput]):
        if not ctx.output.delivery_date:
            return {"has_delivery_date": False}
        try:
            from datetime import date
            # Try to parse as ISO format date
            date.fromisoformat(ctx.output.delivery_date)
            if self.expected_date:
                return {"has_delivery_date": True, "date_valid": ctx.output.delivery_date == self.expected_date}
            return {"has_delivery_date": True, "date_valid": True}
        except (ValueError, TypeError):
            return {"has_delivery_date": True, "date_valid": False}

class TestOrchestratorAgent(unittest.TestCase):

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

    def test_simple_request_no_order_date(self):
        dataset = Dataset(
            name="simple_request_no_order_date",
            cases=[
                Case(
                    name="simple_request_no_order_date",
                    inputs={
                        "customer_request": "I would like to order 5000 reams of A4 paper. (Date of request: 2025-01-01)",
                    }
                ),
            ],
            evaluators=(
                HasReasonableTotal(min_amount=5000*0.05*.9, max_amount=5000*0.05),
                HasDeliveryDate()
            ),
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_quote_request_1(self):
        """Test request 1 from the quote_requests.csv file"""
        a4_price = 500 * 0.05 # from inventory
        letter_price = 300 * 0.06 # from history
        cardstock_price = 200 * 0.15 # from inventory
        total_base = a4_price + letter_price + cardstock_price

        dataset = Dataset(
            name="test_quote_request_1",
            cases=[
                Case(
                    name="test_quote_request_1",
                    inputs={
                        "customer_request": "I would like to request a large order of high-quality paper supplies for an upcoming event. "
                                            "We need 500 reams of A4 paper, 300 reams of letter-sized paper, and 200 reams of cardstock. "
                                            "Please ensure the delivery is made by April 15, 2025. Thank you. "
                                            "(Request date: 2025-01-01)",
                    },
                    evaluators=(
                        HasDeliveryDate(),
                        HasReasonableTotal(min_amount=total_base*.9, max_amount=total_base),
                    )
                ),
            ]
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_quote_request_3(self):
        """Test request 3 from the quote_requests.csv file"""
        a4_price = 10000 * 0.05 # from inventory
        a3_price = 5000 * 0.10 # from history
        printer_price = 500 * 0.05 # from history
        total_base = a4_price + a3_price + printer_price

        dataset = Dataset(
            name="test_quote_request_3",
            cases=[
                Case(
                    name="test_quote_request_3",
                    inputs={
                        "customer_request": "I need to order 10,000 sheets of A4 paper, 5,000 sheets of A3 paper, and 500 reams of printer paper. "
                                            "The supplies must be delivered by April 15, 2025, for our upcoming conference. "
                                            "Please confirm the order and delivery schedule. (Date of request: 2025-04-04)",
                    },
                    evaluators=(
                        HasDeliveryDate(),
                        HasReasonableTotal(min_amount=total_base*.9, max_amount=total_base),
                    )
                ),
            ]
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)


if __name__ == "__main__":
    unittest.main()
