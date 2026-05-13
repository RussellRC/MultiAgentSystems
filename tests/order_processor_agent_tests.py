import os
import unittest
from dataclasses import dataclass
from datetime import date
from typing import List, Dict

from tests.test_utils import eval_report_cases

# Use a file-based SQLite DB with check_same_thread=False so the agent thread can share the same database.
_test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_order_processor_agent.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}?check_same_thread=False"

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

from project.project import (
    init_database,
    DB_ENGINE,
    new_order_processor_agent,
    CustomerRequestDetails,
    ORDER_PROCESSOR_INSTRUCTIONS
)


def _task(query: str) -> CustomerRequestDetails:
    """Task function: run the Order Procesor agent with the given query."""
    result = new_order_processor_agent().run_sync(user_prompt=query, instructions=ORDER_PROCESSOR_INSTRUCTIONS)
    output: CustomerRequestDetails = result.output
    print(output.model_dump_json(indent=2))
    return output

@dataclass
class HasExpectedRequiredFields(Evaluator):
    """Custom evaluator: assert that the parsed request produced the required fields."""
    request_status: str
    items: Dict[str, int]
    words: List[str] = None
    date: str | None = None

    def evaluate(self, ctx: EvaluatorContext[str, CustomerRequestDetails]):
        output = ctx.output

        if output.request_status.lower() != "declined" and output.request_status.lower() != "accepted":
            return {"has_valid_request_status": False}

        evaluation = {
            "has_valid_request_status": True,
            "has_expected_request_status": False,
            "has_expected_words": True,
            "has_expected_items": False,
            "has_expected_date": True
        }

        if self.request_status.lower() == "declined":
            if output.request_status.lower() == "declined":
                evaluation["has_expected_request_status"] = True
        elif self.request_status.lower() == "accepted":
            if output.request_status.lower() == "accepted":
                evaluation["has_expected_request_status"] = True

        if self.date:
            evaluation["has_expected_date"] = self.date == output.delivery_date

        if self.words:
            evaluation["has_expected_words"] = any(word in message for word in self.words for message in output.messages)

        if not self.items:
            evaluation["has_expected_items"] = True
        else:
            output_items = {k.lower(): v for k, v in output.items}
            expected_items = {k.lower(): v for k, v in self.items}
            evaluation["has_expected_items"] = all(item in output_items for item in expected_items)

        return evaluation


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

    def test_simple_request(self):
        dataset = Dataset(
            name="simple_request",
            cases=[
                Case(
                    name="simple request",
                    inputs="I would like to order 500 reams of A4 paper.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="CustomerRequestDetails"),
                HasExpectedRequiredFields(
                    request_status="ACCEPTED",
                    items={
                        "A4 paper": 500
                    },
                    date=date.today().isoformat()
                )
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_request_with_unknown_items(self):
        dataset = Dataset(
            name="request_with_unknown_items",
            cases=[
                Case(
                    name="letter-sized paper is unknown",
                    inputs="I would like to request a large order of high-quality paper supplies for an upcoming event. "
                           "We need 500 reams of A4 paper, 300 reams of letter-sized paper, and 200 reams of cardstock. "
                           "Please ensure the delivery is made by April 15, 2025. Thank you.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="CustomerRequestDetails"),
                HasExpectedRequiredFields(
                    request_status="ACCEPTED",
                    items={
                        "A4 paper": 500,
                        "Letter-sized paper": 300,
                        "Cardstock": 200
                    },
                    date="2025-04-15"
                )
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)


if __name__ == "__main__":
    unittest.main()
