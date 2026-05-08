import os
import tempfile
import unittest
from datetime import date
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

from project.project import (
    init_database,
    DB_ENGINE,
    quoting_agent,
    QuotingAgentOutput, get_supplier_delivery_date,
)

# Use a file-based SQLite DB with check_same_thread=False so the agent thread can share the same database.
_test_db_path = os.path.join(tempfile.gettempdir(), "test_quoting_agent.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}?check_same_thread=False"


def _task(query: str) -> QuotingAgentOutput:
    """Task function: run the quoting agent with the given query."""
    result = quoting_agent.run_sync(query)
    output: QuotingAgentOutput = result.output
    print(output.model_dump_json(indent=2))
    return output


@dataclass
class HasReasonableTotal(Evaluator):
    """Custom evaluator: assert that the quote total is within reasonable bounds."""
    
    min_amount: float
    max_amount: float
    
    def evaluate(self, ctx: EvaluatorContext[str, QuotingAgentOutput]):
        if ctx.output.quote_calculation.total_amount is None:
            return {"total_reasonable": False}
        return {
            "total_reasonable": self.min_amount <= ctx.output.quote_calculation.total_amount <= self.max_amount
        }


@dataclass
class HasExplanation(Evaluator):
    """Custom evaluator: assert that the quote has a non-empty explanation."""

    expected_words: list[str] = None

    def evaluate(self, ctx: EvaluatorContext[str, QuotingAgentOutput]):
        if not ctx.output.quote_explanation:
            return {"has_explanation": False}
        if not self.expected_words:
            return {"has_explanation": len(ctx.output.quote_explanation) > 0}
        explanation = ctx.output.quote_explanation.lower()
        return any(word.lower() in explanation for word in self.expected_words)


@dataclass
class HasDeliveryDate(Evaluator):
    """Custom evaluator: assert that the estimated delivery date is in ISO format."""

    expected_date: str = None

    def evaluate(self, ctx: EvaluatorContext[str, QuotingAgentOutput]):
        if not ctx.output.estimated_delivery_date:
            return {"has_delivery_date": False}
        try:
            from datetime import date
            # Try to parse as ISO format date
            date.fromisoformat(ctx.output.estimated_delivery_date)
            if self.expected_date:
                return {"has_delivery_date": True, "date_valid": ctx.output.estimated_delivery_date == self.expected_date}
            return {"has_delivery_date": True, "date_valid": True}
        except (ValueError, TypeError):
            return {"has_delivery_date": True, "date_valid": False}


class TestQuotingAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the working directory and load environment variables."""
        project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project")
        os.chdir(project_dir)

    def setUp(self):
        """Initialize a fresh database for each test."""
        if os.path.exists(_test_db_path):
            os.remove(_test_db_path)
        init_database(DB_ENGINE)

    def tearDown(self):
        """Clean up after each test."""
        DB_ENGINE.dispose()
        if os.path.exists(_test_db_path):
            os.remove(_test_db_path)

    def test_simple_quote_a4_paper(self):
        """
        Test the simplest case: quote for 100 sheets of A4 paper with NO discounts.
        Expected: 100 * $0.05 = $5.00 base price.
        """
        quantity = 100
        unit_price = 0.05
        expected_date = get_supplier_delivery_date(date.today().isoformat(), quantity)
        dataset = Dataset(
            name="simple_quote_a4",
            cases=[
                Case(
                    name="quote_100_sheets_a4",
                    inputs=f"Give me a quote for {quantity} sheets of A4 paper. DO NOT apply any discounts.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="QuotingAgentOutput"),
                HasReasonableTotal(min_amount=quantity*unit_price, max_amount=quantity*unit_price),
                HasExplanation([str(quantity), "a4"]),
                HasDeliveryDate(expected_date=expected_date),
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        for case_result in report.cases:
            for score_name, score in case_result.scores.items():
                self.assertTrue(score.value, f"{score_name} failed for {case_result.name}")
            for assertion_name, assertion in case_result.assertions.items():
                self.assertTrue(assertion.value, f"{assertion_name} failed for {case_result.name}")


    def test_large_amount_quote_a4_paper(self):
        """
        Test quote for 10000 sheets of A4 paper. Should have a discount due to quantity.
        """
        quantity = 10000
        base_total = quantity * 0.05
        delivery_date = get_supplier_delivery_date(date.today().isoformat(), quantity)

        dataset = Dataset(
            name="large_quote_a4",
            cases=[
                Case(
                    name="quote_10000_sheets_a4",
                    inputs=f"Give me a quote for {quantity} sheets of A4 paper.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="QuotingAgentOutput"),
                HasReasonableTotal(min_amount=400.0, max_amount=base_total - 0.01),
                HasExplanation(expected_words=["discount", "bulk"]),
                HasDeliveryDate(expected_date=delivery_date),
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()

        self.assertEqual(len(report.failures), 0, "No task failures expected")
        for case_result in report.cases:
            for score_name, score in case_result.scores.items():
                self.assertTrue(score.value, f"{score_name} failed for {case_result.name}")
            for assertion_name, assertion in case_result.assertions.items():
                self.assertTrue(assertion.value, f"{assertion_name} failed for {case_result.name}")


    def test_quote_a4_glossy_paper(self):
        """
        Test quote for 100 sheets of A4 Glossy paper.
        A4 Glossy is ambiguous as it matches 2 item names, so the agent should choose the one with the highest price.
        """
        quantity = 100
        base_total = quantity * 0.2
        delivery_date = get_supplier_delivery_date(date.today().isoformat(), quantity)

        dataset = Dataset(
            name="large_quote_a4",
            cases=[
                Case(
                    name="quote_100_sheets_glossy_a4",
                    inputs=f"Give me a quote for {quantity} sheets of A4 glossy paper.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="QuotingAgentOutput"),
                HasReasonableTotal(min_amount=base_total*0.9, max_amount=base_total),
                HasExplanation(["glossy", "a4"]),
                HasDeliveryDate(expected_date=delivery_date),
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()

        self.assertEqual(len(report.failures), 0, "No task failures expected")
        for case_result in report.cases:
            for score_name, score in case_result.scores.items():
                self.assertTrue(score.value, f"{score_name} failed for {case_result.name}")
            for assertion_name, assertion in case_result.assertions.items():
                self.assertTrue(assertion.value, f"{assertion_name} failed for {case_result.name}")


    def test_simple_quote_a4_paper_with_order_date(self):
        """
        Test a quote with an order date
        """
        order_date = "2026-01-01"
        quantity = 100
        base_total = quantity * 0.05
        expected_date = get_supplier_delivery_date(order_date, quantity)
        dataset = Dataset(
            name="simple_quote_a4",
            cases=[
                Case(
                    name="quote_100_sheets_a4_with_date",
                    inputs=f"I need a quote for {quantity} sheets of A4 paper with order date {order_date}",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="QuotingAgentOutput"),
                HasReasonableTotal(min_amount=base_total * .9, max_amount=base_total),
                HasExplanation(),
                HasDeliveryDate(expected_date=expected_date),
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        for case_result in report.cases:
            for score_name, score in case_result.scores.items():
                self.assertTrue(score.value, f"{score_name} failed for {case_result.name}")
            for assertion_name, assertion in case_result.assertions.items():
                self.assertTrue(assertion.value, f"{assertion_name} failed for {case_result.name}")


if __name__ == "__main__":
    unittest.main()
