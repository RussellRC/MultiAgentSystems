import os
import unittest
from dataclasses import dataclass

from tests.test_utils import eval_report_cases

# Use a file-based SQLite DB with check_same_thread=False so the agent thread can share the same database.
_test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_quoting_agent.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}?check_same_thread=False"

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from project.project import (
    init_database,
    DB_ENGINE,
    new_quoting_agent,
    ItemQuote,
)


def _task(query: str) -> ItemQuote:
    """Task function: run the quoting agent with the given query."""
    result = new_quoting_agent().run_sync(query)
    output: ItemQuote = result.output
    print(output.model_dump_json(indent=2))
    return output

@dataclass
class QuoteCalculationEvaluator(Evaluator):

    unit_price: float
    quantity: int

    """
    Validates that the QuoteCalculation is mathematically sound and consistent.
    """
    def evaluate(self, ctx: EvaluatorContext[str, ItemQuote]):
        calc = ctx.output.quote_calculation

        # Verify basic math: total = unit_price * quantity * (1 - discount_rate)
        expected_base = round(calc.unit_price * calc.quantity, 2)
        expected_total = round(expected_base * (1 - calc.discount_rate), 2)
        actual_total = round(calc.total_amount, 2)

        math_is_correct = actual_total == expected_total

        return {
            "quantity_matches": calc.quantity == self.quantity,
            "unit_price_matches": calc.unit_price == self.unit_price,
            "math_consistent": math_is_correct,
        }

@dataclass
class HasReasonableTotal(Evaluator):
    """Custom evaluator: assert that the quote total is within reasonable bounds."""
    
    min_amount: float
    max_amount: float
    
    def evaluate(self, ctx: EvaluatorContext[str, ItemQuote]):
        if ctx.output.quote_calculation.total_amount is None:
            return {"total_reasonable": False}
        return {
            "total_reasonable": self.min_amount <= ctx.output.quote_calculation.total_amount <= self.max_amount
        }

@dataclass
class HasExplanation(Evaluator):
    """Custom evaluator: assert that the quote has a non-empty explanation."""

    may_have_words: list[str] = None
    must_have_words: list[str] = None

    def evaluate(self, ctx: EvaluatorContext[str, ItemQuote]):

        if not ctx.output.quote_explanation:
            return {
                "has_explanation": False,
                "may_have_words": False,
                "must_have_words": False
            }

        return {
            "has_explanation": len(ctx.output.quote_explanation.strip()) > 0,
            "may_have_words": any(word in ctx.output.quote_explanation for word in self.may_have_words),
            "must_have_words": all(word in ctx.output.quote_explanation for word in self.must_have_words)
        }

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

    def test_orchestrator_prompt_letter_sized_paper(self):
        """
        Test using a prompt as sent from the Orchestrator.
        Tes with both given unit price and without it in the prompt.
        NOTE: The expected unit price is 0.06 from the first row in `quotes.csv`
        """
        item_name = "letter-sized paper"
        quantity = 300
        unit_price = 0.06
        base_total = quantity * unit_price
        prompt = (
            "Please give me a quote for the following item of a customer request:\n"
            f"- Item name: {item_name}\n"
            f"- Quantity: {quantity}\n"
        )

        dataset = Dataset(
            name="orchestrator_prompt_letter_sized_paper",
            cases=[
                # Without unit price, the agent should derive it from the search_quote_history
                Case(
                    name=f"{quantity} {item_name} without unit price",
                    inputs=prompt,
                ),
                Case(
                    name=f"{quantity} {item_name} with unit price",
                    inputs=prompt + f"- Unit price: {unit_price}\n",
                ),
            ],
            evaluators=[
                QuoteCalculationEvaluator(quantity=quantity, unit_price=unit_price),
                HasReasonableTotal(min_amount=base_total*0.9, max_amount=base_total),
                HasExplanation(must_have_words=["delivery"], may_have_words=["discount", "bulk"]) # May have discount
            ],
        )
        report = dataset.evaluate_sync(_task, max_concurrency=1)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_orchestrator_prompt_large_quantity(self):
        """Must have discount"""
        item_name = "A4 paper"
        quantity = 10000
        unit_price = 0.05
        base_total = quantity * unit_price
        prompt = (
            "Please give me a quote for the following item of a customer request:\n"
            f"- Item name: {item_name}\n"
            f"- Quantity: {quantity}\n"
        )

        dataset = Dataset(
            name="orchestrator_prompt_large_quantity",
            cases=[
                Case(
                    name=f"{quantity} {item_name} WITHOUT unit price",
                    inputs=prompt,
                ),
                Case(
                    name=f"{quantity} {item_name} WITH unit price",
                    inputs=prompt + f"- Unit price: {unit_price}\n",
                ),
            ],
            evaluators=[
                QuoteCalculationEvaluator(quantity=quantity, unit_price=unit_price),
                HasReasonableTotal(min_amount=base_total*0.8, max_amount=base_total*0.95),
                HasExplanation(must_have_words=["delivery", "discount", "bulk"])
            ],
        )
        report = dataset.evaluate_sync(_task, max_concurrency=1)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)


if __name__ == "__main__":
    unittest.main()
