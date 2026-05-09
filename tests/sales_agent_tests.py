import os
import unittest
from dataclasses import dataclass
from datetime import date, timedelta

from tests.test_utils import eval_report_cases

# Use a file-based SQLite DB with check_same_thread=False so the agent thread can share the same database.
_test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sales_agent.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}?check_same_thread=False"

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

from project.project import (
    init_database,
    DB_ENGINE,
    sales_agent,
    SalesAgentOutput,
    get_supplier_delivery_date,
)

def _task(query: str) -> SalesAgentOutput:
    result = sales_agent.run_sync(query)
    output: SalesAgentOutput = result.output
    print(output.model_dump_json(indent=2))
    return output

@dataclass
class HasOrderStatus(Evaluator):

    status: str

    def evaluate(self, ctx: EvaluatorContext[str, SalesAgentOutput]):
        if ctx.output.order_status == self.status:
            return {"order_status_verified": True}
        return {"order_status_verified": False}

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


class TestSalesAgentScenarioA(unittest.TestCase):

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

    def test_scenario_a_with_delivery_date(self):
        """
        Test fulfillment when a specific delivery date is requested and there is enough stock.
        Order date is not given, so agent must use TODAY.
        """
        requested_date = (date.today() + timedelta(days=3)).isoformat()
        quantity = 100
        expected_total = quantity * 0.05

        dataset = Dataset(
            name="Sales_Scenario_A_With_Date",
            cases=[
                Case(
                    name="order_in_stock_with_date",
                    inputs=f"I want to order {quantity} units of A4 paper. Deliver by {requested_date}.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="ACCEPTED"),
                HasPlacedSalesTransaction(
                    item_name="A4 paper",
                    units=quantity,
                    price=expected_total,
                    expected_date=requested_date
                )
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_scenario_a_without_delivery_date(self):
        """
        Test fulfillment when no delivery date is provided and there is enough stock.
        Neither order date nor delivery date is given, so the agent must use TODAY.
        """
        quantity = 50
        expected_total = quantity * 0.05
        today = date.today().isoformat()

        dataset = Dataset(
            name="Sales_Scenario_A_No_Date",
            cases=[
                Case(
                    name="order_in_stock_default_date",
                    inputs=f"Place an order for {quantity} units of A4 paper.",
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="ACCEPTED"),
                HasPlacedSalesTransaction(
                    item_name="A4 paper",
                    units=quantity,
                    price=expected_total,
                    expected_date=today
                )
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_scenario_b_fulfillable_with_restock(self):
        """
        Test that the agent accepts an order exceeding stock if the
        supplier can deliver within the requested deadline.
        NO order date is given, so agent must use TODAY.
        """
        quantity = 500  # Exceeds the 272 units in stock
        supplier_delivery = get_supplier_delivery_date(date.today().isoformat(), 228)
        requested_date = (date.fromisoformat(supplier_delivery) + timedelta(days=1)).isoformat()
        expected_total = quantity * 0.05

        dataset = Dataset(
            name="Sales_Scenario_B_Success",
            cases=[
                Case(
                    name="order_fulfillable_via_supplier",
                    inputs=(
                        f"I need {quantity} units of A4 paper. "
                        f"It's for a big event on {requested_date}, so I need them by then."
                    ),
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="ACCEPTED"),
                HasPlacedSalesTransaction(
                    item_name="A4 paper",
                    units=quantity,
                    price=expected_total,
                    expected_date=requested_date
                )
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_scenario_b_fulfillable_with_restock_given_order_date(self):
        """
        Test that the agent accepts an order exceeding stock if the
        supplier can deliver within the requested deadline.
        Order date is given, so agent must use it.
        """
        quantity = 500  # Exceeds the 272 units in stock
        order_date = '2025-01-01'
        supplier_delivery = get_supplier_delivery_date(order_date, 228) # '2025-01-05'
        requested_date = (date.fromisoformat(supplier_delivery) + timedelta(days=1)).isoformat() # '2025-01-06'
        expected_total = quantity * 0.05

        dataset = Dataset(
            name="Sales_Scenario_B_Success_With_Delivery_And_Order_Date",
            cases=[
                Case(
                    name="order_fulfillable_via_supplier",
                    inputs=(
                        f"I need {quantity} units of A4 paper. "
                        f"It's for a big event on {requested_date}, so I need them by then. "
                        f"Use {order_date} as the order date."
                    ),
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="ACCEPTED"),
                HasPlacedSalesTransaction(
                    item_name="A4 paper",
                    units=quantity,
                    price=expected_total,
                    expected_date=requested_date
                )
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_scenario_c_non_fulfillable(self):
        """
        Test that the agent DECLINES an order:
        - Exceeding stock
        - Supplier can deliver AFTER the requested deadline.
        - Order date is NOT given, so agent must use today.
        """
        quantity = 500  # Exceeds the 272 units in stock
        supplier_delivery = get_supplier_delivery_date(date.today().isoformat(), 228)
        requested_date = (date.fromisoformat(supplier_delivery) - timedelta(days=1)).isoformat()

        dataset = Dataset(
            name="Sales_Scenario_C_Declined_With_Delivery_Date",
            cases=[
                Case(
                    name="order_fulfillable_via_supplier",
                    inputs=(
                        f"I need to order {quantity} units of A4 paper by {requested_date}."
                    ),
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="DECLINED"),
                HasEmptySalesTransactions()
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_scenario_c_non_fulfillable_with_order_date(self):
        """
        Test that the agent DECLINES an order:
        - Exceeding stock
        - Supplier can deliver AFTER the requested deadline.
        - Both delivery and order dates are given.
        """
        quantity = 500  # Exceeds the 272 units in stock
        order_date = '2025-01-01'
        supplier_delivery = get_supplier_delivery_date(order_date, 228) # '2025-01-05'
        requested_date = (date.fromisoformat(supplier_delivery) - timedelta(days=1)).isoformat() # '2025-01-04'

        dataset = Dataset(
            name="Sales Scenario C: Should decline. Both delivery and order date are given.",
            cases=[
                Case(
                    name="order_fulfillable_via_supplier",
                    inputs=(
                        f"I need to order {quantity} units of A4 paper by {requested_date}. Use {order_date} as the order date."
                    ),
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="DECLINED"),
                HasEmptySalesTransactions()
            ],
        )
        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_apply_discounted_quote_price(self):
        """
        Verifies that the Sales Agent uses the price from the provided quote
        rather than the default catalog price.
        """
        item = "A4 paper"
        qty = 2000
        total_price = qty * 0.05  # Catalog price
        quoted_price = total_price * 0.9  # 10% discount applied by Quoting Agent
        supplier_delivery = get_supplier_delivery_date(date.today().isoformat(), qty)
        delivery_date = (date.fromisoformat(supplier_delivery) + timedelta(days=1)).isoformat()

        # Simulating the Orchestrator's job of passing context
        prompt = (
            f"The customer wants to finalize an order. "
            f"Context: They previously received a quote for {qty} units of {item} at a total price of ${quoted_price:.2f}. "
            f"Please fulfill this order for delivery by {delivery_date}."
        )

        dataset = Dataset(
            name="Sales_Discount_Verification",
            cases=[
                Case(
                    name="accept_quoted_discount",
                    inputs=prompt,
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="ACCEPTED"),
                HasPlacedSalesTransaction(
                    item_name=item,
                    units=qty,
                    price=quoted_price,
                    expected_date=delivery_date
                )
            ],
        )

        report = dataset.evaluate_sync(_task)
        self.assertEqual(len(report.failures), 0)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_decline_when_delivery_date_is_before_order_date(self):
        """
        Edge case: verify the agent fails when the delivery date is before the order date.
        """

        dataset = Dataset(
            name="decline_when_delivery_date_is_before_order_date",
            cases=[
                Case(
                    name="Decline when delivery date is before order date. Both dates given.",
                    inputs=(
                        "I want 10 sheets of paper by 2025-01-01. Use 2026-01-01 as the order date."
                    ),
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="DECLINED"),
                HasEmptySalesTransactions()
            ],
        )

        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

    def test_decline_when_delivery_date_is_before_order_date_without_order_date(self):
        """
        Edge case: verify the agent fails when the delivery date is before the order date.
        Order date is not given, so agent must use TODAY.
        """

        dataset = Dataset(
            name="decline_when_delivery_date_is_before_order_date_without_order_date",
            cases=[
                Case(
                    name="Decline when delivery date is before order date. No order date given.",
                    inputs=(
                        "I want 10 sheets of paper by 2025-01-01."
                    ),
                    expected_output=None,
                ),
            ],
            evaluators=[
                IsInstance(type_name="SalesAgentOutput"),
                HasOrderStatus(status="DECLINED"),
                HasEmptySalesTransactions()
            ],
        )

        report = dataset.evaluate_sync(_task)
        report.print()
        self.assertEqual(len(report.failures), 0, "No task failures expected")
        eval_report_cases(report)

if __name__ == "__main__":
    unittest.main()