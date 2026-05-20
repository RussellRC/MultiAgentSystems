import ast
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import Dict, List, cast, Any, Annotated

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from nocasedict import NocaseDict
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, BeforeValidator, PlainSerializer
from pydantic_ai import Agent, ModelRetry, RunContext, Tool, AgentRunResult, ModelSettings
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals.evaluators import LLMJudge, EvaluatorContext, EvaluationReason
from pydantic_evals.otel import SpanTree
from pythonjsonlogger.json import JsonFormatter
from sqlalchemy import create_engine, Engine, bindparam
from sqlalchemy.sql import text

# Configure logging
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("project")
logHandler = logging.StreamHandler()
formatter = JsonFormatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s',
    rename_fields={"asctime": "time", "levelname": "level"})
logHandler.setFormatter(formatter)
logging.getLogger().addHandler(logHandler)


def _load_log_levels() -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f).get("log_levels", {})
    return {
        "root": "INFO"
    }


_log_levels = _load_log_levels()


def _resolve_level(name: str, default: int) -> int:
    val = _log_levels.get(name)
    if isinstance(val, str):
        return getattr(logging, val.upper(), default)
    return default


logging.getLogger().setLevel(_resolve_level("root", logging.INFO))
logger.setLevel(_resolve_level("project", logging.DEBUG))
logging.getLogger("openai").setLevel(_resolve_level("openai", logging.DEBUG))
logging.getLogger("httpcore").setLevel(_resolve_level("httpcore", logging.INFO))
logging.getLogger("httpx").setLevel(_resolve_level("httpx", logging.INFO))

# Create an SQLite database
DB_URL = os.getenv("DATABASE_URL", "sqlite:///munder_difflin.db")
DB_ENGINE = create_engine(DB_URL)

# List containing the different kinds of papers 
PAPER_SUPPLIES = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates", "category": "product", "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins", "category": "product", "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups", "category": "product", "unit_price": 0.10},  # per cup
    {"item_name": "Table covers", "category": "product", "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes", "category": "product", "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes", "category": "product", "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards", "category": "product", "unit_price": 0.50},  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers", "category": "product", "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags", "category": "product", "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards", "category": "product", "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders", "category": "product", "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]


#################
# MODEL CLASSES #
#################

def validate_nocase_dict(value: Any) -> NocaseDict:
    if isinstance(value, NocaseDict):
        return value
    if isinstance(value, dict):
        return NocaseDict(value)
    raise ValueError("Value must be a dictionary")


class InventorySnapshot(BaseModel):
    """
    Snapshot of calculated stock levels of an item or items as of a specific date.
    """
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    items: Annotated[
        NocaseDict,
        BeforeValidator(validate_nocase_dict),
        PlainSerializer(lambda x: dict(x), return_type=dict),
        Field(description=(
            "Dictionary holding the stock levels of each item, where: "
            "The key is the name of the item (must be a string). "
            "The value is the calculated quantity or stock level for that item (must be an integer)."
        ))
    ]


class InventoryItem(BaseModel):
    """
    An inventory item. Attributes include static details like category, unit price, and its minimum stock level.
    """
    model_config = ConfigDict(from_attributes=True)
    item_name: str = Field(description="Name of the item")
    category: str = Field(description="Category of the item")
    unit_price: float = Field(description="Unit price of the item")
    min_stock_level: int = Field(
        description="Minimum stock level: the minimum quantity that must be kept in stock for this item")


class FinancialTransaction(BaseModel):
    """
    A financial transaction. Can be a stock order or a sale.
    """
    model_config = ConfigDict(from_attributes=True)
    id: str | None = Field(description="Transaction ID (uuid7)", default=None)
    item_name: str = Field(description="Name of the item involved in the transaction")
    transaction_type: str = Field(description="Type of transaction ('stock_orders' or 'sales')")
    units: int = Field(description="Number of units involved in the transaction")
    price: float = Field(description="Total price of the transaction")
    transaction_date: str = Field(description=(
        "Date of the transaction. "
        "For an 'stock_order' transaction, it represents the date of delivery. "
        "For a 'sales' transaction, it represents the date of sale.")
    )


class TopSellingProduct(BaseModel):
    """
    Data about a top-selling product.
    """
    item_name: str = Field(description="Name of the item")
    total_units: int = Field(description="Total number of units sold")
    total_revenue: float = Field(description="Total revenue generated")


class FinancialReport(BaseModel):
    """
    A complete financial report for the company as of a specific date.
    """
    as_of_date: str = Field(description="Date up to which the report is generated in ISO format.",
                            examples=['2026-01-31'])
    cash_balance: float = Field(description="Current cash balance")
    inventory_value: float = Field(description="Total value of inventory")
    total_assets: float = Field(description="Combined cash and inventory value")
    inventory_summary: List[InventoryItem] = Field(description="Summary of the inventory")
    top_selling_products: List[TopSellingProduct] = Field(description="Top 5 products by revenue")


##################################
# UTILITY FUNCTIONS FROM STARTER #
##################################
# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],  # Quantity involved
            "price": [],  # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv(os.path.join(_DATA_DIR, "quote_requests.csv"))
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(os.path.join(_DATA_DIR, "quotes.csv"))
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(PAPER_SUPPLIES, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "id": str(uuid.uuid7()),
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "id": str(uuid.uuid7()),
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        logger.error("Error initializing database.", exc_info=True)
        raise


def create_transaction(item_name: str, transaction_type: str, quantity: int, price: float,
                       transaction_date: str | datetime, ) -> FinancialTransaction:
    """
    This function records a financial transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        transaction_date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        FinancialTransaction: The transaction that was created.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    logger.debug(
        "FUNC (create_transaction): Creating transaction.",
        extra={"item_name": item_name, "transaction_type": transaction_type, "quantity": quantity, "price": price,
               "date": transaction_date}
    )
    try:
        # Convert datetime to ISO string if necessary
        date_str = transaction_date.isoformat() if isinstance(transaction_date, datetime) else transaction_date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Insert Transaction
        with DB_ENGINE.begin() as conn:
            query = text("INSERT INTO transactions (id, item_name, transaction_type, units, price, transaction_date) "
                         "VALUES (:id, :item_name, :transaction_type, :units, :price, :transaction_date) "
                         "RETURNING *")
            params = {
                "id": str(uuid.uuid7()),
                "item_name": item_name,
                "transaction_type": transaction_type,
                "units": quantity,
                "price": price,
                "transaction_date": date_str
            }
            result = conn.execute(query, params)
            transaction = FinancialTransaction.model_validate(result.fetchone())
            return transaction

    except Exception as e:
        logger.error("Error creating transaction.", exc_info=True)
        raise


def get_all_inventory(as_of_date: str) -> InventorySnapshot:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
            SELECT item_name,
                   SUM(CASE
                           WHEN transaction_type = 'stock_orders' THEN units
                           WHEN transaction_type = 'sales' THEN -units
                           ELSE 0
                       END) as stock
            FROM transactions
            WHERE item_name IS NOT NULL
              AND transaction_date <= :as_of_date
            GROUP BY item_name
            HAVING stock > 0 \
            """

    # Execute the query with the date parameter
    result = pd.read_sql(query, DB_ENGINE, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    items = dict(zip(result["item_name"], result["stock"]))
    return InventorySnapshot(items=NocaseDict(items))


def get_stock_level(item_name: str, as_of_date: str | datetime) -> InventorySnapshot | None:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        The stock level (item quantity in stock) as of the given date.
    """
    logger.debug("FUNC (get_stock_level): Getting stock level for item.",
                 extra={"item_name": item_name, "as_of_date": as_of_date})

    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
                  SELECT LOWER(item_name) AS item_name,
                         COALESCE(SUM(CASE
                                          WHEN transaction_type = 'stock_orders' THEN units
                                          WHEN transaction_type = 'sales' THEN -units
                                          ELSE 0
                             END), 0)     AS stock_level
                  FROM transactions
                  WHERE LOWER(item_name) = LOWER(:item_name)
                    AND transaction_date <= :as_of_date \
                  """

    # Execute query and return result
    try:
        with DB_ENGINE.connect() as conn:
            result = conn.execute(text(stock_query), {"item_name": item_name, "as_of_date": as_of_date})
            row = result.fetchone()
            inventory_snapshot = None
            if row is not None and row[0] is not None:
                inventory_snapshot = InventorySnapshot(items=NocaseDict({row[0]: row[1]}))
            logger.debug("FUNC (get_stock_level): Returning stock level for item.",
                         extra={"item_name": item_name,
                                "as_of_date": as_of_date,
                                "stock_level": inventory_snapshot})
            return inventory_snapshot

    except Exception as e:
        logger.error("Error getting stock level", exc_info=True)
        raise


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD), e.g., '2026-01-31'.
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    logger.debug(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        logger.warning(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: str | datetime) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the 'transactions' table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            DB_ENGINE,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        logger.error("Error getting cash balance.", extra={"as_of_date": as_of_date}, exc_info=True)
        return 0.0


def generate_financial_report(as_of_date: str | datetime) -> FinancialReport:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        A complete financial report for the company as of a specific date.
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", DB_ENGINE)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        # Handle cases where stock_info might be None or unexpected format
        if stock_info and item["item_name"] in stock_info.items:
            stock = stock_info.items[item["item_name"]]
        else:
            stock = 0

        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": stock,
            "min_stock_level": item["min_stock_level"],
            "total_value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
                      SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
                      FROM transactions
                      WHERE transaction_type = 'sales'
                        AND transaction_date <= :date
                      GROUP BY item_name
                      HAVING total_units > 0
                      ORDER BY total_revenue DESC
                      LIMIT 5 \
                      """
    top_sales = pd.read_sql(top_sales_query, DB_ENGINE, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    report = {
        "as_of_date": as_of_date,
        "cash_balance": round(cash, 2),
        "inventory_value": round(inventory_value, 2),
        "total_assets": round(cash + inventory_value, 2),
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }
    return FinancialReport.model_validate(report)


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    # Sanitize search terms
    # search_terms = [term for string in search_terms for term in string.split()]
    # search_terms = [term for term in search_terms if term.lower() != "paper"]

    logger.debug("FUNC (search_quote_history): Searching quote history.",
                 extra={"search_terms": search_terms, "limit": limit})

    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with DB_ENGINE.connect() as conn:
        result = conn.execute(text(query), params)
        matching_quotes = [dict(row._mapping) for row in result]
        logger.debug(f"FUNC (search_quote_history): Returning quote history. Found {len(matching_quotes)} records.",
                     extra={"search_terms": search_terms, "limit": limit})
        return matching_quotes


###############################
###############################
###############################
# YOUR MULTI AGENT STARTS HERE
###############################
###############################
###############################

#########
# TOOLS #
#########

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


################
# Shared tools #
################

def get_current_date() -> str:
    """
    Gets the current date in ISO format (YYYY-MM-DD), e.g., "2026-01-31".

    Returns:
         The current date in ISO format.
    """
    return date.today().isoformat()


def get_all_item_names() -> list[str]:
    """
    Retrieves all unique catalog item names that may be quoted.

    Use these catalog names to decide whether a request contains one item or
    multiple items. Quantity words and units such as "sheets", "reams",
    "packs", and "boxes" are not item names.

    Return:
         A sorted list containing all unique catalog item names.
    """
    logger.debug("FUNC (get_all_item_names): Getting all item names.")
    item_names = [item['item_name'] for item in PAPER_SUPPLIES]
    return sorted(set(item_names))


def get_inventory_items_by_name(item_names: list[str]) -> List[InventoryItem]:
    """
    Get the static data (category, unit price, minimum stock levels) of the inventory items with the given name from the database.
    Returns None if none of the items exist.

    Args:
        item_names: List of the names of the items to retrieve.

    Returns:
        A list of InventoryItem elements
    """
    logger.debug("FUNC (get_inventory_items_by_name): Getting inventory items by name.",
                 extra={"item_names": item_names})

    if isinstance(item_names, str):
        item_names = [item_names]

    try:
        with DB_ENGINE.connect() as conn:
            query = text("SELECT * FROM inventory WHERE item_name COLLATE NOCASE IN :item_names")
            query = query.bindparams(bindparam("item_names", expanding=True))
            result = conn.execute(query, {"item_names": item_names})
            rows = result.mappings().all()
            if not rows:
                return []

            adapter = TypeAdapter(List[InventoryItem])
            inventory_items = adapter.validate_python(list(rows))
            return inventory_items

    except Exception as e:
        logger.error("Error getting inventory items.", exc_info=True)
        raise


###########################
# Tools for Inventory Agent
###########################

class ItemStockAnalysis(BaseModel):
    """Detailed breakdown of stock requirements for a single ordered item."""
    item_name: str = Field(description="Name of the catalog item.")
    requested_quantity: int = Field(description="Quantity requested by the customer.")
    current_stock_at_date: int = Field(
        description="Projected stock level for the item exactly on the requested delivery date.")
    min_stock_level: int = Field(description="The minimum stock level required for this item.")

    # Fulfillment Analysis
    can_fulfill_order_item: bool = Field(description="Whether this item can be fulfilled by the desired delivery date.")
    shortage_for_order: int = Field(
        description="Units missing strictly to fulfill the order (0 if can_fulfill_order is True).")
    stock_after_fulfillment: int = Field(
        description="Projected stock after fulfilling this order (can be negative if order is unfulfillable without restocking).")
    drops_below_minimum: bool = Field(
        description="Whether the stock for this item will drop below its minimum stock level. This is True if stock_after_fulfillment falls below the min_stock_level of the item.")
    supplier_delivery_date: str | None = Field(
        description="Date in ISO format (YYYY-MM-DD) by which the supplier can deliver missing units to fulfill the order. Can be missing if shortage is 0 or order can not be fulfilled in time",
        default=None, examples=["2026-01-31"])


class OrderStockAnalysisResult(BaseModel):
    """Complete analysis of an order's impact on inventory."""
    as_of_date: str = Field(description="The desired delivery date of the order. Used for the stock snapshot.")
    can_fulfill_order: bool = Field(
        description="Whether the whole order can be fulfilled by the desired delivery date. This is True if stock at date >= requested quantity for all items in the order.")
    items_analysis: List[ItemStockAnalysis] = Field(
        description="Analysis breakdown for each requested item of the order.")


def analyze_order_stock_requirements(order_items: Dict[str, int], as_of_date: str,
                                     request_date: str) -> OrderStockAnalysisResult:
    """
    Analyzes a customer order against projected stock levels to determine if the order can be fulfilled,
    taking into account if the supplier can cover shortage in time if needed.

    Args:
        order_items: A dictionary mapping order item names to requested quantities.
        as_of_date (str): The requested delivery date of the Order, in ISO format (YYYY-MM-DD), e.g., 2026-01-31.
        request_date (str): The date on which the order is made, in ISO format (YYYY-MM-DD), e.g., 2026-01-31.

    Returns:
        OrderStockAnalysisResult: A structured analysis containing fulfillment booleans and exact reorder quantities.
    """
    logger.debug("FUNC (analyze_order_stock_requirements): Analyzing order.",
                 extra={"order_items": order_items, "as_of_date": as_of_date})

    try:
        # Fetch static data (like minimum stock levels) for all requested items in one go
        item_names = list(order_items.keys())
        inventory_items_data = get_inventory_items_by_name(item_names)

        # Create a quick lookup dictionary for static item data
        items_by_name = {item.item_name.lower(): item for item in (inventory_items_data or [])}

        analysis_list = []

        for item_name, requested_qty in order_items.items():
            item_name = item_name.lower()

            # 1. Get current stock at the requested date
            stock_snapshot = get_stock_level(item_name, as_of_date)
            # Handle cases where item has no transactions yet (None) or doesn't exist in dict
            stock_at_date = stock_snapshot.items.get(item_name, 0) if stock_snapshot else 0

            # 2. Get the minimum stock level
            static_item = items_by_name.get(item_name.lower())
            min_stock = static_item.min_stock_level if static_item else 0

            # 3. Calculate Fulfillment logic
            shortage = max(0, requested_qty - stock_at_date)
            supplier_delivery_date = None
            if shortage == 0:
                can_fulfill = True
            else:
                # We can fulfill if supplier covers shortage in time
                supplier_delivery_date = get_supplier_delivery_date(request_date, shortage)
                try:
                    date_supplier_delivery = date.fromisoformat(supplier_delivery_date)
                    date_request = date.fromisoformat(request_date)
                    can_fulfill = date_supplier_delivery <= date_request
                except Exception as e:
                    logger.error(
                        f"Error comparing request date ({request_date}) vs supplier delivery date ({supplier_delivery_date}).",
                        exc_info=True)
                    raise ValueError(
                        f"An error occurred while comparing request date and supplier delivery date. {str(e)}")

            # 4. Calculate Replenishment logic
            stock_after = stock_at_date - requested_qty
            drops_below_min = stock_after < min_stock

            analysis_list.append(
                ItemStockAnalysis(
                    item_name=item_name,
                    requested_quantity=requested_qty,
                    current_stock_at_date=stock_at_date,
                    min_stock_level=min_stock,
                    can_fulfill_order_item=can_fulfill,
                    shortage_for_order=shortage,
                    stock_after_fulfillment=stock_after,
                    drops_below_minimum=drops_below_min,
                    supplier_delivery_date=supplier_delivery_date
                )
            )

        return OrderStockAnalysisResult(
            as_of_date=as_of_date,
            can_fulfill_order=all(analysis.can_fulfill_order_item for analysis in analysis_list),
            items_analysis=analysis_list
        )

    except Exception as e:
        logger.error("Error analyzing order stock requirements.", exc_info=True)
        raise ValueError(f"An error occurred while analyzing stock requirements: {str(e)}")


class ItemShortageDetails(BaseModel):
    """Represents the shortage of units of an inventory item that must be covered to fulfill a customer order."""
    item_name: str = Field(description="Name of the catalog item.")
    unit_price: float = Field(description="Unit price of the item before discounts.")
    shortage_for_order: int = Field(
        description="Units missing strictly to fulfill the order (0 if can_fulfill_order is True).")
    supplier_delivery_date: str = Field(
        description="Date in ISO format (YYYY-MM-DD) by which the supplier can deliver missing units to fulfill the order.",
        examples=["2026-01-31"])


class OrderFromSupplierResult(BaseModel):
    """Result of ordering the necessary shortage units to cover a customer order."""
    messages: List[str] = Field(description="List of all answer messages in response to the given request.",
                                default_factory=list)
    transactions: List[FinancialTransaction] = Field(
        description="List of the 'stock_orders' transactions that were created to cover the shortage to fulfill the order",
        default_factory=list
    )


def order_shortage_from_supplier(items_to_order: list[ItemShortageDetails], total_base_amount: float,
                                 delivery_date: str) -> OrderFromSupplierResult:
    """
    Orders all the given shortage units from the supplier

    Args:
        items_to_order: List of ItemShortageDetails objects
        total_base_amount: Total amount of the order BEFORE discounts
        delivery_date: The requested delivery date of the Order, in ISO format (YYYY-MM-DD), e.g., 2026-01-31.

    Returns:
        An OrderFromSupplierResult, which contains all the financial transactions of 'stock_orders' type that were created
    """
    logger.debug("FUNC (order_shortage_from_supplier): Ordering shortage from supplier.",
                 extra={"items_to_order": items_to_order, "total_base_amount": total_base_amount,
                        "delivery_date": delivery_date})
    try:
        # Get latest delivery date and sure all are on or before the order delivery date
        dates = [date.fromisoformat(item.supplier_delivery_date) for item in items_to_order if
                 item.supplier_delivery_date]
        latest_delivery_date = max(dates) if dates else None

        if latest_delivery_date > date.fromisoformat(delivery_date):
            return OrderFromSupplierResult(
                messages=[
                    "Unable to fulfill order. Projected delivery dates to replenish stock shortages don't meet the order fulfillment date."]
            )

        # Compute resulting cash balance
        # NOTE: This might not be accurate but "acceptable" for the project.
        cash_balance = get_cash_balance(latest_delivery_date.isoformat())
        if cash_balance - total_base_amount < 0:
            return OrderFromSupplierResult(messages=["Unable to fulfill order. Projected cash balance is not enough"])

        # Place 'stock_order' transactions.
        # NOTE: This should be a DB transaction and rollback
        transactions = []
        for item in items_to_order:
            if item.shortage_for_order > 0:
                price = item.shortage_for_order * item.unit_price
                # create transaction, use order delivery date for simplicity
                transaction = create_transaction(item.item_name, 'stock_orders', item.shortage_for_order, price,
                                                 delivery_date)
                transactions.append(transaction)

        return OrderFromSupplierResult(transactions=transactions)

    except Exception as e:
        logger.error("Error ordering shortage from supplier.", exc_info=True)
        return OrderFromSupplierResult(
            messages=["I'm sorry, we encountered an error while ordering shortage units to cover your order"]
        )


class ReplenishToMinItemDetails(BaseModel):
    """Necessary data to replenish the stock level of the given inventory item up to its minimum stock level."""
    item_name: str = Field(description="Name of the catalog item.")
    unit_price: float = Field(description="Unit price of the item before discounts.")
    min_stock_level: int = Field(description="The minimum stock level required for this item.")
    can_fulfill_order_item: bool = Field(description="Whether this item can be fulfilled by the desired delivery date.")
    stock_after_fulfillment: int = Field(
        description="Projected stock after fulfilling the customer order (can be negative if order is unfulfillable without restocking).")


def replenish_to_minimum(items_to_order: List[ReplenishToMinItemDetails], as_of_date: str) -> OrderFromSupplierResult:
    """
    Replenishes each item in the list to bring its stock level up to its minimum at the given cutoff date.

    Args:
        items_to_order: List of ReplenishToMinItemDetails objects, which have the data of items to be replenished
        as_of_date (str): The requested delivery date of the Order, in ISO format (YYYY-MM-DD), e.g., 2026-01-31.

    Returns:
        An OrderFromSupplierResult, which contains all the financial transactions of 'stock_orders' type that were created
    """
    logger.debug("FUNC (replenish_to_minimum): Bringing up stock to minimum levels.",
                 extra={"items_to_order": items_to_order, "as_of_date": as_of_date})
    try:
        transactions = []
        for item in items_to_order:
            if item.can_fulfill_order_item:
                shortage = max(0, item.min_stock_level - item.stock_after_fulfillment)
                if shortage > 0:
                    price = shortage * item.unit_price
                    # NOTE: Using Order delivery date instead of supplier's one for simplicity
                    transaction = create_transaction(item.item_name, 'stock_orders', shortage, price, as_of_date)
                    transactions.append(transaction)

        return OrderFromSupplierResult(transactions=transactions)

    except Exception as e:
        logger.error("Error bringing up stock to minimum levels.", exc_info=True)
        return OrderFromSupplierResult(
            messages=["I'm sorry, we encountered an error while bringing up stock to minimum levels"])


#########################
# Tools for Quoting Agent
#########################


##########################
# Tools for Ordering Agent
##########################

##########
##########
# AGENTS #
##########
##########

# Set up and load your env parameters and instantiate your model.
load_dotenv()
assert os.getenv("OPENAI_BASE_URL")
assert os.getenv("OPENAI_API_KEY")

gpt_4o_mini = OpenAIChatModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
)


#############################
### Order Processor Agent ###
#############################

class RequestMetadata(BaseModel):
    """
    Represents metadata about a customer order request
    """
    model_config = ConfigDict(
        serialize_by_alias=True,  # Use aliases in JSON output
    )
    mood: str = Field(description="Likely mood or tone of the customer.", default="neutral",
                      examples=["happy", "angry", "sad", "miserable", "pissed off", "stressed", "neutral"],
                      alias="customer_mood")
    job: str = Field(description="Possible job or occupation of the customer.", default="unknown",
                     examples=["event manager", "restaurant manager", "director", "city hall clerk"], alias="job_type")
    need_size: str = Field(description="Qualitative size of the order.", default="medium",
                           examples=["small", "medium", "large"], alias="order_size")
    event: str = Field(description="Event for which the customer wants the order.", default="unknown",
                       examples=["unknown", "meeting", "festival", "concert", "ceremony", "party", "conference",
                                 "gathering"], alias="event_type")


class CustomerRequestDetails(BaseModel):
    """
    Represents the details of a customer request
    """
    delivery_date: str = Field(description="Expected delivery date for the order, in ISO format (YYY-MM-DD).",
                               examples=["2026-01-31"])
    request_date: str = Field(
        description="Date of request. This is the date on which the order is made, in ISO format (YYY-MM-DD).",
        examples=["2026-01-31"])
    items: Dict[str, int] = Field(description="Ordered items and quantities.", default_factory=dict)
    request_metadata: RequestMetadata = Field(description="Metadata about the customer request.", default=None)
    request_status: str = Field(description="Must be either 'ACCEPTED' or 'DECLINED'", default="DECLINED")
    messages: List[str] = Field(description="List of all answer messages in response to the given request.",
                                default_factory=list)


ORDER_PROCESSOR_SYSTEM_PROMPT = (
    "You are the Order Processor for the Beaver's Choice Paper Company\n."
    "Your job is to understand customer requests and determine:\n"
    "- Specific names of the items and their respective quantities\n"
    "- Expected or desired delivery date by the customer\n"
    "- Date of request\n"
    "- Inferred additional context like: mood and occupation of the customer, qualitative order size, event type\n"
    "\n"
    "### GUIDELINES\n"
    "Use the following guidelines to be able to understand and process a customer request:\n"
    "- The `get_all_item_names` tool returns all the catalog item names in the Company's inventory. Use them as examples and help you identify the items from the customer request.\n"
    "- When identifying items, extract only the canonical product name. Strip away all customer preferences, descriptive adjectives, or material specs (e.g., color, texture, biodegradable status, dimensions) unless they are essential to distinguishing the product in the inventory.\n"
    "- Numbers and units of measure such as sheets, reams, packs, boxes, rolls, etc., are part of the quantity, not separate items nor parts of the name.\n"
    "- If unsure whether an attribute is part of the product name or a customer preference, default to the most generic version of the product name.\n"
    "\n"
    "### EXAMPLES\n"
    "- Input: \"100 sheets of heavy cardstock (white)\" -> Item: \"heavy cardstock\", Quantity: 100\n"
    "- Input: \"20 sheets of colored paper (assorted colors)\" -> Item: \"colored paper\", Quantity: 20\n"
    "- Input: \"300 poster boards (24 x 36)\" -> Item: \"poster boards\", Quantity: 300\n"
    "- Input: \"500 paper plates (biodegradable)\" -> Item: \"paper plates\", Quantity: 500\n"
    "\n"
    "### STRICT RULES\n"
    "- You MUST NOT reject an order if one or more requested items are not present in the company's inventory, as long as they are relevant to a paper supplies company.\n"
    "- If the delivery date is not provided, use the current date.\n"
    "- You MUST DECLINE an order request if:\n"
    "   - You were not able to understand it.\n"
    "   - Date of request is not provided.\n"
    "   - The customer didn't specify any quantities for one or more items\n"
    "   - The customer is asking for items that are not relevant to a paper supplies company, e.g., firearms, sporting goods, food, etc.\n"
    "- In your `messages` array, clearly explain whether the request was accepted or declined, referencing the specific reasons that led to your decision.\n"
    "\n"
    "### OUTPUT\n"
    "- You MUST provide your final response in the structured format with all required fields.\n"
)


def new_order_processor_agent() -> Agent:
    order_processor_agent = Agent(
        gpt_4o_mini,
        name="Order Processor Agent",
        system_prompt=ORDER_PROCESSOR_SYSTEM_PROMPT,
        capabilities=[Thinking()],
        output_type=CustomerRequestDetails,
        tools=[
            Tool(get_all_item_names, docstring_format="google", require_parameter_descriptions=True),
            Tool(get_current_date, docstring_format="google", require_parameter_descriptions=True)
        ]
    )
    return order_processor_agent


#####################
### Quoting Agent ###
#####################

class QuoteCalculation(BaseModel):
    """
    Deterministic quote math for a single quoted item.
    """
    item_name: str = Field(description="Name of the quoted item")
    quantity: int = Field(description="Quantity quoted")
    unit_price: float = Field(description="Unit price before discounts")
    base_amount: float = Field(description="Quantity multiplied by unit price before discounts")
    discount_rate: float = Field(description="Applied discount rate as a decimal, 0.0 for no discount", default=0.0)
    discount_amount: float = Field(description="Dollar discount subtracted from the base amount, 0.0 for no discount",
                                   default=0.0)
    total_amount: float = Field(description="Final quoted amount after discounts")


class ItemQuote(BaseModel):
    """
    Represents the response generated by the Quoting Agent for a quote request of a single order item.
    """
    quote_calculation: QuoteCalculation = Field(
        description="Detailed field-by-field breakdown of the quote calculation.")
    quote_explanation: str = Field(
        description="Human-readable explanation of the quote, including base cost, discounts, and estimated delivery date.")
    messages: List[str] = Field(description="List of all answer messages in response to the given request.",
                                default_factory=list)


class OrderQuote(BaseModel):
    """
    Detailed generated quote for a customer order
    """
    item_quotes: List[ItemQuote] = Field(description="List of detailed quotes for each item in the order.",
                                         default_factory=list)
    total_base_amount: float = Field(description="Total amount of the order BEFORE discounts.", default=0.0)
    total_amount: float = Field(description="Total quoted amount of the order, AFTER discounts.", default=0.0)
    messages: List[str] = Field(description="List of answer messages in response to the given request.",
                                default_factory=list)


class SimpleQuoteData(BaseModel):
    """Essential quote details for a single item of an order, needed for inventory and sales processing."""
    item_name: str = Field(description="Name of the item ordered")
    quantity: int = Field(description="Quantity ordered")
    unit_price: float = Field(description="Unit price of the item")
    total_amount: float = Field(description="Total quoted price including discounts")


QUOTING_AGENT_SYSTEM_PROMPT = (
    "You are the Quoting Expert Agent for the Beaver's Choice Paper Company.\n"
    "Your specific responsibilities are:\n"
    "- Generate accurate and competitive quotes for customer paper supply requests.\n"
    "- Apply strategic bulk discounts to encourage larger orders.\n"
    "\n"
    "### Quote Generation Process\n"
    "Use the following steps to guide you towards responding to a quote request:\n"
    "1. Parse the request to identify the Item Name, requested Quantity, and Unit Price (optional).\n"
    "2. Make sure that you can process the request by checking the following requirements:\n"
    "   - The request MUST include ONLY one catalog item.\n"
    "   - The item quantity must be a positive integer number.\n"
    "2.1. If any of these requirements are not met, answer immediately indicating the relevant errors in the `messages` field of the response.\n"
    "3. Use the `search_quote_history` tool with only the item name as the search term to retrieve historical quotes for the item.\n"
    "4. Analyze the returned historical quotes to understand: unit price of the item, applied discounts, and explanations for similar requests. This data should inform your current quote generation to ensure competitiveness and consistency.\n"
    "5. Calculate the base cost (quantity × unit_price). Use the unit price given in the request if present, otherwise use the best match from your historical quote analysis."
    "5.1. If you are unable to do so, answer immediately indicating the relevant errors in the `messages` field of the response.\n"
    "6. Apply any discount you judge appropriate based on your analysis from the quote history, unless explicitly indicated not to do so.\n"
    "7. Generate a detailed quote calculation breakdown, which corresponds to the `quote_calculation` field of your response.\n"
    "8. Based on the quote calculation breakdown, construct a human-readable explanation of the quote, which corresponds to the `quote_explanation` field of the response."
    " Your explanation must include the base cost, discounts applied (if any), reason of the discount, and the estimated delivery date.\n"
    "8.1. Only say a discount was applied when the final total is less than the base cost.\n"
    "\n"
    "### Output Format Requirements\n"
    "- You MUST provide your final quote in the structured format with all required fields.\n"
)


def new_quoting_agent() -> Agent:
    quoting_agent = Agent(
        gpt_4o_mini,
        name="Quoting Agent",
        system_prompt=QUOTING_AGENT_SYSTEM_PROMPT,
        capabilities=[Thinking(effort='high')],
        output_type=ItemQuote,
        output_retries=3,
        tools=[
            Tool(search_quote_history, docstring_format="google", require_parameter_descriptions=True),
        ]
    )

    @quoting_agent.output_validator
    def _validate_quoted_discount(output: ItemQuote) -> ItemQuote:
        """
        Retry quotes that claim a discount but do not reduce the final amount.

        Uses the structured `quote_calculation` fields to validate:
        1. discount_rate and discount_amount are consistent with each other
        2. total_amount is actually lower than base_amount
        """
        quote_calc = output.quote_calculation
        if quote_calc is None:
            return output

        if quote_calc.discount_amount <= 0 and quote_calc.discount_rate <= 0:
            return output

        expected_base_amount = round(quote_calc.quantity * quote_calc.unit_price, 2)
        if abs(quote_calc.base_amount - expected_base_amount) > 0.01:
            raise ModelRetry(
                f"You calculated base_amount={quote_calc.base_amount}, "
                f"but the result of the operation quantity * unit_price is {expected_base_amount}. "
                f"Fix the base_amount to equal quantity * unit_price."
            )

        expected_discount = round(quote_calc.base_amount * quote_calc.discount_rate, 2)
        if abs(quote_calc.discount_amount - expected_discount) > 0.01:
            raise ModelRetry(
                f"The discount_rate={quote_calc.discount_rate} implies discount_amount={expected_discount}, "
                f"but discount_amount={quote_calc.discount_amount} in the quote_calculation. "
                "Fix the discount_amount to equal base_amount * discount_rate."
            )

        if quote_calc.total_amount >= quote_calc.base_amount:
            raise ModelRetry(
                f"You applied a discount of discount_rate={quote_calc.discount_rate}, "
                f"but total_amount ({quote_calc.total_amount}) is NOT lower than base_amount ({quote_calc.base_amount}). "
                "Keep your chosen discount, recalculate total_amount as base cost minus discount."
            )

        return output

    return quoting_agent


#######################
### Inventory Agent ###
#######################

class InventoryAgentOutput(BaseModel):
    """
    Represents the response to an inventory request generated by the Inventory Agent.
    """
    placed_transactions: List[FinancialTransaction] = Field(
        description="List of all 'stock_order' transactions that were successfully placed.",
        default_factory=list)
    messages: List[str] = Field(description="List of all answer messages in response to the given request.",
                                default_factory=list)


INVENTORY_AGENT_SYSTEM_PROMPT = (
    "You are the Inventory Management Agent for the Beaver's Choice Paper Company.\n"
    "Your specific responsibilities are:\n"
    "- Accurately answer queries about inventory levels.\n"
    "- Ensuring that customer orders can be fulfilled based on the inventory stock level forecast as of the order delivery date.\n"
    "- Proactively reordering supplies to replenish inventory and keep optimal stock levels.\n"
    "\n"
    "### Order Fulfillment Verification Process\n"
    "The following process will guide you when requested to verify if an order can be fulfilled:\n"
    "1. Identify if the following elements are available from the request:\n"
    "   - Delivery Date: Expected delivery date for the order by the customer.\n"
    "   - Order Request Date: Date on which the order was made by the customer.\n"
    "   - Quote data: Quote that was generated for each item of the order, which includes: item name, quantity, unit price, and total quoted price.\n"
    "1.1. If any of these elements are missing, answer immediately indicating the relevant errors in the `messages` field of the response.\n"
    "2. Call the `analyze_order_stock_requirements` tool to analyze if the order can be fulfilled in time.\n"
    "2.1. If the analysis yields that the order can not be fulfilled, answer immediately indicating which items can not be fulfilled in time.\n"
    "3. Call the `order_shortage_from_supplier` tool to order from the supplier the necessary shortage units to fulfill the order.\n"
    "3.1. Read all the messages from the `order_shortage_from_supplier` tool call response. If any errors occurred, answer immediately indicating the relevant errors in the `messages` field of your response.\n"
    "4. Call the `replenish_to_minimum` tool to bring any necessary item stock levels up to its minimum.\n"
    "4.1. Read all the messages from the `replenish_to_minimum` tool call response. If any errors occurred you can safely ignore them.\n"
    "\n"
    "### Output\n"
    "- Always provide your final answer in a structured format\n"
    "- The `placed_transactions` field of your response MUST CONTAIN all the transactions returned from the calls to the tools.\n"
)


def new_inventory_agent() -> Agent:
    """
    Creates a new Inventory Management Agent.
    """
    inventory_agent = Agent(
        gpt_4o_mini,
        name="Inventory Agent",
        system_prompt=INVENTORY_AGENT_SYSTEM_PROMPT,
        capabilities=[Thinking(effort='high')],
        output_type=InventoryAgentOutput,
        tools=[
            Tool(analyze_order_stock_requirements, docstring_format="google", require_parameter_descriptions=True),
            Tool(order_shortage_from_supplier, docstring_format='google', require_parameter_descriptions=True),
            Tool(replenish_to_minimum, docstring_format='google', require_parameter_descriptions=True),
            Tool(get_all_inventory, docstring_format='google', require_parameter_descriptions=True),
        ]
    )
    return inventory_agent


###################
### Sales Agent ###
###################

class SalesAgentOutput(BaseModel):
    """
    Represents the response to a sales request generated by the Sales & Ordering Agent.
    """
    request_status: str = Field(description="Must be either 'ACCEPTED' or 'DECLINED'")
    placed_transactions: List[FinancialTransaction] = Field(
        description="List of all 'sales' transactions successfully placed via the 'create_transaction' tool. Empty if the request was declined.",
        default_factory=list
    )
    messages: List[str] = Field(
        description="List of all answer messages in response to the given request.", default_factory=list
    )


SALES_AGENT_SYSTEM_PROMPT = (
    "You are the Sales and Ordering Agent for the Beaver's Choice Paper Company.\n"
    "Your specific responsibilities are:\n"
    "- Finalize sales transactions for customer orders.\n"
    "\n"
    "### Order Fulfillment Procedure\n"
    "The following procedure will guide you to finalize a sales transaction for an order item:\n"
    "1. Identify from the input: Item name, Ordered quantity, Quoted price, Delivery date.\n"
    "1.1. If you are unable to understand the request, decline it and answer immediately clearly explaining the problem in the `messages` field of your response.\n"
    "2. Using your tools to create a financial transaction for recording the sale of the item:\n"
    "   - The `transaction_type` MUST be 'sales'.\n"
    "   - The `transaction_date` MUST be the delivery date.\n"
    "2.1. If you are unable create the transaction due to missing data, or if the operation fails, decline the request and answer immediately clearly explaining the problem in the `messages` field of your response.\n"
    "\n"
    "### Output Requirements\n"
    "- You MUST provide your final response in the structured format with all required fields.\n"
)


def new_sales_agent():
    sales_agent = Agent(
        gpt_4o_mini,
        name="Sales Agent",
        system_prompt=SALES_AGENT_SYSTEM_PROMPT,
        capabilities=[Thinking(effort='high')],
        output_type=SalesAgentOutput,
        output_retries=3,
        tools=[
            Tool(create_transaction, docstring_format='google', require_parameter_descriptions=True)
        ]
    )

    return sales_agent


##########################
### Orchestrator Agent ###
##########################

@dataclass
class OrchestratorDependencies:
    order_processor_agent: Agent
    inventory_agent: Agent
    quoting_agent: Agent
    sales_agent: Agent


class OrchestratorAgentOutput(BaseModel):
    """
    Represents the response of the Orchestrator Agent.
    """
    order_status: str = Field(
        description="Must be either 'ACCEPTED' or 'DECLINED'. An order is 'DECLINED' if it's not possible to fulfill it due to any errors.")
    customer_response: str = Field(description=(
        "Human-readable response for the customer. Must be in plain text, informing him/her of the order status, total price, and delivery dates. "
        "It should also have confirmation about the quantities, quoted prices and applied discounts."))
    total_amount: float = Field(
        description="Grand total price of the order quoted to the customer, including discounts. Computed by the sum of all the individual quotes for each item in the order.")
    delivery_date: str = Field(description="Delivery date for the order, in ISO format.", examples=['2026-01-31'])
    request_metadata: RequestMetadata | None = Field(description="Metadata about the customer request.", default=None)


ORCHESTRATOR_AGENT_SYSTEM_PROMPT = (
    "You are the orchestrator for the Beaver's Choice Paper Company\n."
    "You coordinate between the Inventory Management Agent, the Quoting Agent and the Sales Agent "
    "to handle customer inquiries, checking inventory status, providing accurate quotations, and completing transactions seamlessly.\n"
    "For customer orders, follow this workflow:\n"
    "1. Use `process_order_info` to understand what the customer wants\n"
    "2. Check the `request_status` and `messages` of the response. If the request was declined, then answer immediately to the customer explaining the problem.\n"
    "3. Call the `generate_quote` tool to generate a quote for the order from the Quoting Agent.\n"
    "3.1. Read the messages from all the elements of the `generate_quote` response, if any errors occurred while trying to generate a quote, then answer immediately to the customer explaining the problem.\n"
    "4. Use the `manage_inventory` tool to reserve enough supplies for the order from the Inventory Management Agent.\n"
    "4.1. Read the messages from all the elements of the `manage_inventory` response, if any errors occurred while trying to manage inventory, then answer immediately to the customer explaining the problem.\n"
    "5. Call the `handle_sale` to finalize the sales transaction via the Sales Agent.\n"
    "5.1. Read all the messages from the response produced by the Sales Agent, if the request was declined or there are errors in the messages then answer immediately to the customer explaining the problem.\n"
    "\n"
    "### OUTPUT AND TONE:\n"
    "- Always be polite and friendly.\n"
    "- You MUST provide your final response in the structured format with all required fields.\n"
    "- NEVER mention replenishment, supplier orders, or projected replenishment dates in the response for the customer."
)


def _build_orchestrator_agent() -> Agent:
    orchestrator_agent = Agent(
        gpt_4o_mini,
        name="Orchestrator Agent",
        system_prompt=ORCHESTRATOR_AGENT_SYSTEM_PROMPT,
        capabilities=[Thinking()],
        deps_type=OrchestratorDependencies,
        output_type=OrchestratorAgentOutput,
        tools=[
            Tool(generate_financial_report, docstring_format='google', require_parameter_descriptions=True)
        ]
    )

    @orchestrator_agent.tool(docstring_format='google', require_parameter_descriptions=True)
    def process_order_info(ctx: RunContext[OrchestratorDependencies], customer_message: str) -> CustomerRequestDetails:
        """
        Process a customer request to extract all the necessary information and details to handle it

        Args:
            ctx: Context about the current call
            customer_message: The customer's request

        Returns:
            Processed order information
        """
        try:
            logger.info(
                "Processing customer order information.",
                extra={"customer_message": customer_message}
            )
            user_prompt = (
                "Process the following customer request to extract all the necessary information and details to handle it. This is the customer request:\n"
                f"{customer_message}"
            )
            response = ctx.deps.order_processor_agent.run_sync(user_prompt=user_prompt)
            logger.debug("Got response from order processor.", extra={"output": response.output})
            return response.output

        except Exception as e:
            logger.error(f"Error processing customer order.", exc_info=True,
                         extra={"customer_message": customer_message})
            return CustomerRequestDetails(
                messages=[
                    "I'm sorry, we encountered an error processing your order. Please try again later or contact customer service."],
                request_status="DECLINED",
                delivery_date="",
                request_date="",
                request_metadata=RequestMetadata()
            )

    @orchestrator_agent.tool(docstring_format='google', require_parameter_descriptions=True)
    def generate_quote(ctx: RunContext[OrchestratorDependencies], items: Dict[str, int]) -> OrderQuote:
        """
        Communicates with the Quoting Agent to generate quotes for each item of the customer request,
        and aggregates them in a single Order Quote.

        Args:
            ctx: Context about the current call
            items: A dictionary of the ordered items and their quantities extracted from the customer request.

        Returns:
            The quote for the customer order, containing detailed quotes for each item in the customer request.
        """
        try:
            logger.info("Getting quote for items.", extra={"items": items})
            item_quotes = []
            total_base = 0.0
            total_amount = 0.0
            for item_name, quantity in items.items():

                # NOTE: I tried to delegate this functionality to the agent,
                # but he became prone to exceeding the request limit of 50.
                # Probably a stronger model would work, but I was running low on credits.
                inventory_items = get_inventory_items_by_name([item_name])
                if inventory_items:
                    unit_price = inventory_items[0].unit_price
                else:
                    unit_price = None

                user_prompt = (
                    "Please give me a quote for the following item of a customer request:\n"
                    f"- Item name: {item_name}\n"
                    f"- Quantity: {quantity}\n"
                )
                if unit_price:
                    user_prompt = user_prompt + f"- Unit price: {unit_price}\n"

                response = ctx.deps.quoting_agent.run_sync(user_prompt=user_prompt)
                item_quote: ItemQuote = response.output
                logger.debug("Got response from quoting agent.", extra={"output": item_quote})

                item_quotes.append(item_quote)
                total_base = total_base + item_quote.quote_calculation.base_amount
                total_amount = total_amount + item_quote.quote_calculation.total_amount

            logger.debug("Got all item quotes from quoting agent.", extra={"item_quotes": item_quotes})
            return OrderQuote(
                item_quotes=item_quotes,
                total_base_amount=total_base,
                total_amount=total_amount
            )

        except Exception as e:
            logger.error("Error producing order quote.", exc_info=True, extra={"items": items})
            return OrderQuote(
                messages=[
                    "I'm sorry, we encountered an error generating a quote. Please try again later or contact customer service."]
            )

    @orchestrator_agent.tool(docstring_format='google', require_parameter_descriptions=True)
    def manage_inventory(ctx: RunContext[OrchestratorDependencies], total_base_amount: float, total_amount: float,
                         quotes_data: List[SimpleQuoteData],
                         delivery_date: str, request_date: str) -> InventoryAgentOutput:
        """
        Communicate with the Inventory Agent to manage inventory for an order and replenish supplies if needed

        Args:
            ctx: Context about the current call
            total_base_amount: Total computed amount of the order BEFORE discounts
            total_amount: Total quoted amount of the order, AFTER discounts
            quotes_data: Essential details extracted from the generated quotes.
            delivery_date: The desired delivery date by the customer, in ISO format (YYYY-MM-DD), e.g., '2026-01-31'.
            request_date: Date on which the order is made, in ISO format (YYYY-MM-DD), e.g., '2026-01-31'.

        Returns:
            The response that the Inventory Agent generated.
        """
        try:
            adapter = TypeAdapter(list[SimpleQuoteData])
            quotes_json = adapter.dump_json(quotes_data).decode()

            logger.info(
                f"Managing inventory for order.",
                extra={"quotes_data": quotes_json,
                       "total_base_amount": total_base_amount,
                       "total_amount": total_amount,
                       "delivery_date": delivery_date,
                       "order_date": request_date}
            )
            user_prompt = (
                "A customer has placed an order and we have generated the quote below. "
                "Please ensure the order can be fulfilled and replenish the inventory as needed following your rules.\n"
                f"- Desired delivery date: {delivery_date}\n"
                f"- Order Request Date: {request_date}\n"
                f"- Total base amount: {total_base_amount}\n"
                f"- Total quoted amount: {total_amount}\n"
                f"- Individual item quotes (in JSON format):\n"
                f"{quotes_json}"
            )
            response = ctx.deps.inventory_agent.run_sync(user_prompt=user_prompt)
            logger.debug("Got response from inventory agent.", extra={"output": response.output})
            return response.output

        except Exception as e:
            logger.error(f"Error managing inventory.",
                         exc_info=True,
                         extra={"quotes_data": quotes_data, "delivery_date": delivery_date,
                                "order_date": request_date})
            return InventoryAgentOutput(
                messages=[
                    "I'm sorry, we encountered an error managing the inventory for your order. Please try again later or contact customer service."],
            )

    @orchestrator_agent.tool(docstring_format='google', require_parameter_descriptions=True)
    def handle_sale(ctx: RunContext[OrchestratorDependencies], quotes_data: List[SimpleQuoteData],
                    delivery_date: str, request_date: str) -> List[SalesAgentOutput]:
        """
        Communicate with the Sales agent to finalize a sales transaction for an order.

        Args:
            ctx: Context about the current call
            quotes_data: Essential details extracted from the generated quotes for the order.
            delivery_date: The desired delivery date by the customer, in ISO format (YYYY-MM-DD), e.g., '2026-01-31'.
            request_date: Date on which the order is made, in ISO format (YYYY-MM-DD), e.g., '2026-01-31'.

        Returns:
            A list of SalesAgentOutput objects generated by the Sales Agent.
            Each element corresponds to a sales operation for a single quoted item of the order.
        """
        try:
            logger.info(
                "Finalizing a sales transaction for the order.",
                extra={
                    "quotes_data": [q.model_dump() for q in quotes_data],
                    "delivery_date": delivery_date,
                    "order_date": request_date
                }
            )
            sales_responses = []

            # NOTE: This is not ideal: the creation of the transactions is not done transactionally :)
            for quote in quotes_data:
                user_prompt = (
                    "The customer wants to finalize an order. Record the sale transaction for the following item:\n"
                    f"- Item name: {quote.item_name}\n"
                    f"- Ordered quantity: {quote.quantity} units\n"
                    f"- Quoted total price ${quote.total_amount:.2f}.\n"
                    f"- Delivery date: {delivery_date}."
                )
                response = ctx.deps.sales_agent.run_sync(user_prompt=user_prompt)
                sales_responses.append(response.output)

            logger.debug("Got response from sales agent.", extra={"output": sales_responses})
            return sales_responses

        except Exception as e:
            logger.error(f"Error finalizing a sales transaction for an order.",
                         exc_info=True,
                         extra={
                             "quotes_data": quotes_data,
                             "delivery_date": delivery_date,
                             "order_date": request_date
                         })
            return [SalesAgentOutput(
                messages=[
                    "I'm sorry, we encountered an error finalizing a sales transaction for your order. Please try again later or contact customer service."],
                request_status="DECLINED"
            )]

    return orchestrator_agent


class OrchestratorAgent:

    def __init__(self):
        self.order_processor_agent = new_order_processor_agent()
        self.inventory_agent = new_inventory_agent()
        self.quoting_agent = new_quoting_agent()
        self.sales_agent = new_sales_agent()
        self.orchestrator_dependencies = OrchestratorDependencies(
            self.order_processor_agent,
            self.inventory_agent,
            self.quoting_agent,
            self.sales_agent,
        )

        self.orchestrator_agent = _build_orchestrator_agent()

    def process_customer_order(self, customer_message) -> AgentRunResult[OrchestratorAgentOutput]:
        """
        Process a customer order through the coordinated agent workflow
        :param customer_message: The customer's order request
        :return: A human-readable response to the customer
        """
        logger.info("Processing customer order through orchestrator agent.",
                    extra={"customer_message": customer_message})
        response = self.orchestrator_agent.run_sync(user_prompt=customer_message,
                                                    deps=self.orchestrator_dependencies)
        return response


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    logger.info("Initializing Database...")
    init_database(DB_ENGINE)
    try:
        quote_requests_sample = pd.read_csv(os.path.join(_DATA_DIR, "quote_requests_sample.csv"))
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample["delivery_date"] = pd.to_datetime(
            quote_requests_sample["delivery_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["delivery_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        logger.error("FATAL: Error loading test data", exc_info=True)
        return None

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report.cash_balance
    current_inventory = report.inventory_value

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    orchestrator = OrchestratorAgent()

    judge = LLMJudge(
        model=gpt_4o_mini,
        include_input=True,
        assertion=False,
        score={'include_reason': True},
        model_settings=ModelSettings(temperature=0.0),
        rubric=(
            "Verify that the agent's prose response to the customer is accurate and complete.\n"
            "If the order is ACCEPTED: The prose MUST explicitly mention the total price and delivery date.\n"
            "If the order is DECLINED: The prose MUST provide a professional explanation for the rejection.\n"
            "The prose must not contradict the structured JSON data (price/dates)."
        )
    )
    ############
    ############
    ############

    results = []
    for index, row in quote_requests_sample.iterrows():
        idx = cast(int, index)
        request_date = row["request_date"].strftime("%Y-%m-%d")
        delivery_date = row["delivery_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx + 1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Delivery Date: {delivery_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        result = orchestrator.process_customer_order(request_with_date)
        orchestrator_agent_output: OrchestratorAgentOutput = result.output
        judge_ctx = EvaluatorContext(
            name=None,
            inputs=request_with_date,
            output=orchestrator_agent_output,
            expected_output=None,
            metadata=result.metadata,
            duration=getattr(result, "duration", 0.0),
            _span_tree=SpanTree(),
            attributes={},
            metrics={},
        )
        judge_report = judge.evaluate_sync(judge_ctx)
        evaluation_reason: EvaluationReason = judge_report.get("LLMJudge")  # TODO: is there a better way to get this?

        # Update state
        # NOTE: I changed the date in the line below from request_date to delivery_date
        # because my whole implementation hinges on the EXPECTED/PROJECTED state as of delivery dates of the orders,
        # not on the request date.
        report = generate_financial_report(delivery_date)
        current_cash = report.cash_balance
        current_inventory = report.inventory_value

        print(f"Response: {orchestrator_agent_output.model_dump_json(indent=2)}")
        print(f"Judge Report: {json.dumps(asdict(evaluation_reason), indent=2)}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")
        print(f"==========================\n\n")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "delivery_date": delivery_date, # NOTE: I added this
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": orchestrator_agent_output.customer_response,
                "judge_score": evaluation_reason.value, # NOTE: I added this
                "judge_reason": evaluation_reason.reason # NOTE: I added this
            }
        )

        time.sleep(1)

    # Final report
    # NOTE: I changed the date in the line below because my whole implementation hinges on the PROJECTED state
    # as of delivery dates of the orders, not on the "request date".
    final_date = quote_requests_sample["delivery_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report.cash_balance:.2f}")
    print(f"Final Inventory: ${final_report.inventory_value:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
