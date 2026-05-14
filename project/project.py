import ast
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pythonjsonlogger.json import JsonFormatter
from sqlalchemy import create_engine, Engine, bindparam
from sqlalchemy.sql import text

# Configure logging
logger = logging.getLogger("project")
logHandler = logging.StreamHandler()
formatter = JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
logHandler.setFormatter(formatter)
logging.getLogger().addHandler(logHandler)


def _load_log_levels() -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f).get("log_levels", {})


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


###############
# MODEL CLASSES
###############

class InventorySnapshot(BaseModel):
    """
    Snapshot of calculated stock levels of an item or items as of a specific date.
    """
    model_config = ConfigDict(from_attributes=True)
    items: Dict[str, int] = Field(
        description=(
            "Dictionary holding the stock levels of each item, where: "
            "The key is the name of the item (must be a string). "
            "The value is the calculated quantity or stock level for that item (must be an integer)."
        ),
        default_factory=dict)


class FactoryState(BaseModel):
    """
    Represents the shared state across agents in the Munder Difflin system.
    """
    min_cash_threshold: float = Field(default=5000.0, description="Minimum Cash Balance that MUST be maintained.")


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


# Module-level shared state instance — all agents read from and write to this singleton
_factory_state = FactoryState()


class FinancialTransaction(BaseModel):
    """
    A financial transaction. Can be a stock order or a sale.
    """
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(description="Transaction ID (uuid7)")
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
    as_of_date: str = Field(description="Date in ISO format up to which the report is generated")
    cash_balance: float = Field(description="Current cash balance")
    inventory_value: float = Field(description="Total value of inventory")
    total_assets: float = Field(description="Combined cash and inventory value")
    inventory_summary: List[InventoryItem] = Field(description="Summary of the inventory")
    top_selling_products: List[TopSellingProduct] = Field(description="Top 5 products by revenue")


class QuoteCalculation(BaseModel):
    """
    Deterministic quote math for a single quoted item.
    """
    item_name: str = Field(description="Name of the quoted inventory item")
    quantity: int = Field(description="Quantity quoted")
    unit_price: float = Field(description="Unit price before discounts")
    base_amount: float = Field(description="Quantity multiplied by unit price before discounts")
    discount_rate: float = Field(description="Applied discount rate as a decimal, 0.0 for no discount", default=0.0)
    discount_amount: float = Field(description="Dollar discount subtracted from the base amount, 0.0 for no discount",
                                   default=0.0)
    total_amount: float = Field(description="Final quoted amount after discounts")


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
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
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
    return InventorySnapshot(items=items)


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
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
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
                      ORDER BY total_revenue DESC LIMIT 5 \
                      """
    top_sales = pd.read_sql(top_sales_query, DB_ENGINE, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    report = {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
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
        list = [dict(row._mapping) for row in result]
        logger.debug(f"FUNC (search_quote_history): Returning quote history. Found {len(list)} records.",
                     extra={"search_terms": search_terms, "limit": limit})
        return list


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


#########################
# SHARED STATE MANAGEMENT
#########################

def get_minimum_cash_balance() -> float:
    """
    Checks the Minimum Cash Balance threshold that must be kept for the Factory.

    Returns:
        Minimum Cash Balance that MUST be maintained.
    """
    return _factory_state.min_cash_threshold


def get_current_date() -> str:
    """
    Gets the current date in ISO format, e.g. "2026-01-31".

    Returns:
         The current date in ISO format.
    """
    return date.today().isoformat()


###########################
# Tools for Inventory Agent
###########################
def create_transaction(
        item_name: str,
        transaction_type: str,
        quantity: int,
        price: float,
        date: str | datetime,
) -> FinancialTransaction:
    """
    This function records a financial transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        FinancialTransaction: The transaction that was created.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    logger.debug(
        "FUNC (create_transaction): Creating transaction.",
        extra={"item_name": item_name, "transaction_type": transaction_type, "quantity": quantity, "price": price,
               "date": date}
    )
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

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
                  SELECT item_name,
                         COALESCE(SUM(CASE
                                          WHEN transaction_type = 'stock_orders' THEN units
                                          WHEN transaction_type = 'sales' THEN -units
                                          ELSE 0
                             END), 0) AS stock_level
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
                inventory_snapshot = InventorySnapshot(items={row[0]: row[1]})
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
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
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


def get_inventory_items_by_name(item_names: list[str]) -> list[InventoryItem] | None:
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
                return None

            adapter = TypeAdapter(List[InventoryItem])
            inventory_items = adapter.validate_python(list(rows))
            return inventory_items

    except Exception as e:
        logger.error("Error getting inventory items.", exc_info=True)
        raise


def get_items_below_min_stock_level(stock_levels: InventorySnapshot) -> InventorySnapshot | None:
    """
    Identify items that are at or below their minimum stock level from the provided inventory/stock snapshot.

    Args:
        stock_levels (InventorySnapshot): A snapshot of item stock levels.

    Returns:
        InventorySnapshot: A snapshot of items that require replenishment. Returns None if no items are below threshold or not found.
    """
    logger.debug(
        "FUNC (get_items_below_min_stock_level): Getting items below min stock level.",
        extra={"stock_levels": stock_levels.items}
    )

    item_names = list(stock_levels.items.keys())
    inventory_items = get_inventory_items_by_name(item_names)

    if inventory_items is None:
        return None

    # Filter items where calculated stock is <= min_stock_level
    below_min_items = {}
    for item in inventory_items:
        # Get the calculated stock level from the snapshot
        calculated_stock = stock_levels.items.get(item.item_name)

        # If item found in snapshot and stock is below threshold
        if calculated_stock is not None and calculated_stock <= item.min_stock_level:
            below_min_items[item.item_name] = calculated_stock

    return InventorySnapshot(items=below_min_items) if below_min_items else None


def update_inventory_item_stock(item_name: str, current_stock: int) -> InventoryItem:
    """
    Update the current stock quantity of the inventory item in the database.

    Args:
        inventory_item: The inventory item to update.
    """
    try:
        with DB_ENGINE.begin() as conn:
            query = text(
                "UPDATE inventory SET current_stock = :current_stock WHERE LOWER(item_name) = LOWER(:item_name) RETURNING *")
            params = dict(item_name=item_name, current_stock=current_stock)
            result = conn.execute(query, params)
            return InventoryItem.model_validate(result.fetchone())

    except Exception as e:
        logger.error("Error updating inventory item", exc_info=True)
        raise


#########################
# Tools for Quoting Agent
#########################

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


##########################
# Tools for Ordering Agent
##########################


##########
# AGENTS #
##########

# Set up and load your env parameters and instantiate your model.
load_dotenv()
assert os.getenv("OPENAI_BASE_URL")
assert os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

gpt_4o_mini = OpenAIChatModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(openai_client=client)
)
gpt_4o = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(openai_client=client)
)


class InventoryAgentOutput(BaseModel):
    """
    Represents the response to an inventory request generated by the Inventory Agent.
    """
    inventory_items: List[InventoryItem] = Field(
        description="List of all queried inventory items for static data retrieval", default_factory=list)
    calculated_stock_levels: InventorySnapshot = Field(
        description="Object holding a dictionary of items and their stock levels that were computed as of the request date.",
        default=None)
    placed_transactions: List[FinancialTransaction] = Field(
        description="List of all 'stock_order' transactions that were SUCCESSFULLY placed via the 'create_transaction' tool. MUST be empty if no tool calls were made.",
        default_factory=list)
    messages: List[str] = Field(description="List of all answer messages in response to the given request.",
                                default_factory=list)


INVENTORY_AGENT_SYSTEM_PROMPT = (
    "You are the Inventory Management Agent for the Beaver's Choice Paper Company.\n"
    "Your specific responsibilities are:\n"
    "- Accurately answer queries about inventory levels.\n"
    "- Reordering supplies to replenish inventory and keep optimal stock levels.\n"
    "\n"
    "### Rules for Answering Inventory Queries\n"
    "- You have tools to get the stock level of a single item, or get a snapshot of the whole inventory at a cutoff date.\n"
    "- Use the requested delivery date, NOT the order date to get such stock levels. If the delivery date is not provided, then use the current date.\n"
    "- Keep track of the inventory stock levels and add them in the `calculated_stock_levels` field of the response object.\n"
    "\n"
    "### Rules for Reordering Supplies\n"
    "Reordering or replenishing supplies for an inventory item is done by creating a financial transaction, the following procedure will guide you to do so.\n"
    "For each item in the customer order, do the following:\n"
    "1. Will stock level be enough to satisfy the requested quantity?"
    "   - Get the stock level for the item with the requested delivery date as cutoff date.\n"
    "   - If the item is not present in the inventory, assume that the stock level is 0\n"
    "   - Identify if the sock level in inventory will be enough to satisfy the requested quantities in the customer order.\n"
    "   - If the stock level won't be enough to satisfy the requested quantity, proceed to step 2, then step 3\n"
    "   - If the stock level will be enough to satisfy the requested quantity, proceed to step 3 only\n"
    "2. Replenishing stock level to be able to fulfill the order (placing a `stock_orders` transaction):\n"
    "   - Calculate the missing quantity to fulfill the order\n"
    "   - Using your tools, get the estimated delivery date from the supplier for the missing quantity. Use the 'order date' given to you in the request for this operation; if such date was not provided then use the current date.\n"
    "   - If the supplier estimated delivery date is ON or BEFORE the requested order delivery date we know we can fulfill this order item. If it is not, then answer immediately to the customer explaining the problem.\n"
    "   - If given a unit price for the item, use it to compute the transaction price. Otherwise get the unit price from your tools. The transaction price by multiplying the order quantity by the item's unit price\n"
    "   - Use your tools to get the Company's Minimum Cash Balance threshold and the cash balance as of the supplier delivery date\n"
    "   - Compute the resulting Cash Balance of placing the transaction: (cash balance as of supplier delivery date - transaction price)\n."
    "   - Identify if the resulting cash balance will drop below the Minimum Cash Balance threshold. If it will, then answer immediately to the customer explaining the problem.\n"
    "   - Otherwise, call `create_transaction` to replenish the stock level with the missing quantity. The type MUST be `stock_orders` and the transaction date MUST be the estimated supplier delivery date. This ensures the inventory system correctly hides pending inventory until it physically arrives.\n"
    "   - Keep track of the transaction object that was created in the `placed_transactions` field of your response.\n"
    "3. Keeping the item stock level above it's minimum:\n"
    "   - Get the minimum stock level for the item using your tools (this is static data of the item)\n"
    "   - Calculate the new stock level for the item AFTER the order is fulfilled (stock level at requested delivery date - order item quantity)\n"
    "   - Identify if the new stock level for the item will drop below its minimum level. If it won't, then there's no need to replenish the stock levels for the item\n"
    "   - Otherwise, follow the exact same procedure as Step 2 to replenish the stock level for the item, but you must calculate the exact replenishment quantity using this formula: `replenishment_quantity = minimum_stock_level - new_stock_level`.\n"
    "   - Use this `replenishment_quantity` when calling `create_transaction`. NEVER order a negative quantity.\n"
    "\n"
    "### Output\n"
    "- Always provide your final answer in a structured format\n"
)


def new_inventory_agent() -> Agent:
    """
    Creates a new Inventory Management Agent.
    """
    inventory_agent = Agent(
        gpt_4o_mini,
        name="Inventory Agent",
        system_prompt=INVENTORY_AGENT_SYSTEM_PROMPT,
        capabilities=[Thinking()],
        output_type=InventoryAgentOutput,
        tools=[
            get_minimum_cash_balance,  # to check whether before putting a "stock_orders" transaction
            get_inventory_items_by_name,  # to get unit_price
            get_stock_level,
            get_all_inventory,
            get_supplier_delivery_date,
            create_transaction,
            get_current_date,
            get_cash_balance
        ]
    )
    return inventory_agent


class QuotingAgentOutput(BaseModel):
    """
    Represents the response to a quote request generated by the Quoting Agent.
    """
    quote_calculation: QuoteCalculation = Field(
        description="Detailed field-by-field breakdown of the quote calculation.")
    quote_explanation: str = Field(
        description="Human-readable explanation of the quote, including base cost, discounts, and estimated delivery date.")
    estimated_delivery_date: str = Field(description="Estimated delivery date for the order in ISO format")
    messages: List[str] = Field(description="List of all answer messages in response to the given request.",
                                default_factory=list)


class SimpleQuoteData(BaseModel):
    """Essential quote details needed for inventory and sales processing."""
    item_name: str = Field(description="Name of the catalog item")
    quantity: int = Field(description="Quantity ordered")
    unit_price: float = Field(description="Unit price of the item")
    total_amount: float = Field(description="Total quoted price including discounts")


ERROR_QUOTE_CALCULATION = QuoteCalculation(
    item_name="",
    quantity=-1,
    unit_price=-1,
    base_amount=-1,
    discount_rate=-1,
    discount_amount=-1,
    total_amount=-1,
)

QUOTING_AGENT_SYSTEM_PROMPT = (
    "You are the Quoting Expert Agent for the Beaver's Choice Paper Company.\n"
    "Your specific responsibilities are:\n"
    "- Generate accurate and competitive quotes for customer paper supply requests.\n"
    "- Apply strategic bulk discounts to encourage larger orders.\n"
    "- Provide clear delivery estimates for quoted items.\n"
    "\n"
    "### Quote Generation Process\n"
    "Use the following steps to guide you towards responding to a quote request:\n"
    "1. Parse the request to identify the following elements: Item name, Quantity, Desired delivery date, Order date.\n"
    "   - An item is a catalog product name only. Numbers and units of measure such as sheets, reams, packs, boxes, rolls, etc., are part of the quantity, not separate items.\n"
    "   - Pricing instructions such as 'do not apply discounts' or 'apply a bulk discount' are not item names and must not make an otherwise valid one-item request invalid.\n"
    "   - Example: '1000 sheets of A4 paper' is exactly one item: item_name='A4 paper', quantity=1000.\n"
    "   - Example: '100 sheets of A4 paper. DO NOT apply any discounts.' is exactly one item: item_name='A4 paper', quantity=100, apply_discount=false.\n"
    "   - Example: 'A4 paper and cardstock' is two items and must be rejected.\n"
    "2. Make sure that you can process the request by checking the following requirements:\n"
    "   - The request MUST include ONLY one catalog item.\n"
    "   - The item quantity must be a positive integer number.\n"
    "2.1. If any of these requirements are not met, answer immediately indicating the relevant errors in the `messages` field of the response.\n"
    "3. If the order date is not given, use today's date as the order date.\n"
    "4. If desired delivery date is not given, estimate the delivery date with `get_supplier_delivery_date` using the order date and quantity.\n"
    "5. Use the `search_quote_history` tool with only the item name as the search term. Analyze the returned historical quotes to understand: unit price, discounts, and explanations for similar requests. This information should inform your current quote generation to ensure competitiveness and consistency.\n"
    "6. If you are unable to determine the item unit price from the quote history search, use the `get_inventory_items_by_name` tool to get it. "
    "If you find multiple potential matches, use the most expensive one to break the tie, e.g., 'A4 glossy paper' can match both 'Glossy paper' and 'A4 paper', but Glossy is more expensive, so use this one.\n."
    "7. Calculate the base cost (quantity × unit_price). If you are unable to do so, answer immediately indicating the relevant errors in the `messages` field of the response.\n"
    "8. Apply any discount you judge appropriate based on your analysis from the quote history, unless explicitly indicated not to do so.\n"
    "9. Generate a detailed quote calculation breakdown, which corresponds to the `quote_calculation` field of your response. This includes:\n"
    "   - Item name\n"
    "   - Quantity\n"
    "   - Unit price\n"
    "   - Base amount (quanity * unit price)\n"
    "   - Discount rate (percentage) if applicable\n"
    "   - Discount amount (dollar value) if applicable\n"
    "   - Total amount (final total after discounts)\n"
    "10. Based on the quote calculation breakdown, construct a human-readable explanation of the quote, which corresponds to the `quote_explanation` field of the response."
    " Your explanation must include the base cost, discounts applied (if any), reason of the discount, and the estimated delivery date.\n"
    "10.1. Only say a discount was applied when the final total is less than the base cost.\n"
    "\n"
    "### Output Format Requirements\n"
    "- You MUST provide your final quote in the structured format with all required fields.\n"
)


def new_quoting_agent() -> Agent:
    quoting_agent = Agent(
        gpt_4o,
        name="Quoting Agent",
        system_prompt=QUOTING_AGENT_SYSTEM_PROMPT,
        capabilities=[Thinking(effort='high')],
        output_type=QuotingAgentOutput,
        output_retries=3,
        tools=[
            get_all_item_names,  # to match item names against quote request
            search_quote_history,  # get past quotes
            get_inventory_items_by_name,  # to get unit_price
            get_supplier_delivery_date,  # estimates delivery based on quantity
            get_current_date,
        ]
    )

    @quoting_agent.output_validator
    def _validate_quoted_discount(output: QuotingAgentOutput) -> QuotingAgentOutput:
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


class SalesAgentOutput(BaseModel):
    """
    Represents the response to a sales request generated by the Sales & Ordering Agent.
    """
    request_status: str = Field(description="Must be either 'ACCEPTED' or 'DECLINED'")
    placed_transactions: List[FinancialTransaction] = Field(
        description="List of all 'sales' transactions successfully placed via the 'create_transaction' tool. Empty if declined.",
        default_factory=list
    )
    messages: List[str] = Field(
        description="List of all answer messages in response to the given request, explaining the fulfillment decision.",
        default_factory=list
    )


SALES_AGENT_SYSTEM_PROMPT = (
    "You are the Sales and Ordering Agent for the Beaver's Choice Paper Company.\n"
    "Your specific responsibilities are:\n"
    "- Finalize sales transactions for customer orders.\n"
    "- Strictly verify that inventory levels and supplier delivery timelines can meet the customer's requested date before accepting the request.\n"
    "\n"
    "### Order Fulfillment Rules\n"
    "Before creating any sales transaction, you MUST perform the following checks:\n"
    "1. Identify from the user input / request: the item name, exact quantity, the desired delivery date, and the order date. Here are some guidelines to help you:\n"
    "   - The 'order date' is the date the order is placed (the start date for supplier lead times).\n"
    "   - The 'desired delivery date' is when the customer wants to receive the order.\n"
    "   - An item is a catalog product name only. Numbers and units of measure such as sheets, reams, packs, boxes, rolls, etc., are part of the quantity, not separate items.\n"
    "   - Use your tools to verify the exact catalog item name.\n"
    "   - If the item name has many potential matches, break the tie using the most expensive one., "
    " e.g., a request for 'A4 glossy paper' can match both 'Glossy paper' and 'A4 paper', but Glossy is more expensive, so use this one.\n"
    "   - If no desired delivery date is provided, use the current date as delivery date.\n"
    "   - If no order date is provided, use the current date as order date.\n"
    "1.1. If you are unable to understand the request, or if the desired delivery date is BEFORE the order date, immediately answer indicating the relevant errors in the `messages` field of the response.\n"
    "2. Get the stock level for the requested item on the desired delivery date using your tools.\n"
    "3. Determine fulfillment capability:\n"
    "   - Scenario A (In Stock): If stock level >= requested quantity, you can fulfill the order. Proceed to create the sales transaction.\n"
    "   - If stock level < requested quantity, compute the quantity needed to fulfill the order, then call get_supplier_delivery_date(order_date, quantity_needed) to get the date when the supplier can deliver replenishment stock.\n"
    "   - Scenario B (Short Stock, Fast Delivery): If the supplier delivery date is ON or BEFORE the customer's requested delivery date, you can fulfill the order. Proceed to create the sales transaction.\n"
    "   - Scenario C (Cannot Fulfill): If the supplier delivery date is AFTER the customer's requested delivery date, you MUST decline the order. DO NOT create a transaction.\n"
    "4. When creating a transaction:\n"
    "   - The `transaction_type` MUST be `sales`.\n"
    "   - If the given input includes a previously quoted total price (e.g. 'quote for N units at $X'), use that exact $X as the `price`. Do NOT recalculate or override it."
    " Only calculate (quantity * unit_price) when no quoted price is given.\n"
    "   - The `transaction_date` MUST equal the desired delivery date.\n"
    "5. In your `messages` array, clearly explain whether the order was accepted or declined, referencing the specific dates, stock levels, and timelines that led to your decision.\n"
    "\n"
    "### Output Format Requirements\n"
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
            get_all_item_names,  # To match item names against quote request
            get_inventory_items_by_name,  # To get catalog names and unit prices
            get_stock_level,  # To verify inventory
            get_supplier_delivery_date,  # To check timelines if stock is short
            create_transaction,  # To finalize the sale
            get_current_date
        ]
    )

    return sales_agent


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
    delivery_date: str = Field(description="Expected delivery date for the order, in ISO format.",
                               examples=["2026-01-31"])
    request_date: str = Field(description="Date on which the order is made, in ISO format.",
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
    "- Expected or desired delivery date by the customer"
    "- Date of request"
    "- Inferred additional context like: mood of the customer, his/her occupation or job, event type\n"
    "\n"
    "### CONSTRAINTS:\n"
    "- You DO NOT care about inventory availability (quantities), you only care about understanding the customer's request and breaking it down into useful pieces and data.\n"
)

ORDER_PROCESSOR_INSTRUCTIONS = (
    "Identify the following elements from the customer request:\n"
    "- Item names and respective quantities\n"
    "- Delivery date: Date on which the customer wants the order to be delivered\n"
    "- Request date: Date on which the order is made\n"
    "Use the following guidelines to achieve your task:\n"
    "  - If the desired delivery date is not provided by the customer, use the current date as delivery date.\n"
    "  - If the request date is not provided in the input, use the current date as delivery date.\n"
    "  - The `get_all_item_names` tool contains all the catalog item names in the Company, use it to get examples of item names.\n"
    "  - An item is a catalog product name only. Numbers and units of measure such as sheets, reams, packs, boxes, rolls, etc., are part of the quantity, not separate items.\n"
    "\n"
    "### OUTPUT\n"
    "- You MUST provide your final response in the structured format with all required fields.\n"
    "- All customer orders must be either ACCEPTED or REJECTED"
    "- You MUST NOT reject an order if one or more requested items are not present in the company's inventory, as long as they are relevant to a paper supplies company\n"
    "- The possible reasons for you to decline a request are:\n"
    "   - You were not able to understand it\n"
    "   - The customer didn't specify any quantities for one or more items\n"
    "   - The customer is asking for items that are not relevant to a paper supplies company, e.g. firearms, sporting goods, food, etc.\n"
    "- In your `messages` array, clearly explain whether the request was accepted or declined, referencing the specific reasons that led to your decision.\n"
)


def new_order_processor_agent() -> Agent:
    order_processor_agent: Agent = Agent(
        gpt_4o_mini,
        name="Order Processor Agent",
        system_prompt=ORDER_PROCESSOR_SYSTEM_PROMPT,
        capabilities=[Thinking()],
        output_type=CustomerRequestDetails,
        tools=[
            get_current_date,
            get_all_item_names
        ]
    )
    return order_processor_agent


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
    delivery_date: str | None = Field(description="Delivery date for the order, in ISO format.", default=None)
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
    "4. Call the `get_latest_delivery_date` date tool to get the possible latest delivery date from the individual quotes.\n"
    "4.1. If the latest delivery date is before or at the customer's requested delivery date, then the order can be fulfilled and you can proceed. Otherwise answer to the customer explaining why the order cannot be fulfilled.\n"
    "5. Use the `manage_inventory` tool to proactively replenish supplies via the Inventory Management Agent. Provide the latest delivery date that you got on a previous step.\n"
    "6. Call the `handle_sale` to finalize the sales transaction via the Sales Agent.\n"
    "6.1. Read all the messages from the response produced by the Sales Agent, if the request was declined or there are errors in the messages then answer immediately to the customer explaining the problem.\n"
    "\n"
    "### OUTPUT AND TONE:\n"
    "- Always be polite and friendly.\n"
    "- You MUST provide your final response in the structured format with all required fields.\n"
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
            get_current_date,
            get_all_item_names,
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
                "Processing customer order.",
                extra={"customer_message": customer_message}
            )
            user_prompt = (
                "Process the following customer request to extract all the necessary information and details to handle it. This is the customer request:\n"
                f"{customer_message}"
            )
            response = ctx.deps.order_processor_agent.run_sync(
                user_prompt=user_prompt,
                instructions=ORDER_PROCESSOR_INSTRUCTIONS,
            )
            logger.debug("Got response from order processor.", extra={"output": response.output})
            return response.output

        except Exception as e:
            logger.error(f"Error processing customer order.", exc_info=True,
                         extra={"customer_message": customer_message})
            return CustomerRequestDetails(
                messages=[
                    "I'm sorry, we encountered an error processing your order. Please try again later or contact customer service."],
                delivery_date="",
                request_metadata=RequestMetadata()
            )

    @orchestrator_agent.tool(docstring_format='google', require_parameter_descriptions=True)
    def generate_quote(ctx: RunContext[OrchestratorDependencies], items: Dict[str, int], delivery_date: str, request_date: str) -> List[
        QuotingAgentOutput]:
        """
        Communicate with the Quoting Agent to generate a quote of a single item of a customer request.

        Args:
            ctx: Context about the current call
            items: A dictionary of the ordered items and their quantities extracted from the customer request.
            delivery_date: The expected delivery date for the order, in ISO format.
            request_date: Date on which the order is made, in ISO format.

        Returns:
            A list of QuotingAgentOutput objects generated by the Quoting Agent.
            Each element corresponds to a quote for a single item-quantity pair in the order / customer request.
        """
        try:
            logger.info("Getting quote for items.", extra={"items": items, "delivery_date": delivery_date})
            quotes = []
            for item_name, quantity in items.items():
                user_prompt = (
                    "Please give me a quote. Here are the details about the customer request:\n"
                    f"- Item name: {item_name}\n"
                    f"- Quantity: {quantity}\n"
                    f"- Desired delivery date: {delivery_date}\n"
                    f"- Order date: {request_date}\n"
                )
                response = ctx.deps.quoting_agent.run_sync(user_prompt=user_prompt)
                logger.debug("Got response from quoting agent.", extra={"output": response.output})
                quotes.append(response.output)

            logger.debug("Got all item quotes from quoting agent.", extra={"quotes": quotes})
            return quotes
        except Exception as e:
            logger.error("Error producing quote.", exc_info=True,
                         extra={"items": items, "delivery_date": delivery_date})
            return [QuotingAgentOutput(
                messages=[
                    "I'm sorry, we encountered an error generating a quote. Please try again later or contact customer service."],
                quote_calculation=ERROR_QUOTE_CALCULATION,
                quote_explanation="",
                estimated_delivery_date=""
            )]

    @orchestrator_agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
    def get_latest_delivery_date(item_quotes: List[QuotingAgentOutput]) -> str:
        """
        Compares the estimated delivery dates from all the given quotes to get the latest one

        Args:
            item_quotes: Individual item quotes generated by the Quoting Agent

        Returns:
            The latest estimated delivery date in ISO format.
        """
        logger.debug("FUNC (get_latest_delivery_date): Getting latest delivery date.",
                     extra={"item_quotes": item_quotes})
        delivery_dates = [quote.estimated_delivery_date for quote in item_quotes if
                          quote.estimated_delivery_date is not None]
        if not delivery_dates:
            raise ValueError("Invalid state: no delivery dates found in item quotes.")
        return max(date.fromisoformat(date_str) for date_str in delivery_dates).isoformat()

    @orchestrator_agent.tool(docstring_format='google', require_parameter_descriptions=True)
    def manage_inventory(ctx: RunContext[OrchestratorDependencies], quotes_data: List[SimpleQuoteData],
                         latest_delivery_date: str, request_date: str) -> InventoryAgentOutput:
        """
        Communicate with the Inventory Agent to manage inventory for an order and replenish supplies if needed

        Args:
            ctx: Context about the current call
            quotes_data: Essential details extracted from the generated quotes.
            latest_delivery_date: The latest estimated delivery date from all the quotes, in ISO format.
            request_date: Date on which the order is made, in ISO format.

        Returns:
            The response that the Inventory Agent generated.
        """
        try:
            logger.info(
                f"Managing inventory for order.",
                extra={"quotes_data": [q.model_dump() for q in quotes_data],
                       "latest_delivery_date": latest_delivery_date,
                       "order_date": request_date}
            )
            items = {quote.item_name: {
                "quantity": quote.quantity,
                "unit_price": quote.unit_price,
            } for quote in quotes_data}

            user_prompt = (
                "A customer has placed an order with the details below. Please check the inventory stock levels for each item and replenish as needed following your rules.\n"
                f"- Desired delivery date: {latest_delivery_date}\n"
                f"- Order date: {request_date}\n"
                f"- Items (in JSON format):\n"
                f"{json.dumps(items)}"
            )
            response = ctx.deps.inventory_agent.run_sync(user_prompt=user_prompt)
            logger.debug("Got response from inventory agent.", extra={"output": response.output})
            return response.output

        except Exception as e:
            logger.error(f"Error managing inventory.",
                         exc_info=True,
                         extra={"quotes_data": quotes_data, "latest_delivery_date": latest_delivery_date,
                                "order_date": request_date})
            return InventoryAgentOutput(
                messages=[
                    "I'm sorry, we encountered an error managing the inventory for your order. Please try again later or contact customer service."],
            )

    @orchestrator_agent.tool(docstring_format='google', require_parameter_descriptions=True)
    def handle_sale(ctx: RunContext[OrchestratorDependencies], quotes_data: List[SimpleQuoteData],
                    latest_delivery_date: str, request_date: str) -> List[SalesAgentOutput]:
        """
        Communicate with the Sales agent to finalize a sales transaction.

        Args:
            ctx: Context about the current call
            quotes_data: Essential details extracted from the generated quotes.
            latest_delivery_date: The latest estimated delivery date from all the quotes, in ISO format.
            request_date: Date on which the order is made, in ISO format.

        Returns:
            A list of SalesAgentOutput objects generated by the Sales Agent.
            Each element corresponds to a sales operation for a single quoted item of the order.
        """
        try:
            logger.info(
                "Finalizing a sales transaction for the order.",
                extra={
                    "quotes_data": [q.model_dump() for q in quotes_data],
                    "latest_delivery_date": latest_delivery_date,
                    "order_date": request_date
                }
            )
            sales_responses = []
            for quote in quotes_data:
                user_prompt = (
                    "The customer wants to finalize an order.\n"
                    f"They previously received a quote for {quote.quantity} units "
                    f"of {quote.item_name} at a total price of ${quote.total_amount:.2f}. "
                    f"Please fulfill this order for delivery by {latest_delivery_date}."
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
                             "latest_delivery_date": latest_delivery_date,
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

    def process_customer_order(self, customer_message) -> OrchestratorAgentOutput:
        """
        Process a customer order through the coordinated agent workflow
        :param customer_message: The customer's order request
        :return: A human-readable response to the customer
        """

        try:
            logger.info("Processing customer order", extra={"customer_message": customer_message})
            response = self.orchestrator_agent.run_sync(user_prompt=customer_message,
                                                        deps=self.orchestrator_dependencies)
            output: OrchestratorAgentOutput = response.output
            return output

        except Exception as e:
            logger.error(f"Error processing customer order.", exc_info=True,
                         extra={"customer_message": customer_message})
            return OrchestratorAgentOutput(
                customer_response="I'm sorry, we encountered an error processing your order. Please try again later or contact customer service.",
                total_amount=-1,
                request_metadata=None
            )


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    logger.info("Initializing Database...")
    init_database(DB_ENGINE)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        logger.error("FATAL: Error loading test data", exc_info=True)
        return

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
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx + 1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
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

        response = orchestrator.process_customer_order(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report.cash_balance
        current_inventory = report.inventory_value

        print(f"Response: {response.model_dump_json(indent=2)}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report.cash_balance:.2f}")
    print(f"Final Inventory: ${final_report.inventory_value:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
