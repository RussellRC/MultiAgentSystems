# Agent Workflow Diagram

```mermaid
flowchart TD

    subgraph "Multi-Agent System"
        Orchestrator[OrchestratorAgent\nClassifies, Routes & Coordinates]
        InventoryAgent[InventoryAgent\nAnswers stock queries & manages reorders]
        QuotesAgent[QuotesAgent\nGenerates competitive quotes]
        SalesAgent[SalesAgent\nFinalizes transactions]

        ClassifyInput{Classify Request}
        IsStockQuery[Stock/Inventory Query?]
        IsQuoteRequest[Quote/Price Request?]
        IsOrderRequest[Order/Purchase Request?]
    end

    CustomerInput --> Orchestrator
    Orchestrator --> ClassifyInput
    ClassifyInput --> IsStockQuery
    ClassifyInput --> IsQuoteRequest
    ClassifyInput --> IsOrderRequest

    IsStockQuery --> |Yes| InventoryAgent
    IsQuoteRequest --> |Yes| QuotesAgent
    IsOrderRequest --> |Yes| SalesAgent

    InventoryAgent --> Orchestrator
    QuotesAgent --> Orchestrator
    SalesAgent --> Orchestrator

    Orchestrator --> SystemOutput([System Response])

    subgraph "Tools by Agent"
        InventoryAgent --> InvTools["get_inventory_items_by_name\nget_stock_level\nget_all_inventory\nget_items_below_min_stock_level\nget_supplier_delivery_date\ncreate_transaction(stock_orders)\nget_cash_balance"]
        QuotesAgent --> QuoteTools["search_quote_history"]
        SalesAgent --> SalesTools["get_stock_level\nget_supplier_delivery_date\ncreate_transaction(sales)\ngenerate_financial_report"]
    end

    subgraph "Data Source"
        SQLiteDB[("SQLite Database\n(munder_difflin.db)")]
    end

    InvTools <--> SQLiteDB
    QuoteTools <--> SQLiteDB
    SalesTools <--> SQLiteDB

    classDef agent fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef tool fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef db fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    class Orchestrator,InventoryAgent,QuotesAgent,SalesAgent agent;
    class InvTools,QuoteTools,SalesTools tool;
    class SQLiteDB db;
    class ClassifyInput,IsStockQuery,IsQuoteRequest,IsOrderRequest decision;
```

## Orchestrator Responsibilities
- Request Classification: Determine if input is: stock query, quote request, or order request
- Routing: Send to InventoryAgent (stock), QuotesAgent (pricing), SalesAgent (fulfillment)
- State Coordination: Track cash balance & inventory state across agent calls (shared FactoryState pattern from l6)
- Multi-step workflows: For orders: InventoryAgent → check stock → QuotesAgent → price → SalesAgent → fulfill
- Error Handling: Handle insufficient inventory, low cash, unavailable items
- Response Aggregation: Combine agent outputs into coherent customer response


### Request examples
- Example: "what is the current stock of paper plates"
  - Type: Stock / Inventory query
  - Handled by: Inventory Agent

- Example: "I need 500 sheets of glossy paper"
  - Type: Quote Request 
  - Handled by: Quotes Agent

- Example: "I would like to place an order for..."
  - Type: Order purchase
  - Handled by: Sales Agent

- Example: "500 A4 paper, 300 cardstock, 200 washi tape"
  - Type: Complex multi-item
  - Handled by: Orchestrator --> Inventory Agent (check stock) --> Quotes Agent (price) --> Sales Agent (fulfill)

