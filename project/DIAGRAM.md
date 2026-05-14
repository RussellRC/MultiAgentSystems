# Agent Workflow Diagram

```mermaid
flowchart TD
  subgraph "Multi-Agent System"
    Orchestrator[Orchestrator Agent\nRoutes & Orchestrates]
    OrderProcessorAgent[Order Processor Agent\nParse and understand customer request]
    QuotingAgent[Quoting Agent\nGenerates competitive quotes]
    InventoryAgent[Inventory Agent\nAnswers stock queries & manages reorders]
    SalesAgent[Sales Agent\nFinalizes transactions]

    subgraph OrderWorkflow [Sequential Order Workflow Execution]
      direction TB
      OrderProcessorAgent -.-> QuotingAgent
      QuotingAgent -.-> InventoryAgent
      InventoryAgent -.-> SalesAgent
    end
  end

  CustomerInput([Customer Request]) --> Orchestrator
  Orchestrator --> OrderProcessorAgent
  Orchestrator --> QuotingAgent
  Orchestrator --> InventoryAgent
  Orchestrator --> SalesAgent
  Orchestrator --> SystemOutput([System Response])

  subgraph "Tools by Agent"
    OrderProcessorAgent --> ProcTools["get_current_date\nget_all_item_names"]
    QuotingAgent --> QuoteTools["get_all_item_names\nsearch_quote_history\nget_supplier_delivery_date\nestimates delivery based on quantity\nget_current_date"]
    InventoryAgent --> InvTools["get_minimum_cash_balance\nget_inventory_items_by_name\nget_stock_level\nget_all_inventory\nget_supplier_delivery_date,\ncreate_transaction,\nget_current_date,\nget_cash_balance"]
    SalesAgent --> SalesTools["get_all_item_names\nget_inventory_items_by_name\nget_stock_level\nget_supplier_delivery_date\ncreate_transaction\nget_current_date"]
  end

  subgraph "Data Source"
    SQLiteDB[("SQLite Database\n(munder_difflin.db)")]
  end

  ProcTools <--> SQLiteDB
  InvTools <--> SQLiteDB
  QuoteTools <--> SQLiteDB
  SalesTools <--> SQLiteDB
  
  Orchestrator:::agent
  OrderProcessorAgent:::agent
  QuotingAgent:::agent
  InventoryAgent:::agent
  SalesAgent:::agent
  ProcTools:::tool
  InvTools:::tool
  QuoteTools:::tool
  SalesTools:::tool
  SQLiteDB:::db

  style OrderWorkflow stroke:#D50000
  classDef agent fill: #e1f5fe, stroke: #03a9f4, stroke-width: 2px
  classDef tool fill: #fff3e0, stroke: #ff9800, stroke-width: 2px
  classDef db fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
```

## Orchestrator Responsibilities
* **Sequential Coordination:** Manage the multi-step order lifecycle by calling specialized tools in sequence.
* **Data Propagation:** Ensure quote details and delivery dates are passed correctly between agents.
* **Response Aggregation:** Synthesize the results from all agent steps into a single, cohesive response for the customer.

**Workflow Sequence for Orders:**
OrderProcessingAgent: Parse and understand the customer request.
QuotingAgent: Price items and provide a formal quote.
InventoryAgent: Verify stock levels and execute replenishment if below minimums.
SalesAgent: Fulfill the order and update the financial ledger.


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
  - Handled by: Orchestrator --> Order Processor Agent --> Quoting Agent (pricing) --> Inventory Agent (stock keeping) --> Sales Agent (fulfill)

