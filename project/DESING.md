# Design

## Agent Workflow Diagram

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
    OrderProcessorAgent --> ProcTools["get_all_item_names\nget_current_date\n"]
    QuotingAgent --> QuoteTools["get_all_item_names\nsearch_quote_history\nget_inventory_items_by_name\n"]
    InventoryAgent --> InvTools["analyze_order_stock_requirements\norder_shortage_from_supplier\nreplenish_to_minimum"]
    SalesAgent --> SalesTools["create_transaction"]
  end

  subgraph "Data Source"
    SQLiteDB[("SQLite Database\n(munder_difflin.db)")]
  end

  ProcTools --> SQLiteDB
  QuoteTools --> SQLiteDB
  InvTools --> SQLiteDB
  SalesTools --> SQLiteDB
  
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

### Orchestrator Agent Responsibilities
* **Sequential Coordination:** Manage the multi-step order lifecycle by calling specialized tools in sequence.
* **Data Propagation:** Ensure quote details and delivery dates are passed correctly between agents.
* **Response Aggregation:** Synthesize the results from all agent steps into a single, cohesive response for the customer.

### Order Processor Agent Responsibilities
* Understand the user request and identify the necessary elements for order processing
* Validate the input elements
* Transform the request into a structured object

### Quoting Agent Responsibilities
* Generate a pricing quote for an order item
* Apply discounts based on historical generated quotes

### Inventory Agent Responsibilities
* Validate projected stock levels to fulfill orders
* Replenish stock levels to fulfill orders
* Replenish stock levels to keep required minimums

### Sales Agent
* "Finalize" sales by creating sales transactions 

## Workflow Sequence for Purchase Order processing
1. **Orchestrator** agent receives the user prompt and begins the workflow
2. **Orchestrator** requests the **Order Processing Agent** to parse and understand the user prompt to identify the necessary elements to handle the purchase order:
   - Order Delivery Date
   - Order Request Date: date at which the order is placed
   - Item-quantity pairs
   - Metadata: customer mood and job, order size and event type  
3. For each item in the order, the **Orchestrator** requests the **Quoting Agent** to generate a quote.
   - Before calling the agent, the `generate_quote` tool gets the price from `inventory` table and passes it in the prompt to the Quoting Agent if present 
   - The **Quoting Agent** uses the `search_quote_history` with the item name to identify unit price and apply at-will discounts  
   - The **Quoting Agent** returns a quote calculation breakdown along with a human-readable response
   - The **Orchestrator** aggregates all individual item quotes into a single Order Quote
4. The **Orchestrator** calls the **Inventory Agent** to handle the inventory
   - The **Inventory Agent** uses the `analyze_order_stock_requirements` tool to know whether the order can be fulfilled in time based on ordered quantities, projected stock levels and shortage coverage deliverability from the supplier
   - If the order is fulfillable, the **Inventory Agent** uses the `order_shortage_from_supplier` to create the necessary the `stock_orders` transactions to cover any shortages to fulfill the order
   - Finally, the **Inventory Agent** uses the `replenish_to_minimum` tool to replenish any pertinent item up to its minimum stock level
5. For each item in the order, the **Orchestrator** calls the **Sales Agent** to finalize the order purchase
   - The **Orchestrator** passes the quoted prices
   - The **Sales Agent** just uses the `create_transaction` tool to create a `sales` transaction for the order item with the given values
   - The **Orchestrator** aggregates all individual results
6. The aggregator generates a response and returns it


### Request flow example (simplified)

User input:
> I want to order 500 A4 paper, 300 cardstock, 200 washi tape. Deliver by `X`. (Date of request: `Y`)

1. **orchestrator_agent** --> call Tool `process_order_info(customer_message)`
   - `process_order_info` --> `order_processor_agent.run_sync("Extract info from this user request {customer_message}")`
   - **order_processor_agent** returns `CustomerRequestDetails`
      ```json
      {
        "items": {
          "A4 paper": 500,
          "cardstock": 300,
          "washi tape": 200, 
        },
        "delivery_date": "X",
        "request_date": "Y",
        "request_metadata": {...}
     }
      ```
2. **orchestrator_agent** --> call Tool `generate_quote(items)`
   - `generate_quote` --> (For each item in the order) `quoting_agent.run_sync("Generate a quote for this item & quantity {item_name}, {quanitty})`
   - **quoting_agent** returns `ItemQuote`
      ```json
      {
        "quote_calculation": { item_name, quanity, unit_price, base_amount, discount_rate, total },
        "quote_explanation": "Human-readable explanation of pricing with discount"
      } 
      ```
   - **`generate_quote`** --> Aggregate and return `OrderQuote`
      ```json
      {
        "item_quotes": [...],
        "base_total": sum(item_base_total),
        "total_amount": sum(total_amount)
      }
      ```
3. **orchestrator_agent** --> call Tool `manage_inventory(order_quote_data, delivery_date, request_date)`
   - `manage_inventory` --> `inventory_agent.run_sync("Handle the inventory for this order. {order_quote_data}, {delivery_date}, {request_date)"`
   - **inventory_agent** returns `InventoryAgentOutput`
     ```json
     {
        "placed_transactions": [...],
        "messages": [...]
     }
     ```
4. **orchestrator_agent** --> call Tool `handle_sale(order_quote_data, delivery_date, request_date)`
   - `handle_sale` --> (For each item in the order) `sales_agent.run_sync("Record the sale transaction for this order item {item_data}")`
   - **sales_agent** returns `SalesAgentOutput`
   ```json
      {
        "placed_transactions": [...],
        "messages": [...]
      }
   ```
   - `handle_sale` --> Aggregate results
5. **orchestrator_agent** --> Produce a response