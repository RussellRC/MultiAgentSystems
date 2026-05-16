# Project Reflections

## Framework Selection
I did a quick deep dive into the proposed options before settling on my stack.
I looked at `npcpy`, but it’s really built for heavy-duty data science and mathematical workflows (`NumPy`/`Pandas`).\
Since this project is focused on business logic and inventory orchestration, `npcpy` felt like a "wrong tool for the job".

I decided to go with `Pydantic-AI` instead for a few practical reasons:

**Intuitive Abstractions**\
While reading the documentation, it was fairly easy to understand.\
The way it encompasses and handles all the main concepts and functionality that we learned throughout the program
(`RunContext`, different prompt types, thinking, tool definitions, evaluation, etc.) seemed very straightforward to me.

**Type Safety**\
Since we’ve used `Pydantic` throughout this program, it felt natural.\
Later on, I realized that it’s great at keeping the data clean and preventing the agents
from "hallucinating" prices or stock numbers (more on that below).

**Learning Something New**\
We used `smolagents` in Course 4, so I saw this as a great opportunity to expand my technical breadth.


## Design decisions

### TL;DR: Architectural Pivot

* **Initial Approach:** "Smart Agents" with complex system prompts designed to handle multi-step reasoning and business logic (restocking, date math, inventory analysis).

* **The Problem:** High prompt complexity led to indefinite loops, hallucinations, and exceeding API limits, 
  even when using stronger models like GPT-4 and high-effort Thinking.

* **Final Decision:** I shifted to a "Thin Agent, Fat Tools" architecture. I offloaded deterministic logic—like calculating shortages and reorder quantities—into function tools (e.g., `analyze_order_stock_requirements`).
  * **Stateless Orchestration:** I intentionally avoided the complexity of an in-memory shared context.
    Instead, I used the Database as the single source of truth, passing explicit DTOs between agents via the Orchestrator.

* **Result:** Reduced "thinking complexity" for the agents, eliminated math errors, and improved reliability across all test cases.

### My engineering journey (Wall of Text version)

My initial approach to implementing the project was to build "smart" and "robust" agents. 
I wanted to make each agent capable of handling a broad range of tasks within their respective domains, 
while still verifying the correctness of the inputs and also ensuring that the system state remained consistent.\
For example, the Inventory Agent would be capable of handling different types of requests:
answering questions about inventory, handling stock levels to fulfill orders,
and autonomously replenishing supplies. This involved a lot of math and business logic.

At first, it looked promising (individual unit tests were working as expected).
However, some issues started to arise: the system prompts were long and complex,
and there was a considerable amount of overlapping functionality.

Moreover, once I implemented the Orchestrator Agent and workflow, it became clear that the agent's prompts
were more complex than needed. In fact, the complexity was hampering the agents to produce satisfactory responses, 
and ultimately unable to succesfully answer the intputs from the `quote_requests_sample.csv` file.
For example, the agents would go into long or infinite loops and exceed the Vocaerum API request limits,
or hallucinate and produce bogus responses.

Then I tried using a stronger model (`gpt_4`) and `effort='high'` thinking for the more complex agents (Quoting and Inventory). 
Although it certainly helped, I was quickly running out of the provided Vocareum budgets,
but more importantly, the agents still couldn't produce accurate and satisfactory responses for most of the provided test cases.

In the end, after running through multiple iterations of testing, refining the prompts, and simplifying the agents, 
I decided for a **"Thin Agent, Fat Tools"** approach. 
This proved to be the most effective and cost-efficient implementation given the requirements and constraints.
> The best example of this is in the `analyze_order_stock_requirements` and `order_shortage_from_supplier` tools. 
> I tried very, very hard, but just couldn't craft a good enough prompt to make the Inventory Agent successfully and consistently 
> handle the complexity of all the computing and business logic done in those functions.


## Areas of improvement for the project
**Atomicity and Consistency**\
The current system lacks transactional integrity. Because the workflow spans multiple agent calls and database writes, 
a failure in the middle of a sale could result in "orphan" transactions (e.g., `stock_order` transaction is created but the corresponding `sales` one is not).
* **Idempotency & Compensation:** Most tools are not currently idempotent. 
  Implementing a Saga Pattern or compensation logic would allow the system to "roll back" or "undo" partial successes if a downstream step fails.
* **Concurrency Control:** The system is currently designed for sequential processing. 
  To support a multi-user environment, I would need to implement distributed locking or optimistic concurrency control 
  to prevent race conditions (imagine the chaos if two users tried to buy the last ream of A4 paper simultaneously 😉).

**Intelligent Context & Session Persistence**\
While I chose a stateless architecture for simplicity, the system would benefit from a more sophisticated memory layer. 
* **Shared Context:** Implementing a shared state would allow agents to access immutable data 
  (like the `CustomerRequestDetails` or the `OrderQuote`) without having to pass DTOs and variables back and forth.
* **Long-Term Memory:** Currently, every request is treated as a "first-time" interaction. 
  Integrating a persistent Thread/Session Management layer would allow the agent to remember the customer's specific preferences
  or follow up on previous orders across different days.

**Advanced Retrieval (RAG) for Pricing**\
The `search_quote_history` tool currently relies on basic database queries. 
For non-specific items like "high-quality colored glossy paper", the Quoting Agent struggles to find relevant historical matches
and get the correct (or most accurate) unit price.
* **Semantic Search:** Upgrading the RAG mechanism to use Vector Embeddings would allow the agent to find "spiritually similar"
  items even when the exact name doesn't match, leading to much more accurate and competitive pricing for custom requests.

**Observability and Debugging**\
Debugging the "black box" agent reasoning was the primary pain point of this project.
* **Logfire Integration:** As noted in the Pydantic-AI documentation, integrating a production-grade tool like Logfire 
  would provide the visual traces needed to debug complex reasoning loops and tool-call sequences efficiently.

**Request Prioritization and Intelligent Queuing**
While the current implementation extracts metadata such as customer mood, job type, event type, and "need size" from the requests,
this data is not used at all for workflow control.\
A production-grade version could use these signals to implement Priority Queuing. 
For example, requests marked as "Urgent" or originating from high-volume users could be routed to 
higher-tier models (like GPT-4o) or prioritized in the processing pipeline to ensure faster fulfillment and higher service standards.

**Interactive & Asynchronous Workflows**\
The application is currently a "request-response" system focused solely on order fulfillment.
* **Conversational Logic:** I would love a more agentic loop where the system could ask clarifying questions rather than assuming or failing,
  e.g., _"Did you want the high-quality colored glossy A4, A3, or Cardstock paper?"_ 
* **Full Lifecycle Management:** Expanding the domain to handle order tracking, returns, and proactive stock alerts 
  would transform the tool from a "dumb sales bot" into a comprehensive Beaver's Choice Business Assistant Almighty.

**Tech debt**
* Some prompts are brittle as they reference specific fields of the response, which might not be needed if the Pydantic models have good descriptions
* Create output validators for all the agents
* More unit tests
  * Make it possible to test the orchestrator workflow partially
  * Edge-cases for all individual agent tests
* More evaluators (with scores).
  * LLM Judge maybe?
* Ensure that every tool returns a Pydantic model.
* Correlation IDs for request tracing in the logs

## Key takeaways

**The "Thin Agent, Fat Tools" Architectural Pivot**\
There is a fundamental trade-off between agent autonomy and system reliability. 
My experience with this project led me to move away from "Smart Agents" in favor of a Thin Agent, Fat Tools pattern. 
* As prompt complexity grows, agents are more likely to enter infinite loops or hallucinate. 
  By offloading business logic into deterministic tools, I ensured 100% accuracy in math and date logic.
* Simpler prompts require fewer "thinking resources" (time and tokens), making the system significantly more cost-effective and responsive.
* While robust tools simplify individual agents, the Orchestrator's role becomes more critical. 
  The complexity moves from "how to calculate" to "how to sequence," which was much easier to debug in a central workflow.

**Schema-First Design and Strict I/O Contracts**\
Defining clear, strict Pydantic models for inputs and outputs acts as a "guardrail" for the LLM.
* Using the `OrderQuote` model with a granular `QuoteCalculation` breakdown forced the Quoting Agent to "show its work."
  This significantly improved accuracy compared to returning just an "answer with explanation" string.
* Using consistent variable namings and Pydantic models for passing data between agents and tools prevented "silent failures" 
  that I discovered through my testing (agents would pass incorrect dates or quantities).

**"Stateless" Orchestration over Shared Context**
I intentionally chose to pass data explicitly through the Orchestrator rather than implementing a Shared Context state.
* By relying on the database as the "Single Source of Truth" and passing DTOs between agents, I avoided the complexity of syncing in-memory state with the database.
* This stateless approach allowed me to make the system more transparent.
  While troubleshooting and debugging, I could easily track every piece of data passed between the agents.
* I also discovered that for passing data from agents to tools, the simpler the DTOs, the better.

**The Power of "Chain of Thought" and Prompt Engineering**\
I found that enabling "Thinking" capabilities on the agents is a must-have for multi-step tasks. 
Without it, my agents were just incapable of executing complex procedures correctly.\
On the other hand, although good system prompts are about well-redacted instructions, 
they are also about role definition and clear rules and boundaries.

**The Importance of Automated Evaluation**\
One of my biggest takeaways was that you cannot improve what you cannot measure.
* From the very beginning, I realized that unit testing was essential and necessary for evaluating the performance of the agents, and fixing them.
* Using `pydantic_evals` allowed me to easily see how prompt changes affected performance for the agents, 
  both in isolation and in the whole workflow.

