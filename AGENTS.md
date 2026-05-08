# AGENTS.md — MultiAgentSystems (Munder Difflin)

## Quick Commands

```bash
# Install deps (Poetry)
poetry install

# Run tool tests
poetry run python -m unittest tests/tools_tests.py

# Run agent-specific tests
poetry run python -m unittest tests/[agent]_tests.py
```

## Architecture

- **Single-file project**:
  - All code and resources live in `project/`.
  - The active implementation is `project/project.py`
  - Tests live in `tests/`.
  - Untouched starter template is in `starter/project_starter.py`.
- **Agent frameworks**: `pydantic-ai` with `gpt-4o` models via an OpenAI-compatible Vocareum proxy.
- **Testing frameworks**: `unittest` and `pydantic-evals`
- **Database**: SQLite (`munder_difflin.db`). Tables: `transactions`, `inventory`, `quote_requests`, `quotes`.
- **Inventory**: Randomized subset (40%, seed=137) of ~57 paper/product types → 22 items. Starting cash: $50,000.
- **Workflow**: Orchestrator → Inventory Agent / Quoting Agent / Ordering Agent (defined in `project.py`). Tools registered as Python functions agents.

## Environment

- **`.env`** in `project/` sets `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Loaded via `python-dotenv` at module import time (`project.py:539`).
- **`DATABASE_URL`** env var overrides the DB path. Tests set it to a different file *before* importing the module.
- **Never commit `.env`** — it contains live Vocareum credentials.

## Test Setup and Gotchas

- Tests are located in the `tests/` directory.
- Tests do `os.chdir()` to the `project` directory in `setUpClass` so `init_database()` can find `quotes.csv` and `quote_requests.csv`.
- `init_database()` **recreates all tables** on each call (`if_exists="replace"`). Safe to call before every test.

## Key Conventions

- Transaction types are exactly `'stock_orders'` or `'sales'` (string literals, not enums).
- Stock orders use the **estimated delivery date** as the transaction date, not the request date — this keeps pending orders invisible in inventory until they arrive.
- `project.py` uses `uuid7` for transaction IDs; `project_starter.py` relies on SQLite auto-increment.
- All DB queries in `project.py` use `sqlalchemy.sql.text()` with parameterized params — never string-interpolate user input into SQL.

## Submission Requirements (Udacity)

1. Completed Python file (only one — `project.py`)
2. Workflow diagram (image file)
3. Reflective report explaining the system
4. `test_results.csv` output from a full test run
