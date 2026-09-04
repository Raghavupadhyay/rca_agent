# RCA Agent: Intelligent Delivery SLA Diagnostics

An advanced, multi-turn conversational agent designed to perform deterministic **Root Cause Analysis (RCA)** on quick-commerce delivery operations. This system helps operations managers identify why specific stores or cities are failing to meet **OR2A (Order Received to Assigned)** SLAs.

## 🚀 Key Features

- **Deterministic RCA**: Unlike standard LLM agents, this system uses a "frozen" Python logic layer (`rca_logic.py`) to calculate metrics and attribute root causes, ensuring 100% accuracy and zero hallucinations.
- **Multi-turn Conversation**: Built with **LangGraph**, the agent maintains context (City, Store, Date, Hour) across multiple turns, allowing for natural drill-down analysis.
- **MCP Integration**: Uses the **Model Context Protocol (MCP)** to dynamically read operational playbooks and metric definitions from the `docs/` folder.
- **Unified Interface**: A FastAPI backend serves both a REST API and a built-in interactive chat interface.
- **SQL Escape Hatch**: For complex ad-hoc queries, the agent can generate and execute safe, read-only SQL against the performance database.

## 🛠️ Tech Stack

- **Orchestration**: LangChain & LangGraph (StateGraph)
- **Model**: Google Gemini 2.5 Flash
- **API Framework**: FastAPI & Uvicorn
- **Database**: SQLite (Performance Data)
- **Knowledge Base**: MCP Filesystem Server
- **Environment**: Python 3.12+ / `uv` for package management

## 📁 Project Structure

```text
rca_agent/
├── backend/
│   ├── main.py          # FastAPI application & MCP server lifecycle
│   ├── agent.py         # LangGraph orchestration & message extraction logic
│   ├── rca_logic.py     # The "Brain": Deterministic RCA playbook implementation
│   ├── database.py      # SQLite connection management
│   ├── tools.py         # Tool bindings for LLM (Performance, RCA, SQL, etc.)
├── docs/                # Knowledge Base (accessed via MCP)
│   ├── metrics.md       # Definitions for OR2A, SLA Breaches, etc.
│   └── playbook.md      # Diagnostic steps for Demand, Supply, and Pileups
├── scripts/             # Utility Scripts
│   ├── csv_to_sql.py    # ETL script to load CSV data into SQLite
│   └── test_agent.py    # CLI-based testing script for the agent
├── data.sqlite          # Primary application database
├── README.md            # You are here
└── pyproject.toml       # Project dependencies
```

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.12 or higher.
- A Google Gemini API Key.

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_api_key"
GEMINI_MODEL="gemini-2.5-flash"
DB_PATH="data.sqlite"
```

### 3. Install Dependencies
Using `uv`:
```bash
uv sync
```
Or using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Initialize the Database
The agent expects a table named `my_table` in `data.sqlite`. You can use the provided script to load your CSV data:
```bash
python scripts/csv_to_sql.py
```

## 🏃 Running the Application

### Start the FastAPI Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
- **Web UI**: Open `http://localhost:8000` in your browser for the interactive chat interface.
- **API Endpoint**: `POST /api/chat` (expects `{"message": "string", "thread_id": "string"}`).

## 🧠 RCA Methodology

The agent attributes performance issues to four primary buckets:

1.  **Demand Spike**: `total_orders > 110% of order_projection`.
2.  **Sustained Pileup**: Orders carried over for 3 or more consecutive hours.
3.  **Booking Gap**: `< 90%` of scheduled capacity was booked.
4.  **Utilization Gap**: Rider man-hour ratio is `< 0.85` (often due to no-shows).

**Note**: Root causes are only analyzed for "Problem Hours" where the `avg_or2a` exceeds the defined SLA threshold (default: 0 min).
