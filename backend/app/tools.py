"""
tools.py — LangChain tool definitions wired to rca_logic and database.

Each tool has:
  - A tight, unambiguous description so the LLM picks the right one
  - A Pydantic input schema for structured calling
  - Minimal logic — it delegates to rca_logic or database, never re-implements

MCP filesystem tools (for read_doc) are initialised separately in agent.py
and added to this list at runtime.
"""

import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

from backend.app import rca_logic
from backend.app.database import query, TABLE_NAME


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class PerformanceInput(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    store: Optional[str] = Field(default=None, description="Store code e.g. STORE_101")
    city: Optional[str] = Field(default=None, description="City name e.g. Bangalore")
    hour: Optional[float] = Field(default=None, description="Hour of day as a number e.g. 10")


class HourRCAInput(BaseModel):
    store: str = Field(description="Store code e.g. STORE_101")
    date: str = Field(description="Date in YYYY-MM-DD format")
    hour: float = Field(description="Hour of day as a number e.g. 10")


class StoreDayRCAInput(BaseModel):
    store: str = Field(description="Store code e.g. STORE_101")
    date: str = Field(description="Date in YYYY-MM-DD format")


class ListStoresInput(BaseModel):
    city: Optional[str] = Field(default=None, description="City name to filter by")
    date: Optional[str] = Field(default=None, description="Date to filter by in YYYY-MM-DD format")


class SQLInput(BaseModel):
    sql: str = Field(description="A read-only SELECT query to run against the gold table")


class HourRangeRCAInput(BaseModel):
    store: str = Field(description="Store code e.g. STORE_101")
    date: str = Field(description="Date in YYYY-MM-DD format")
    hour_start: float = Field(description="Start hour inclusive e.g. 6")
    hour_end: float = Field(description="End hour inclusive e.g. 11")


# ---------------------------------------------------------------------------
# Tool 1: query_performance
# Use for "how did X do?" questions — city, store, or single hour.
# ---------------------------------------------------------------------------

@tool("query_performance", args_schema=PerformanceInput)
def query_performance(date: str, store: Optional[str] = None, city: Optional[str] = None, hour: Optional[float] = None) -> str:
    """
    Returns a performance summary (order volumes, avg OR2A, breach rate, problem hours)
    for a city, store, or specific store-hour on a given date.
    Use this when the user asks 'how did X do', 'what were the numbers for', or
    wants an overview before a drill-down. Do NOT use for root-cause questions.
    """
    try:
        if city and not store:
            result = rca_logic.get_city_performance(city, date)
        elif store:
            result = rca_logic.get_store_performance(store, date, hour)
        else:
            return json.dumps({"error": "Provide at least a city or store."})
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 2: run_hour_rca
# Use for "why did this specific hour underperform?" questions.
# ---------------------------------------------------------------------------

@tool("run_hour_rca", args_schema=HourRCAInput)
def run_hour_rca(store: str, date: str, hour: float) -> str:
    """
    Runs the full deterministic RCA playbook (demand spike, pileup, booking gap,
    utilization gap) for a single store × date × hour.
    Use when the user asks why a specific hour was bad, or when drilling into a
    particular hour after a day-level overview.
    """
    try:
        result = rca_logic.run_hour_rca(store, date, hour)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 3: run_store_day_rca
# Use for "why did this store underperform?" day-level questions.
# ---------------------------------------------------------------------------

@tool("run_store_day_rca", args_schema=StoreDayRCAInput)
def run_store_day_rca(store: str, date: str) -> str:
    """
    Runs the RCA playbook across ALL problem hours for a store on a given date.
    Returns a day-level summary plus per-hour RCA results for every problem hour.
    Use when the user asks 'why did STORE_X underperform' or 'what went wrong at STORE_X'.
    """
    try:
        result = rca_logic.run_store_day_rca(store, date)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 4: run_hour_range_rca
# Use for "walk me through the morning hours" type questions.
# ---------------------------------------------------------------------------

@tool("run_hour_range_rca", args_schema=HourRangeRCAInput)
def run_hour_range_rca(store: str, date: str, hour_start: float, hour_end: float) -> str:
    """
    Runs the RCA playbook for every hour in [hour_start, hour_end] inclusive
    for a given store and date. Use for 'walk me through the morning/evening hours'
    or 'what happened between hour X and Y'.
    """
    try:
        rows = rca_logic.query(
            f"""
            SELECT hour FROM {TABLE_NAME}
            WHERE store = ?
              AND charge_date = ?
              AND hour >= ?
              AND hour <= ?
            ORDER BY hour ASC
            """,
            (store, date, hour_start, hour_end),
        )
        hours = [r["hour"] for r in rows]
        if not hours:
            return json.dumps({"error": f"No data for {store} on {date} between hours {hour_start}-{hour_end}"})

        results = [rca_logic.run_hour_rca(store, date, h) for h in hours]
        return json.dumps({"store": store, "date": date, "hour_start": hour_start, "hour_end": hour_end, "hours": results}, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 5: list_stores
# Utility — enumerate stores in the data, optionally filtered by city.
# ---------------------------------------------------------------------------

@tool("list_stores", args_schema=ListStoresInput)
def list_stores(city: Optional[str] = None, date: Optional[str] = None) -> str:
    """
    Returns the list of distinct store codes present in the data,
    optionally filtered by city and/or date.
    Use when the user asks 'which stores are in city X' or when you need to
    enumerate stores before running a city-wide analysis.
    """
    try:
        stores = rca_logic.list_stores(city, date)
        return json.dumps({"stores": stores, "count": len(stores)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 6: query_sql
# Escape hatch for ad-hoc questions that don't fit the above tools.
# ---------------------------------------------------------------------------

@tool("query_sql", args_schema=SQLInput)
def query_sql(sql: str) -> str:
    """
    Executes a read-only SELECT query against the quick_commerce_orders_gold table
    and returns the results as JSON. Use only for ad-hoc analysis questions that
    cannot be answered by the other tools. Never use for RCA — use run_store_day_rca
    or run_hour_rca instead. Only SELECT statements are permitted.
    Table name: quick_commerce_orders_gold
    """
    try:
        results = query(sql)
        # Cap at 200 rows to avoid flooding the context
        truncated = len(results) > 200
        return json.dumps({
            "rows": results[:200],
            "row_count": len(results),
            "truncated": truncated,
        }, default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# All tools list (MCP read_doc added dynamically in agent.py)
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    query_performance,
    run_store_day_rca,
    run_hour_rca,
    run_hour_range_rca,
    list_stores,
    query_sql,
]