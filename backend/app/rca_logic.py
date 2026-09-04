"""
rca_logic.py — Deterministic implementation of the RCA playbook.

All thresholds live here as named constants.
No LLM involvement — this is pure Python logic that mirrors the playbook exactly.
The agent calls these functions and renders the returned dicts into natural language.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from backend.app.database import query, query_one, TABLE_NAME

# ---------------------------------------------------------------------------
# Thresholds (source of truth — match amazon_rca_logic.md exactly)
# ---------------------------------------------------------------------------

DEMAND_SPIKE_THRESHOLD = 1.10        # total_orders > order_projection * 1.10
BOOKING_GAP_THRESHOLD = 0.90         # current_capacity_booked < 0.90
UTILIZATION_GAP_THRESHOLD = 0.85     # man_hour < 0.85
SUSTAINED_PILEUP_HOURS = 3           # pileup in 3+ consecutive hours = sustained
OR2A_SLA_THRESHOLD = 0               # healthy: avg_or2a <= 0 (client-specific)


# ---------------------------------------------------------------------------
# Data fetch helpers
# ---------------------------------------------------------------------------

def get_store_hour_row(store: str, date: str, hour: int | float) -> dict | None:
    """Fetch a single store × date × hour row from the gold table."""
    return query_one(
        f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE store = ?
          AND charge_date = ?
          AND hour = ?
        """,
        (store, date, hour),
    )


def get_store_day_rows(store: str, date: str) -> list[dict]:
    """Fetch all hours for a store × date, ordered by hour."""
    return query(
        f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE store = ?
          AND charge_date = ?
        ORDER BY hour ASC
        """,
        (store, date),
    )


def get_problem_hours(store: str, date: str) -> list[dict]:
    """Return only the problem hours (is_problem_hour = 1) for a store × date."""
    return query(
        f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE store = ?
          AND charge_date = ?
          AND is_problem_hour = 1
        ORDER BY hour ASC
        """,
        (store, date),
    )


def get_hours_before(store: str, date: str, hour: int | float, n: int) -> list[dict]:
    """
    Fetch up to n rows immediately before `hour` for the same store × date,
    ordered ascending (oldest first). Used for pileup continuity checks.
    """
    return query(
        f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE store = ?
          AND charge_date = ?
          AND hour < ?
        ORDER BY hour DESC
        LIMIT ?
        """,
        (store, date, hour, n),
    )


# ---------------------------------------------------------------------------
# Individual RCA checks
# ---------------------------------------------------------------------------

def check_demand(row: dict) -> dict:
    """
    Check 1 — Demand Spike
    Flag if total_orders > order_projection * DEMAND_SPIKE_THRESHOLD
    """
    total_orders = row.get("total_orders") or 0
    order_projection = row.get("order_projection") or 0

    if order_projection and order_projection > 0:
        pct_over = ((total_orders - order_projection) / order_projection) * 100
        flag = total_orders > order_projection * DEMAND_SPIKE_THRESHOLD
    else:
        pct_over = 0.0
        flag = False  # can't judge spike without a projection

    return {
        "flag": flag,
        "total_orders": int(total_orders),
        "order_projection": round(float(order_projection), 1) if order_projection else None,
        "pct_over": round(pct_over, 1),
    }


def check_pileup(row: dict, prior_rows: list[dict]) -> dict:
    """
    Check 2 — Pileup
    Flag if pileup_flag = 1.
    Sustained = pileup_flag = 1 for this hour AND at least 2 prior consecutive hours
    (total 3+).
    prior_rows should be ordered most-recent-first (DESC) from get_hours_before().
    """
    pileup_flag = int(row.get("pileup_flag") or 0)
    pileup_count = int(row.get("pileup_count") or 0)
    flag = pileup_flag == 1

    # Count consecutive pileup hours going backwards
    consecutive = 1 if flag else 0
    for pr in prior_rows:
        if int(pr.get("pileup_flag") or 0) == 1:
            consecutive += 1
        else:
            break

    sustained = flag and (consecutive >= SUSTAINED_PILEUP_HOURS)

    return {
        "flag": flag,
        "pileup_count": pileup_count,
        "sustained": sustained,
        "consecutive_hours": consecutive,
    }


def check_booking(row: dict) -> dict:
    """
    Check 3a — Booking Gap (Supply L1)
    Flag if current_capacity_booked < BOOKING_GAP_THRESHOLD
    """
    booked_size = int(row.get("booked_size") or 0)
    current_size = int(row.get("current_size") or 0)

    if current_size and current_size > 0:
        capacity_booked = booked_size / current_size
    else:
        capacity_booked = None

    flag = (capacity_booked is not None) and (capacity_booked < BOOKING_GAP_THRESHOLD)

    return {
        "flag": flag,
        "booked_size": booked_size,
        "current_size": current_size,
        "capacity_booked": round(capacity_booked, 4) if capacity_booked is not None else None,
        "capacity_booked_pct": round(capacity_booked * 100, 1) if capacity_booked is not None else None,
    }


def check_utilization(row: dict) -> dict:
    """
    Check 3b — Utilization Gap (Supply L2)
    Flag if man_hour < UTILIZATION_GAP_THRESHOLD.
    man_hour = rider_hours_per_hour / booked_hours_per_hour (already a ratio in the table).
    Note: rider_hours_per_hour and booked_hours_per_hour are stored in minutes — divide by 60 for display.
    """
    man_hour = row.get("man_hour")
    noshow_count = int(row.get("noshow_count") or 0)

    # rider_hours_per_hour and booked_hours_per_hour are in minutes in the DB
    rider_minutes = row.get("rider_hours_per_hour") or 0
    booked_minutes = row.get("booked_hours_per_hour") or 0
    rider_hours = round(rider_minutes / 60, 2)
    booked_hours = round(booked_minutes / 60, 2)

    flag = (man_hour is not None) and (float(man_hour) < UTILIZATION_GAP_THRESHOLD)

    return {
        "flag": flag,
        "man_hour": round(float(man_hour), 4) if man_hour is not None else None,
        "man_hour_pct": round(float(man_hour) * 100, 1) if man_hour is not None else None,
        "noshow_count": noshow_count,
        "rider_hours": rider_hours,
        "booked_hours": booked_hours,
    }


# ---------------------------------------------------------------------------
# Single-hour RCA
# ---------------------------------------------------------------------------

def run_hour_rca(store: str, date: str, hour: int | float) -> dict:
    """
    Run all three RCA checks for a single store × date × hour.
    Returns a structured dict. The LLM renders this dict into natural language.
    """
    row = get_store_hour_row(store, date, hour)
    if row is None:
        return {
            "store": store,
            "date": date,
            "hour": hour,
            "error": f"No data found for {store} on {date} hour {hour}",
        }

    prior_rows = get_hours_before(store, date, hour, n=SUSTAINED_PILEUP_HOURS - 1)

    demand = check_demand(row)
    pileup = check_pileup(row, prior_rows)
    booking = check_booking(row)
    utilization = check_utilization(row)

    # Collect triggered flag names for the summary line (only if it's a problem hour)
    triggered_flags = []
    is_problem = int(row.get("is_problem_hour") or 0) == 1
    
    if is_problem:
        if demand["flag"]:
            triggered_flags.append("DEMAND SPIKE")
        if pileup["flag"]:
            triggered_flags.append("SUSTAINED PILEUP" if pileup["sustained"] else "PILEUP")
        if booking["flag"]:
            triggered_flags.append("BOOKING GAP")
        if utilization["flag"]:
            triggered_flags.append("UTILIZATION GAP")

    return {
        "store": store,
        "date": date,
        "hour": int(hour),
        "avg_or2a": round(float(row.get("avg_or2a") or 0), 2),
        "or2a_threshold": OR2A_SLA_THRESHOLD,
        "is_problem_hour": is_problem,
        "total_orders": int(row.get("total_orders") or 0),
        "breached_count": int(row.get("breached_count") or 0),
        "breached_rate": round(float(row.get("breached_rate") or 0), 4),
        "demand": demand,
        "pileup": pileup,
        "supply": {
            "booking": booking,
            "utilization": utilization,
        },
        "triggered_flags": triggered_flags,
        "any_flag": len(triggered_flags) > 0,
    }


# ---------------------------------------------------------------------------
# Store-day RCA (all problem hours)
# ---------------------------------------------------------------------------

def run_store_day_rca(store: str, date: str) -> dict:
    """
    Run RCA for all problem hours of a store × date.
    Also returns a day-level summary with weighted metrics.
    """
    all_rows = get_store_day_rows(store, date)
    if not all_rows:
        return {
            "store": store,
            "date": date,
            "error": f"No data found for {store} on {date}",
        }

    problem_rows = [r for r in all_rows if int(r.get("is_problem_hour") or 0) == 1]

    # --- Day-level weighted aggregates ---
    total_orders_sum = sum(int(r.get("total_orders") or 0) for r in all_rows)
    breached_count_sum = sum(int(r.get("breached_count") or 0) for r in all_rows)

    weighted_or2a_num = sum(
        float(r.get("avg_or2a") or 0) * int(r.get("total_orders") or 0)
        for r in all_rows
    )
    weighted_avg_or2a = (
        round(weighted_or2a_num / total_orders_sum, 2) if total_orders_sum else None
    )
    weighted_breached_rate = (
        round(breached_count_sum / total_orders_sum, 4) if total_orders_sum else None
    )

    worst_hour_row = max(all_rows, key=lambda r: float(r.get("avg_or2a") or 0))

    # --- Run per-hour RCA for each problem hour ---
    hour_rcas = []
    for row in problem_rows:
        hour_rca = run_hour_rca(store, date, row["hour"])
        hour_rcas.append(hour_rca)

    return {
        "store": store,
        "date": date,
        "day_summary": {
            "total_orders": total_orders_sum,
            "breached_count": breached_count_sum,
            "weighted_avg_or2a": weighted_avg_or2a,
            "weighted_breached_rate": weighted_breached_rate,
            "total_hours": len(all_rows),
            "problem_hours_count": len(problem_rows),
            "problem_hours": [int(r["hour"]) for r in problem_rows],
            "worst_hour": int(worst_hour_row["hour"]),
            "worst_hour_or2a": round(float(worst_hour_row.get("avg_or2a") or 0), 2),
        },
        "hour_rcas": hour_rcas,
    }


# ---------------------------------------------------------------------------
# Performance summary (for "how did X do?" questions)
# ---------------------------------------------------------------------------

def get_store_performance(store: str, date: str, hour: int | float | None = None) -> dict:
    """
    Summarise performance for a store × date, optionally filtered to a single hour.
    Returns weighted metrics suitable for display.
    """
    if hour is not None:
        rows = [get_store_hour_row(store, date, hour)]
        rows = [r for r in rows if r is not None]
    else:
        rows = get_store_day_rows(store, date)

    if not rows:
        return {"store": store, "date": date, "hour": hour, "error": "No data found"}

    total_orders_sum = sum(int(r.get("total_orders") or 0) for r in rows)
    breached_count_sum = sum(int(r.get("breached_count") or 0) for r in rows)
    problem_hours = [r for r in rows if int(r.get("is_problem_hour") or 0) == 1]

    weighted_or2a_num = sum(
        float(r.get("avg_or2a") or 0) * int(r.get("total_orders") or 0)
        for r in rows
    )
    weighted_avg_or2a = (
        round(weighted_or2a_num / total_orders_sum, 2) if total_orders_sum else None
    )
    weighted_breached_rate = (
        round(breached_count_sum / total_orders_sum, 4) if total_orders_sum else None
    )

    return {
        "store": store,
        "date": date,
        "hour": hour,
        "total_orders": total_orders_sum,
        "breached_count": breached_count_sum,
        "weighted_avg_or2a": weighted_avg_or2a,
        "weighted_breached_rate": weighted_breached_rate,
        "weighted_breached_rate_pct": round(weighted_breached_rate * 100, 2) if weighted_breached_rate is not None else None,
        "problem_hours_count": len(problem_hours),
        "problem_hours": [int(r["hour"]) for r in problem_hours],
        "hour_breakdown": [
            {
                "hour": int(r["hour"]),
                "total_orders": int(r.get("total_orders") or 0),
                "avg_or2a": round(float(r.get("avg_or2a") or 0), 2),
                "breached_rate": round(float(r.get("breached_rate") or 0), 4),
                "is_problem_hour": int(r.get("is_problem_hour") or 0),
            }
            for r in rows
        ],
    }


def get_city_performance(city: str, date: str) -> dict:
    """
    City-level summary for a given date.
    Aggregates across all stores in the city, weighted by order volume.
    """
    rows = query(
        f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE city = ?
          AND charge_date = ?
        ORDER BY store, hour ASC
        """,
        (city, date),
    )

    if not rows:
        return {"city": city, "date": date, "error": "No data found"}

    # Store-level roll-up
    store_map: dict[str, list] = {}
    for r in rows:
        store_map.setdefault(r["store"], []).append(r)

    store_summaries = []
    for store, store_rows in store_map.items():
        total_orders_sum = sum(int(r.get("total_orders") or 0) for r in store_rows)
        breached_count_sum = sum(int(r.get("breached_count") or 0) for r in store_rows)
        problem_hours = [r for r in store_rows if int(r.get("is_problem_hour") or 0) == 1]

        weighted_or2a_num = sum(
            float(r.get("avg_or2a") or 0) * int(r.get("total_orders") or 0)
            for r in store_rows
        )
        weighted_avg_or2a = (
            round(weighted_or2a_num / total_orders_sum, 2) if total_orders_sum else None
        )

        store_summaries.append({
            "store": store,
            "total_orders": total_orders_sum,
            "breached_count": breached_count_sum,
            "weighted_avg_or2a": weighted_avg_or2a,
            "weighted_breached_rate_pct": round(breached_count_sum / total_orders_sum * 100, 2) if total_orders_sum else None,
            "problem_hours_count": len(problem_hours),
        })

    # City-level totals
    city_total_orders = sum(s["total_orders"] for s in store_summaries)
    city_breached = sum(s["breached_count"] for s in store_summaries)
    city_weighted_or2a_num = sum(
        (s["weighted_avg_or2a"] or 0) * s["total_orders"] for s in store_summaries
    )
    city_weighted_avg_or2a = (
        round(city_weighted_or2a_num / city_total_orders, 2) if city_total_orders else None
    )

    return {
        "city": city,
        "date": date,
        "city_totals": {
            "total_orders": city_total_orders,
            "breached_count": city_breached,
            "weighted_avg_or2a": city_weighted_avg_or2a,
            "weighted_breached_rate_pct": round(city_breached / city_total_orders * 100, 2) if city_total_orders else None,
            "total_stores": len(store_summaries),
        },
        "store_summaries": sorted(store_summaries, key=lambda s: s.get("weighted_avg_or2a") or 0, reverse=True),
    }


def list_stores(city: str | None = None, date: str | None = None) -> list[str]:
    """Return distinct store codes, optionally filtered by city and/or date."""
    conditions = []
    params = []
    if city:
        conditions.append("city = ?")
        params.append(city)
    if date:
        conditions.append("charge_date = ?")
        params.append(date)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = query(f"SELECT DISTINCT store FROM {TABLE_NAME} {where} ORDER BY store", tuple(params))
    return [r["store"] for r in rows]