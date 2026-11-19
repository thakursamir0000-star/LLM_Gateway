"""Monitoring dashboard over the gateway's SQLite request log.

    streamlit run dashboard/app.py

Reads the database named in the config (or $GATEWAY_DB). Shows requests,
cost, p50/p95 latency, error rate, the share sent to each model, stable vs
canary, per-app spend against budgets, and alerts.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from gateway.config import database_target, load_config, resolve_path
from gateway.manage.budgets import BudgetPolicy
from gateway.manage.releases import ReleaseManager
from gateway.monitoring import metrics
from gateway.monitoring.alerts import evaluate
from gateway.monitoring.store import Store, iso, utc_now

# Reference palette: categorical slots for series, reserved status colors for request outcomes.
BLUE, ORANGE = "#2a78d6", "#eb6834"
STATUS = {"ok": "#0ca30c", "rejected": "#fab219", "error": "#d03b3b"}
RANGES = {"Last 24 hours": timedelta(hours=24), "Last 7 days": timedelta(days=7), "Last 30 days": timedelta(days=30),
          "All time": None}

st.set_page_config(page_title="LLM Gateway", layout="wide")
config = load_config()
db_path = database_target(config)


@st.cache_resource
def open_store(path: str) -> Store:
    return Store(path)


st.title("LLM Gateway")
target_str = str(db_path)
is_sqlite = "://" not in target_str or target_str.startswith("sqlite")
if is_sqlite and not Path(db_path).exists():
    st.info(f"No request log yet at `{db_path}`. Send a request to the gateway, then refresh.")
    st.stop()
store = open_store(target_str)

with st.sidebar:
    range_name = st.selectbox("Time range", list(RANGES), index=1)
    apps = metrics.app_ids(store)
    app_choice = st.selectbox("App", ["All apps", *apps])
    st.button("Refresh")
    st.caption(f"Database: `{db_path}`")

now = utc_now()
span = RANGES[range_name]
since = iso(now - span) if span else None
app_id = None if app_choice == "All apps" else app_choice
bucket = "hour" if span is not None and span <= timedelta(days=1) else "day"

# ── deployment and alerts ───────────────────────────────────────────
state = ReleaseManager(config, store).state()
release = config.releases[state.active]
canary = f" · canary **{state.canary}** on {state.canary_share:.0%} of traffic" if state.canary else " · no canary"
router_file = resolve_path(config.routing.classifier_path)
router = config.routing.router if config.routing.router == "rules" or router_file.exists() else "rules (no trained model)"
batch = (f"; batch: cheap `{release.batch.cheap}`, strong `{release.batch.strong}`" if release.batch else "")
st.markdown(f"Release **{state.active}** (cheap `{release.cheap}`, strong `{release.strong}`{batch}, prompt "
            f"`{release.prompt}`){canary} · mode `{config.routing.default_mode}` · router `{router}` · "
            f"threshold {config.routing.threshold:.2f} · default latency class `{config.latency.default}`")

for alert in evaluate(store, config.alerts, now):
    st.error(f"**Alert ({alert.kind.replace('_', ' ')}):** {alert.message}", icon="⚠️")

# ── headline numbers ────────────────────────────────────────────────
s = metrics.summary(store, since=since, app_id=app_id)


def ms(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f} ms"


cols = st.columns(6)
cols[0].metric("Requests", f"{s['requests']:,}")
cols[1].metric("Error rate", f"{s['error_rate']:.1%}", help="Failed requests / (ok + failed). Rejected requests "
               "(client errors, budget limits) are not errors.")
cols[2].metric("Cost", f"${s['cost_usd']:,.4f}", help="List-price cost from the config's model prices.")
cols[3].metric("p50 latency", ms(s["p50_ms"]))
cols[4].metric("p95 latency", ms(s["p95_ms"]))
cols[5].metric("Sent to strong", f"{s['strong_share']:.0%}")
cols = st.columns(6)
cols[0].metric("Rejected", f"{s['rejected']:,}")
cols[1].metric("Cascade escalations", f"{s['escalated']:,}")
cols[2].metric("Fallbacks", f"{s['fallbacks']:,}")
cols[3].metric("Router p50 (added)", "–" if s["router_p50_ms"] is None else f"{s['router_p50_ms']:.2f} ms")
rated = s["feedback_up"] + s["feedback_down"]
cols[4].metric("Thumbs up", "–" if not rated else f"{s['feedback_up'] / rated:.0%}", help=f"{rated} ratings")
cols[5].metric("Tokens", f"{s['tokens']:,}")

if s["requests"] == 0:
    st.info("No requests in this range.")
    st.stop()

# ── over time ───────────────────────────────────────────────────────
series = pd.DataFrame(metrics.timeseries(store, since=since, app_id=app_id, bucket=bucket))
# Ordinal periods (not a time scale), so bars keep their width when there are only a few of them.
series["time"] = pd.to_datetime(series["bucket"]).dt.strftime("%b %d %H:00" if bucket == "hour" else "%b %d")
x = alt.X("time:O", title=None, sort=list(series["time"]), axis=alt.Axis(labelAngle=0))
unit = "hour" if bucket == "hour" else "day"
bar_size = max(3, min(28, 480 // max(1, len(series))))  # thin bars, whatever the number of periods
st.caption("Charts group requests by UTC " + unit + ".")

left, right = st.columns(2)
with left:
    st.subheader(f"Requests per {unit}")
    outcomes = series.rename(columns={"errors": "error"}).melt(
        id_vars=["time"], value_vars=list(STATUS), var_name="status", value_name="count")
    chart = alt.Chart(outcomes).mark_bar(size=bar_size, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=x, y=alt.Y("count:Q", title="requests", stack=True),
        color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS), range=list(STATUS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("time:O", title=unit), "status:N", "count:Q"])
    st.altair_chart(chart, use_container_width=True)
with right:
    st.subheader(f"Cost per {unit} (USD)")
    chart = alt.Chart(series).mark_bar(color=BLUE, size=bar_size, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=x, y=alt.Y("cost_usd:Q", title="USD"),
        tooltip=[alt.Tooltip("time:O", title=unit), alt.Tooltip("cost_usd:Q", title="cost", format="$.5f"),
                 "requests:Q"])
    st.altair_chart(chart, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Latency (successful requests)")
    latency = series.melt(id_vars=["time"], value_vars=["p50_ms", "p95_ms"], var_name="percentile",
                          value_name="ms").dropna()
    latency["percentile"] = latency["percentile"].str.replace("_ms", "")
    chart = alt.Chart(latency).mark_line(point=alt.OverlayMarkDef(size=64), strokeWidth=2).encode(
        x=x, y=alt.Y("ms:Q", title="ms"),
        color=alt.Color("percentile:N", scale=alt.Scale(domain=["p50", "p95"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("time:O", title=unit), "percentile:N", alt.Tooltip("ms:Q", format=",.0f")])
    st.altair_chart(chart, use_container_width=True)
with right:
    st.subheader("Error rate")
    chart = alt.Chart(series).mark_line(point=alt.OverlayMarkDef(size=64), strokeWidth=2, color=BLUE).encode(
        x=x, y=alt.Y("error_rate:Q", title=None, axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("time:O", title=unit), alt.Tooltip("error_rate:Q", format=".1%"), "errors:Q", "ok:Q"])
    limit = alt.Chart(pd.DataFrame({"limit": [config.alerts.error_rate]})).mark_rule(
        color=STATUS["error"], strokeWidth=1).encode(y="limit:Q")
    st.altair_chart(chart + limit, use_container_width=True)
    st.caption(f"Red rule: alert limit ({config.alerts.error_rate:.0%} over {config.alerts.window_minutes} min).")

# ── models, releases, apps ──────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("Share of answered requests per model")
    share = pd.DataFrame(metrics.model_share(store, since=since, app_id=app_id))
    if share.empty:
        st.caption("No answered requests in this range.")
    else:
        share["label"] = share["model"] + " (" + share["tier"] + ")"
        bars = alt.Chart(share).mark_bar(color=BLUE, cornerRadiusTopRight=2, cornerRadiusBottomRight=2).encode(
            x=alt.X("share:Q", title=None, axis=alt.Axis(format="%")),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["model:N", "tier:N", "requests:Q", alt.Tooltip("share:Q", format=".1%"),
                     alt.Tooltip("cost_usd:Q", title="cost", format="$.5f")])
        st.altair_chart(bars, use_container_width=True)
with right:
    st.subheader("Stable vs canary")
    releases = pd.DataFrame(metrics.release_stats(store, since=since))
    if not releases.empty:
        releases["side"] = releases["canary"].map({0: "stable", 1: "canary"})
        st.dataframe(releases[["release", "side", "requests", "error_rate", "p50_ms", "cost_usd", "feedback_up",
                               "feedback_down"]], hide_index=True, use_container_width=True,
                     column_config={"error_rate": st.column_config.NumberColumn("error rate", format="percent"),
                                    "p50_ms": st.column_config.NumberColumn("p50 ms", format="%.0f"),
                                    "cost_usd": st.column_config.NumberColumn("cost", format="$%.5f")})

st.subheader("By latency class")
classes = pd.DataFrame(metrics.latency_class_stats(store, since=since, app_id=app_id))
if not classes.empty:
    st.dataframe(classes[["latency_class", "requests", "error_rate", "fallbacks", "p50_ms", "p95_ms", "cost_usd"]],
                 hide_index=True, use_container_width=True,
                 column_config={"latency_class": "latency class",
                                "error_rate": st.column_config.NumberColumn("error rate", format="percent"),
                                "p50_ms": st.column_config.NumberColumn("p50 ms", format="%.0f"),
                                "p95_ms": st.column_config.NumberColumn("p95 ms", format="%.0f"),
                                "cost_usd": st.column_config.NumberColumn("cost", format="$%.5f")})
    st.caption("Interactive requests go to the release's fast models, batch requests to its batch models "
               "(X-Gateway-Latency header, or the app's default in the config).")

st.subheader("Spend today (UTC) against budgets")
policy = BudgetPolicy(config.budgets)
spend = pd.DataFrame(metrics.app_spend(store, iso(now)[:10]))
if spend.empty:
    st.caption("No requests today.")
else:
    spend["daily limit"] = spend["app_id"].map(policy.limit)
    spend["used"] = [min(1.0, c / limit) if limit else None for c, limit in zip(spend["cost_usd"], spend["daily limit"],
                                                                                strict=True)]
    st.dataframe(spend, hide_index=True, use_container_width=True,
                 column_config={"cost_usd": st.column_config.NumberColumn("spent", format="$%.5f"),
                                "daily limit": st.column_config.NumberColumn(format="$%.2f"),
                                "used": st.column_config.ProgressColumn("used", min_value=0, max_value=1,
                                                                        format="percent")})

with st.expander("Recent requests"):
    st.dataframe(pd.DataFrame(metrics.recent(store, limit=100, app_id=app_id)), hide_index=True,
                 use_container_width=True)
with st.expander("Alert history"):
    history = store.alerts(50)
    if history:
        st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
    else:
        st.caption("No alerts have fired.")
