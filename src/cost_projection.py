"""
Problem 3 (rebuilt for Scenario 1 scope): AWS infrastructure cost projection,
grounded entirely in real values measured from the CMU dataset - no invented
session-length assumption.

Why this changed from the original version
--------------------------------------------
The study's scope is Scenario 1 ATO defense: keystroke-dynamics verification
of a single AUTHENTICATION EVENT (login or step-up re-authentication), not
continuous full-session monitoring. CMU only measures password-entry events,
so the cost model now measures exactly what CMU can support: the cost of
streaming keystroke timing during ONE authentication event, at three polling
resolutions, scaled to a large number of such events.

Two inputs are now measured directly from your real dataset instead of
guessed:
  1. Event duration: mean time to type the CMU password, reconstructed from
     real DD/H timing values (~2.58s, computed below - not hardcoded, so it
     stays correct if you rerun this against updated data).
  2. Payload size: bytes of a realistic single-keystroke JSON message, built
     from real H/DD/UD values sampled across the dataset (~134 bytes).

Deployment scale: "1 million concurrent sessions" from the proposal is
reinterpreted, consistent with Scenario 1, as 1 million AUTHENTICATION
EVENTS per month (logins / step-up re-auths) - not 1 million simultaneous
5-minute browsing sessions. State this interpretation explicitly if citing
this table.

Two client architectures are costed
-----------------------------------
The polling interval only drives message volume if the client transmits
every poll sample. That is one possible design, not the only one, so the
projection reports both ends of the range:

  - sample_streaming: the client forwards one message per poll tick for the
    duration of the event (messages = duration / dt). Message volume, and
    therefore cost, falls in direct proportion to the polling interval. This
    is the UPPER BOUND on what a coarser interval can save.
  - event_driven: the client sends one message per key press and per key
    release (2 x 11 keys = 22 messages per password), however often it polls.
    The polling interval then only sets the resolution of the timestamps
    inside those messages; message volume and cost do not change with it.

Read together: the saving from a coarser interval is real only for
sample-streaming designs. For event-driven designs the benefit of the
accuracy results is tolerance of coarse timestamps, not a lower AWS bill.

AWS WebSocket API pricing used (public price list, verified mid-2026):
  - $1.00 per million messages
  - $0.25 per million connection-minutes
  - $0.09/GB data transfer out, after first 100GB/month free

Run: python3 src/cost_projection.py
"""
import argparse
import json
import numpy as np
import pandas as pd

from downsample import KEYS

MESSAGE_COST_PER_MILLION = 1.00
CONNECTION_MINUTE_COST_PER_MILLION = 0.25
DATA_TRANSFER_COST_PER_GB = 0.09
FREE_TIER_GB = 100

GROUP_DT_SECONDS = {"control_1ms": 0.001, "test_20ms": 0.020, "test_30ms": 0.030}
ARCHITECTURES = ["sample_streaming", "event_driven"]
KEY_EVENTS_PER_PASSWORD = 2 * len(KEYS)   # one press + one release per key


def measure_event_duration(df: pd.DataFrame) -> float:
    """Mean real elapsed time (seconds) to type the CMU password, reconstructed
    from actual DD (press-to-press) and final H (hold) timing values."""
    dd_cols = [f"DD.{KEYS[i]}.{KEYS[i+1]}" for i in range(len(KEYS) - 1)]
    total_duration = df[dd_cols].sum(axis=1) + df["H.Return"]
    return float(total_duration.mean())


def measure_payload_bytes(df: pd.DataFrame, n_samples: int = 400) -> float:
    """Mean byte size of a realistic single-keystroke JSON message, built from
    real H/DD/UD values sampled across the dataset."""
    sizes = []
    step = max(1, len(df) // n_samples)
    for i in range(0, len(df), step):
        row = df.iloc[i]
        msg = {
            "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "key_index": 3,
            "timestamp_ms": 15234,
            "H": round(float(row["H.t"]), 4),
            "DD": round(float(row["DD.t.i"]), 4),
            "UD": round(float(row["UD.t.i"]), 4),
        }
        sizes.append(len(json.dumps(msg).encode("utf-8")))
    return float(np.mean(sizes))


def project_costs(event_duration_sec: float, payload_bytes: float,
                   n_events: int) -> pd.DataFrame:
    rows = []
    for architecture, group, dt in [(a, g, dt) for a in ARCHITECTURES
                                    for g, dt in GROUP_DT_SECONDS.items()]:
        if architecture == "sample_streaming":
            messages_per_event = event_duration_sec / dt
        else:
            messages_per_event = KEY_EVENTS_PER_PASSWORD
        total_messages = messages_per_event * n_events
        total_connection_minutes = (event_duration_sec / 60) * n_events

        message_cost = (total_messages / 1_000_000) * MESSAGE_COST_PER_MILLION
        connection_cost = (total_connection_minutes / 1_000_000) * CONNECTION_MINUTE_COST_PER_MILLION

        total_bytes = total_messages * payload_bytes
        total_gb = total_bytes / (1024 ** 3)
        billable_gb = max(0, total_gb - FREE_TIER_GB)
        transfer_cost = billable_gb * DATA_TRANSFER_COST_PER_GB

        total_cost = message_cost + connection_cost + transfer_cost

        rows.append({
            "architecture": architecture,
            "group": group,
            "poll_interval_ms": dt * 1000,
            "messages_per_event": round(messages_per_event),
            "total_messages_millions": total_messages / 1_000_000,
            "total_data_gb": total_gb,
            "message_cost_usd": message_cost,
            "connection_cost_usd": connection_cost,
            "data_transfer_cost_usd": transfer_cost,
            "total_monthly_cost_usd": total_cost,
        })

    result_df = pd.DataFrame(rows)
    # savings are relative to the 1ms row of the SAME architecture
    baseline_cost = result_df.groupby("architecture")["total_monthly_cost_usd"].transform("first")
    result_df["cost_savings_vs_1ms_pct"] = (1 - result_df["total_monthly_cost_usd"] / baseline_cost) * 100
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=1_000_000,
                         help="Number of authentication events per month (default 1,000,000)")
    parser.add_argument("--csv", type=str, default="data/raw/DSL-StrongPasswordData.csv",
                         help="Path to the real CMU CSV to measure duration/payload from")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    event_duration_sec = measure_event_duration(df)
    payload_bytes = measure_payload_bytes(df)

    print("=" * 70)
    print("AWS API GATEWAY (WEBSOCKET) COST PROJECTION - Scenario 1 (auth events)")
    print("=" * 70)
    print(f"Measured from real CMU data:")
    print(f"  Event duration (mean password-entry time): {event_duration_sec:.3f} sec")
    print(f"  Payload size (mean, real feature values):   {payload_bytes:.1f} bytes")
    print(f"Assumed deployment scale: {args.n_events:,} authentication events/month")
    print("=" * 70)

    result_df = project_costs(event_duration_sec, payload_bytes, args.n_events)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(result_df.to_string(index=False))

    result_df.to_csv("results/cost_projection.csv", index=False)
    print("\nSaved results/cost_projection.csv")

    for architecture, arch_df in result_df.groupby("architecture", sort=False):
        baseline = arch_df["total_monthly_cost_usd"].iloc[0]
        print(f"\n{architecture} (1ms baseline ${baseline:,.2f}/month):")
        for _, row in arch_df.iloc[1:].iterrows():
            print(f"  {row['group']}: ${row['total_monthly_cost_usd']:,.2f}/month "
                  f"({row['cost_savings_vs_1ms_pct']:.1f}% cheaper than 1ms)")