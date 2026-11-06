"""Q2 - FDMA vs TDMA with dedicated-channel allocation under 4 traffic datasets,
plus capacity-with-overheads calculation and the telemetry design decision."""
import heapq
from common import *

# ------------------------------------------------------------ capacity with overheads
# Assumptions (GSM-like, cited in README: 3GPP TS 45.001/45.002; ETSI GSM 05.xx)
BAND_HZ = 2e6
FDMA_CH_HZ, TDMA_CAR_HZ, SLOTS = 25e3, 200e3, 8
FDMA_GUARD_CH = 2          # 1 channel guard at each band edge (assumed)
FDMA_CTRL_CH = 2           # dedicated control/signalling channels (assumed)
TDMA_GUARD_CAR = 1         # 100 kHz guard each band edge = 1 carrier (GSM: 125 -> 124 usable)
TDMA_CTRL_SLOTS = 2        # BCCH/CCCH slot + SDCCH slot (assumed)
# GSM normal burst: 156.25 bit periods per slot, 114 payload bits; 26-frame multiframe: 24 traffic + SACCH + idle
TDMA_PAYLOAD_EFF = (114 / 156.25) * (24 / 26)
FDMA_EFF = 1.0
FDMA_SETUP_S, TDMA_SETUP_S = 0.25, 0.45     # assumed channel-assignment hold time per call (TDMA adds frame sync/timing advance)


def capacity_table():
    f_nom = int(BAND_HZ // FDMA_CH_HZ); t_nom = int(BAND_HZ // TDMA_CAR_HZ) * SLOTS
    f_use = f_nom - FDMA_GUARD_CH - FDMA_CTRL_CH
    t_use = (int(BAND_HZ // TDMA_CAR_HZ) - TDMA_GUARD_CAR) * SLOTS - TDMA_CTRL_SLOTS
    rows = [
        dict(scheme="FDMA", nominal_channels=f_nom, guard_loss=FDMA_GUARD_CH, control_loss=FDMA_CTRL_CH,
             usable_traffic_channels=f_use, per_channel_payload_efficiency=FDMA_EFF,
             effective_capacity_channel_equiv=round(f_use * FDMA_EFF, 1)),
        dict(scheme="TDMA", nominal_channels=t_nom, guard_loss=TDMA_GUARD_CAR * SLOTS, control_loss=TDMA_CTRL_SLOTS,
             usable_traffic_channels=t_use, per_channel_payload_efficiency=round(TDMA_PAYLOAD_EFF, 3),
             effective_capacity_channel_equiv=round(t_use * TDMA_PAYLOAD_EFF, 1)),
    ]
    df = pd.DataFrame(rows); save_csv(df, "q2_capacity_overheads.csv"); return df


# ------------------------------------------------------------ traffic datasets
def make_dataset(kind, rg, T=3600.0):
    """Returns arrays (arrival, hold, duty). duty = fraction of the held time spent actually transmitting."""
    if kind in ("A_voice", "B_bursty", "C_mix_70_30"):
        lam = 70 / 120.0                                   # 70 Erlang offered, mean hold 120 s
        n = rg.poisson(lam * T); arr = np.sort(rg.uniform(0, T, n)); hold = rg.exponential(120, n)
        if kind == "A_voice": duty = np.ones(n)
        elif kind == "B_bursty": duty = np.full(n, 0.10)
        else: duty = np.where(rg.random(n) < 0.7, 1.0, 0.10)
    else:                                                  # D_edge: short calls + synchronised flash crowds
        n1 = rg.poisson(10 * T); a1 = rg.uniform(0, T, n1)
        flash = np.repeat(np.arange(60, T, 120.0), 100)    # 100 calls arrive at the same instant every 120 s
        arr = np.sort(np.concatenate([a1, flash])); n = len(arr)
        hold = np.maximum(1.0, rg.exponential(4.0, n)); duty = np.ones(n)
    return arr, hold, duty


def simulate_pool(arr, hold, duty, C, eff, setup, T=3600.0, nominal=80):
    """Dedicated allocation, blocked-calls-cleared (loss system)."""
    ends = []; blocked = 0; busy = active = 0.0
    for a, h, d in zip(arr, hold, duty):
        while ends and ends[0] <= a: heapq.heappop(ends)
        if len(ends) >= C: blocked += 1; continue
        e = a + h + setup; heapq.heappush(ends, e)
        clip = min(e, T) - a
        busy += clip; active += d * min(h, max(0, T - a - setup))
    n = len(arr)
    cap = nominal * T
    return dict(offered_calls=n, blocked_calls=blocked, blocking_pct=100 * blocked / n,
                occupancy_pct_of_usable=100 * busy / (C * T),
                useful_util_of_held_pct=100 * active * eff / busy if busy else 0,
                useful_util_of_nominal_pct=100 * active * eff / cap,
                idle_while_held_pct_of_nominal=max(0.0, 100 * (busy - active - (n - blocked) * setup) / cap),
                setup_wasted_pct_of_nominal=100 * (n - blocked) * setup / cap,
                framing_loss_pct_of_nominal=100 * active * (1 - eff) / cap,
                structural_overhead_pct_of_nominal=100 * (nominal - C) / nominal)


def run_datasets(reps=20):
    cap = capacity_table()
    Cf = int(cap.loc[0, "usable_traffic_channels"]); Ct = int(cap.loc[1, "usable_traffic_channels"])
    rows = []
    for ki, kind in enumerate(["A_voice", "B_bursty", "C_mix_70_30", "D_edge_short_sync"]):
        for rep in range(reps):
            rg = rng(2000 + ki * 100 + rep)
            ds = make_dataset("D_edge" if kind.startswith("D") else kind, rg)
            for scheme, C, eff, setup in [("FDMA", Cf, FDMA_EFF, FDMA_SETUP_S), ("TDMA", Ct, TDMA_PAYLOAD_EFF, TDMA_SETUP_S)]:
                r = simulate_pool(*ds, C, eff, setup); r.update(dataset=kind, scheme=scheme, replicate=rep); rows.append(r)
    raw = pd.DataFrame(rows); save_csv(raw, "q2_raw_all_replicates.csv")
    summ = raw.groupby(["dataset", "scheme"]).mean(numeric_only=True).reset_index().drop(columns="replicate")
    save_csv(summ, "q2_summary_mean.csv")
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    ds = list(summ.dataset.unique()); x = np.arange(len(ds)); w = 0.38
    for ax, col, ttl in zip(axs[:3], ["blocking_pct", "useful_util_of_nominal_pct", "idle_while_held_pct_of_nominal"],
                            ["Blocking probability (%)", "Useful utilisation (% of nominal 80 ch)",
                             "Capacity idle while held (% of nominal)"]):
        for i, s in enumerate(["FDMA", "TDMA"]):
            v = [summ[(summ.dataset == d) & (summ.scheme == s)][col].values[0] for d in ds]
            ax.bar(x + (i - .5) * w, v, w, label=s)
        ax.set_xticks(x); ax.set_xticklabels(["A voice", "B bursty", "C 70/30", "D edge"], fontsize=8); ax.set_title(ttl); ax.legend()
    # stacked breakdown for TDMA vs FDMA
    comps = ["useful_util_of_nominal_pct", "idle_while_held_pct_of_nominal", "setup_wasted_pct_of_nominal",
             "framing_loss_pct_of_nominal", "structural_overhead_pct_of_nominal"]
    names = ["useful", "idle while held", "setup", "framing", "guard+control"]
    for d_i, d in enumerate(["B_bursty"]):
        for i, s in enumerate(["FDMA", "TDMA"]):
            bottom = 0
            for c, n, colr in zip(comps, names, ["#2a9d8f", "#e76f51", "#e9c46a", "#8d99ae", "#6d597a"]):
                v = summ[(summ.dataset == d) & (summ.scheme == s)][c].values[0]
                axs[3].bar(i, v, bottom=bottom, color=colr, label=n if i == 0 else None); bottom += v
        axs[3].set_xticks([0, 1]); axs[3].set_xticklabels(["FDMA", "TDMA"]); axs[3].set_title("Where nominal capacity goes (dataset B bursty)")
        axs[3].legend(fontsize=6)
    fig.suptitle("Q2 - dedicated channel allocation, 20 replicates x 4 datasets")
    save_fig(fig, "q2_fdma_vs_tdma.png")
    return summ


# ------------------------------------------------------------ telemetry design decision
def erlang_b(A, C):
    b = 1.0
    for k in range(1, C + 1): b = A * b / (k + A * b)
    return b


def max_load_erlang_b(C, target=0.01):
    lo, hi = 0.0, float(C) * 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if erlang_b(mid, C) > target: hi = mid
        else: lo = mid
    return lo


def telemetry(C=76, rate_bps=9600, payload=200, header=30, setup=0.25, period=60):
    tx = (payload + header) * 8 / rate_bps
    duty_tx = tx / period
    duty_circuit = (tx + setup) / period
    rows = [
        dict(mode="Permanent dedicated channel per device", devices_supported=C, airtime_duty_pct=100 * duty_tx,
             channel_busy_fraction_per_device_pct=100.0, signalling_msgs_per_report=0),
        dict(mode="Dedicated channel per report (circuit set-up each time), 1% blocking",
             devices_supported=int(max_load_erlang_b(C) / duty_circuit), airtime_duty_pct=100 * duty_tx,
             channel_busy_fraction_per_device_pct=100 * duty_circuit, signalling_msgs_per_report=3),
        dict(mode="Shared pool, slotted-ALOHA style contention (G=0.1 per channel)",
             devices_supported=int(C * 0.1 / duty_tx * 0.9), airtime_duty_pct=100 * duty_tx,
             channel_busy_fraction_per_device_pct=100 * duty_tx, signalling_msgs_per_report=0.1)  # extra (re)transmissions per report,
    ]
    df = pd.DataFrame(rows); df["report_airtime_s"] = tx
    save_csv(df, "q2_telemetry_decision.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh([r["mode"][:48] for r in rows], df.devices_supported, color=["#e76f51", "#e9c46a", "#2a9d8f"])
    ax.set_xscale("log"); ax.set_xlabel("devices supported (log scale)")
    for i, v in enumerate(df.devices_supported): ax.text(v, i, f" {v}", va="center")
    ax.set_title(f"Q2 telemetry site: 230 B/min per device, report airtime {tx:.2f} s ({100*duty_tx:.2f} % duty)")
    save_fig(fig, "q2_telemetry_decision.png")
    return df


if __name__ == "__main__":
    print("Q2 ...")
    s = run_datasets(); print(s.round(2).to_string()); print(telemetry().round(2).to_string())
    c = capacity_table(); print(c.to_string())
    save_json(dict(C_fdma=int(c.loc[0, "usable_traffic_channels"]), C_tdma=int(c.loc[1, "usable_traffic_channels"]),
                   bursty_util_fdma=float(s[(s.dataset == "B_bursty") & (s.scheme == "FDMA")].useful_util_of_held_pct.values[0]),
                   bursty_util_tdma=float(s[(s.dataset == "B_bursty") & (s.scheme == "TDMA")].useful_util_of_held_pct.values[0]),
                   voice_util_fdma=float(s[(s.dataset == "A_voice") & (s.scheme == "FDMA")].useful_util_of_nominal_pct.values[0]),
                   voice_util_tdma=float(s[(s.dataset == "A_voice") & (s.scheme == "TDMA")].useful_util_of_nominal_pct.values[0])),
              "q2_summary.json")
