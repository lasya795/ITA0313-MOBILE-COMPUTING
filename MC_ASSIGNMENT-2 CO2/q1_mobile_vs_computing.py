"""Q1 - Mobile communication vs mobile computing.
Part A: own classification criteria + predictions vs rule-derived class.
Part B: simulation of case (e): offline meter-reading app at 90/60/30 % link availability.
"""
from collections import deque
from common import *

# ---------------------------------------------------------------- Part A
# Criteria (0..1 scale): how strongly each property applies to the case
CRITERIA = ["moves_while_in_use", "moves_between_sessions", "wireless", "link_dependence",
            "disconnection_tolerance_needed", "context_awareness", "device_limits"]

# predicted_* columns were written BEFORE the rule-based analysis was run.
CASES = [
 dict(id="a", name="Rugged tablet, work orders over cellular", moves_while_in_use=0.7, moves_between_sessions=1,
      wireless=1, link_dependence=0.9, disconnection_tolerance_needed=0.6, context_awareness=0.6, device_limits=0.6,
      predicted_class="mobile", predicted_problem="mobile communication"),
 dict(id="b", name="Laptop on Ethernet in different offices", moves_while_in_use=0.0, moves_between_sessions=1,
      wireless=0, link_dependence=0.6, disconnection_tolerance_needed=0.4, context_awareness=0.5, device_limits=0.3,
      predicted_class="nomadic", predicted_problem="mobile computing"),
 dict(id="c", name="Fixed wireless router at substation", moves_while_in_use=0.0, moves_between_sessions=0,
      wireless=1, link_dependence=0.9, disconnection_tolerance_needed=0.1, context_awareness=0.0, device_limits=0.1,
      predicted_class="fixed-wireless", predicted_problem="mobile communication"),
 dict(id="d", name="Push-to-talk handset", moves_while_in_use=1.0, moves_between_sessions=1,
      wireless=1, link_dependence=1.0, disconnection_tolerance_needed=0.3, context_awareness=0.2, device_limits=0.3,
      predicted_class="mobile", predicted_problem="mobile communication"),
 dict(id="e", name="Offline meter-reading tablet app (sync later)", moves_while_in_use=1.0, moves_between_sessions=1,
      wireless=1, link_dependence=0.2, disconnection_tolerance_needed=1.0, context_awareness=0.6, device_limits=0.9,
      predicted_class="mobile", predicted_problem="mobile computing"),
 dict(id="f*", name="OWN AMBIGUOUS: laptop on Wi-Fi inside a moving van with cellular router", moves_while_in_use=1.0,
      moves_between_sessions=1, wireless=1, link_dependence=0.7, disconnection_tolerance_needed=0.7,
      context_awareness=0.8, device_limits=0.3,
      predicted_class="mobile", predicted_problem="both (network mobility + app caching)"),
]


def classify(c):
    if c["wireless"] and c["moves_while_in_use"] >= 0.5:
        cls = "mobile"
    elif c["wireless"]:
        cls = "fixed-wireless"
    elif c["moves_between_sessions"]:
        cls = "nomadic"
    else:
        cls = "fixed-wired"
    comm = c["link_dependence"] * c["wireless"]            # pure link problem
    comp = np.mean([c["disconnection_tolerance_needed"], c["context_awareness"], c["device_limits"]])
    if comm > comp + 0.1:
        prob = "mobile communication"
    elif comp > comm + 0.1:
        prob = "mobile computing"
    else:
        prob = "both (network mobility + app caching)"
    return cls, prob, round(comm, 2), round(float(comp), 2)


def part_a():
    rows = []
    for c in CASES:
        cls, prob, comm, comp = classify(c)
        r = {k: c[k] for k in ["id", "name"] + CRITERIA}
        r.update(predicted_class=c["predicted_class"], derived_class=cls,
                 predicted_problem=c["predicted_problem"], derived_problem=prob,
                 comm_score=comm, comp_score=comp,
                 prediction_matches=(cls == c["predicted_class"] and prob == c["predicted_problem"]))
        rows.append(r)
    df = pd.DataFrame(rows)
    save_csv(df, "q1_classification.csv")
    fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off")
    show = df[["id", "predicted_class", "derived_class", "predicted_problem", "derived_problem",
               "comm_score", "comp_score", "prediction_matches"]]
    t = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.5)
    ax.set_title("Q1 - predicted vs rule-derived classification")
    save_fig(fig, "q1_classification_table.png")
    return df


# ---------------------------------------------------------------- Part B
def link_trace(avail, n_ticks, rg, mean_down=100):
    """Two-state Markov link (1 tick = 1 minute). Stationary availability = avail."""
    mean_up = avail / (1 - avail) * mean_down
    p_down, p_up = 1 / mean_up, 1 / mean_down
    s = rg.random() < avail
    tr = np.empty(n_ticks, bool)
    u = rg.random(n_ticks)
    for t in range(n_ticks):
        tr[t] = s
        s = (u[t] >= p_down) if s else (u[t] < p_up)
    return tr


def run_strategy(trace, strategy, n_updates, gen_interval, cap=None):
    q = deque(); delivered = delayed = overflow = 0; delays = []
    for t in range(len(trace)):
        up = trace[t]
        if t % gen_interval == 0 and t // gen_interval < n_updates:
            if strategy == "send_immediately":
                if up: delivered += 1; delays.append(0)
                else: overflow += 1                       # lost (no retry)
                continue
            if len(q) >= cap: overflow += 1               # device storage full
            else: q.append(t)
        if strategy != "send_immediately" and up and q:   # one update / tick when link up
            ts = q.popleft(); d = t - ts
            delivered += 1; delays.append(d); delayed += d > 0
    stranded = len(q)
    return dict(delivered_on_time=delivered - delayed, delivered_delayed=delayed,
                lost=overflow + stranded, lost_overflow_or_failed=overflow, stranded_at_end=stranded,
                mean_delay_min=float(np.mean(delays)) if delays else float("nan"),
                max_delay_min=max(delays) if delays else 0)


def part_b(n_updates=1000, gen_interval=5, drain=3000, reps=30):
    strategies = [("send_immediately", None), ("store_forward_cap100", 100), ("store_forward_cap20", 20)]
    n_ticks = n_updates * gen_interval + drain
    rows, rep_rows = [], []
    for avail in [0.9, 0.6, 0.3]:
        for rep in range(reps):
            tr = link_trace(avail, n_ticks, rng(int(avail * 100) * 1000 + rep))
            for name, cap in strategies:                 # same link trace for all strategies
                r = run_strategy(tr, name, n_updates, gen_interval, cap)
                r.update(availability=avail, strategy=name, replicate=rep,
                         measured_availability=round(float(tr.mean()), 3))
                rep_rows.append(r)
    raw = pd.DataFrame(rep_rows)
    save_csv(raw, "q1_sim_raw_all_replicates.csv")
    seed_run = raw[raw.replicate == 0].copy()
    save_csv(seed_run, "q1_sim_counts_seed_run.csv")
    mean = raw.groupby(["availability", "strategy"]).mean(numeric_only=True).reset_index().drop(columns="replicate")
    save_csv(mean, "q1_sim_mean_over_30_replicates.csv")
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["send_immediately", "store_forward_cap100", "store_forward_cap20"]
    w = 0.25; xs = np.arange(3)
    for i, name in enumerate(labels):
        sub = mean[mean.strategy == name].sort_values("availability", ascending=False)
        on, de, lo = sub.delivered_on_time.values, sub.delivered_delayed.values, sub.lost.values
        axs[0].bar(xs + (i - 1) * w, on, w, color="#2a9d8f", label="delivered on time" if i == 0 else None)
        axs[0].bar(xs + (i - 1) * w, de, w, bottom=on, color="#e9c46a", label="delivered delayed" if i == 0 else None)
        axs[0].bar(xs + (i - 1) * w, lo, w, bottom=on + de, color="#e76f51", label="lost" if i == 0 else None)
        for j in range(3): axs[0].text(xs[j] + (i - 1) * w, 1010, ["IM", "SF100", "SF20"][i], ha="center", fontsize=6, rotation=90)
    axs[0].set_xticks(xs); axs[0].set_xticklabels(["90 %", "60 %", "30 %"]); axs[0].set_ylim(0, 1150)
    axs[0].set_xlabel("link availability"); axs[0].set_ylabel("updates (of 1000)")
    axs[0].set_title("Q1 case (e): outcome of 1000 meter updates (mean of 30 runs)"); axs[0].legend(loc="lower left")
    for name in labels[1:]:
        sub = mean[mean.strategy == name].sort_values("availability", ascending=False)
        axs[1].plot(sub.availability * 100, sub.mean_delay_min, "o-", label=name)
    axs[1].set_xlabel("link availability (%)"); axs[1].set_ylabel("mean delivery delay (min)")
    axs[1].set_title("Mean delay of delivered updates (immediate = 0)"); axs[1].legend(); axs[1].invert_xaxis()
    save_fig(fig, "q1_simulation.png")
    return mean


if __name__ == "__main__":
    print("Q1 ...")
    part_a(); m = part_b(); print(m.round(2).to_string())
