"""Q7 - Diagnose, Redesign, Validate.
Chosen limitation (from Q2): dedicated FDMA/TDMA channels sit idle under bursty traffic (held-channel
utilisation ~10 % at 10 % duty, and sessions are BLOCKED while ~90 % of held airtime is idle).

DESIGN (summary; full text in README):
  Original  : each session holds one of C channels for its whole lifetime (loss system, Erlang-B).
  Redesign  : sessions hold NO channel. Admission control admits a session while
              active_sessions < N_max = floor(rho_max * C / declared_duty).
              Each burst sends a 1-message reservation request on the common control channel, is granted the
              first free channel from the pool (FIFO queue), transmits, then releases it immediately.
              Exploits: (1) known duty cycle of the traffic descriptor, (2) idle gaps between bursts.
  New costs : reservation signalling (2 msgs / burst), 20 ms per-burst header/sync overhead,
              50 ms grant latency, queueing delay, extra state (queue + admission counter).
"""
import heapq
from common import *
from q2_fdma_tdma import capacity_table

PREDICTION = """PREDICTION (written before running the experiment)
1. Session blocking: original follows Erlang-B on 76 channels and rises steeply above 76 Erlang;
   redesign should stay ~0 % until the admission limit N_max = 0.85*76/0.10 = 646 sessions.
2. Utilisation of occupied channel-time: original ~10 % (= duty cycle); redesign > 90 % (a channel is held only while sending).
3. Carried useful airtime at 300 Erlang: original ~7.6 channel-equivalents (76 x 10 %); redesign ~30 (4x more).
4. Costs: signalling messages per carried session grow from ~3 to ~3 + 2*(bursts per session ~15) = ~33 (10x);
   mean burst delay stays < 0.1 s at rho<=0.85 but grows sharply for rho_max >= 0.95.
5. For continuous traffic (duty = 100 %), redesign should be WORSE (admission limit 0.85*C < C, extra overhead) -> original preferable.
"""
HOLD_MEAN, BURST_MEAN, T = 300.0, 2.0, 3600.0
RHO_MAX = 0.85
GRANT_LATENCY, BURST_OVERHEAD = 0.05, 0.02


def gen_sessions(A, rg):
    n = rg.poisson(A / HOLD_MEAN * T); arr = np.sort(rg.uniform(0, T, n)); dur = rg.exponential(HOLD_MEAN, n)
    return arr, dur


def gen_bursts(start, end, duty, rg):
    if duty >= 0.999: return [(start, end - start)]
    idle = BURST_MEAN * (1 / duty - 1); t = start + rg.exponential(idle); out = []
    while t < end:
        b = min(rg.exponential(BURST_MEAN), end - t); out.append((t, b)); t += b + rg.exponential(idle)
    return out


def original(arr, dur, duty, C, rg):
    ends = []; adm = []; blocked = 0
    for a, d in zip(arr, dur):
        while ends and ends[0] <= a: heapq.heappop(ends)
        if len(ends) >= C: blocked += 1; continue
        heapq.heappush(ends, a + d); adm.append((a, a + d))
    air = busy = 0.0; nb = 0
    for s, e in adm:
        for t, b in gen_bursts(s, e, duty, rg):
            if t < T: air += min(b, T - t); nb += 1
        busy += min(e, T) - s
    n = len(arr)
    return dict(sessions_offered=n, sessions_blocked=blocked, blocking_pct=100 * blocked / n,
                carried_airtime_ch_s=air, busy_channel_s=busy, util_of_occupied_pct=100 * air / busy if busy else 0,
                pool_busy_pct=100 * busy / (C * T), mean_burst_delay_s=0.0, p95_burst_delay_s=0.0,
                signalling_msgs=2 * n + (n - blocked), bursts_served=nb)


def redesign(arr, dur, duty, C, rg, rho=RHO_MAX):
    nmax = int(np.floor(rho * C / duty)); ends = []; adm = []; blocked = 0
    for a, d in zip(arr, dur):
        while ends and ends[0] <= a: heapq.heappop(ends)
        if len(ends) >= nmax: blocked += 1; continue
        heapq.heappush(ends, a + d); adm.append((a, a + d))
    bursts = []
    for s, e in adm: bursts += [(t, b) for t, b in gen_bursts(s, e, duty, rg) if t < T]
    bursts.sort(); free = [0.0] * C; delays = []; air = busy = 0.0
    for t, b in bursts:
        f = heapq.heappop(free); st = max(t, f); sv = b + BURST_OVERHEAD
        heapq.heappush(free, st + sv); delays.append(st - t + GRANT_LATENCY); air += b; busy += sv
    n = len(arr); dl = np.array(delays) if delays else np.zeros(1)
    return dict(sessions_offered=n, sessions_blocked=blocked, blocking_pct=100 * blocked / n,
                carried_airtime_ch_s=air, busy_channel_s=busy, util_of_occupied_pct=100 * air / busy if busy else 0,
                pool_busy_pct=100 * busy / (C * T), mean_burst_delay_s=float(dl.mean()), p95_burst_delay_s=float(np.percentile(dl, 95)),
                signalling_msgs=2 * n + (n - blocked) + 2 * len(bursts), bursts_served=len(bursts), N_max=nmax)


def experiment(label, grid, C, reps=5):
    rows = []
    for gi, g in enumerate(grid):
        A, duty, rho = g
        for rep in range(reps):
            arr, dur = gen_sessions(A, rng(7000 + gi * 50 + rep))
            for name, fn in [("original", lambda r: original(arr, dur, duty, C, r)), ("redesign", lambda r: redesign(arr, dur, duty, C, r, rho))]:
                r = fn(rng(7500 + gi * 50 + rep)); r.update(experiment=label, design=name, offered_erlang=A, duty=duty, rho_max=rho, replicate=rep); rows.append(r)
    return pd.DataFrame(rows)


def run():
    with open(os.path.join(DATA, "q7_prediction_written_before_run.txt"), "w") as f: f.write(PREDICTION)
    C = int(capacity_table().loc[0, "usable_traffic_channels"]); print("  usable FDMA channels C =", C)
    e1 = experiment("E1_load_sweep", [(A, 0.10, RHO_MAX) for A in [20, 40, 76, 150, 300, 500, 650, 800]], C)
    e2 = experiment("E2_duty_sweep", [(150, d, RHO_MAX) for d in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]], C)
    e3 = experiment("E3_rho_sweep", [(1200, 0.10, r) for r in [0.5, 0.7, 0.85, 0.95, 0.99]], C)
    raw = pd.concat([e1, e2, e3]); save_csv(raw, "q7_raw_all_replicates.csv")
    mean = raw.groupby(["experiment", "design", "offered_erlang", "duty", "rho_max"]).mean(numeric_only=True).reset_index().drop(columns="replicate")
    save_csv(mean, "q7_summary_mean.csv")
    pd.set_option("display.width", 250)
    print(mean[mean.experiment == "E1_load_sweep"][["design", "offered_erlang", "blocking_pct", "carried_airtime_ch_s", "util_of_occupied_pct", "mean_burst_delay_s", "signalling_msgs"]].round(2).to_string())

    m1 = mean[mean.experiment == "E1_load_sweep"]; m2 = mean[mean.experiment == "E2_duty_sweep"]; m3 = mean[mean.experiment == "E3_rho_sweep"]
    col = {"original": "#e76f51", "redesign": "#2a9d8f"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for d in ["original", "redesign"]:
        s = m1[m1.design == d]
        axs[0, 0].plot(s.offered_erlang, s.blocking_pct, "o-", c=col[d], label=d)
        axs[0, 1].plot(s.offered_erlang, s.carried_airtime_ch_s / T, "o-", c=col[d], label=d)
        axs[0, 2].plot(s.offered_erlang, s.util_of_occupied_pct, "o-", c=col[d], label=d)
        axs[1, 0].plot(s.offered_erlang, s.signalling_msgs, "o-", c=col[d], label=d)
        axs[1, 1].plot(s.offered_erlang, s.mean_burst_delay_s * 1000, "o-", c=col[d], label=d)
    axs[0, 0].set_title("Session blocking (%)"); axs[0, 1].set_title("Carried useful airtime (channel-equivalents)"); axs[0, 2].set_title("Utilisation of occupied channel-time (%)")
    axs[1, 0].set_title("Signalling messages (counted)"); axs[1, 1].set_title("Mean burst delay (ms)")
    for a in axs[0].tolist() + axs[1, :2].tolist(): a.set_xlabel("offered load (Erlang of sessions, 10 % duty)"); a.legend()
    axs[1, 0].set_yscale("log")
    for d in ["original", "redesign"]:
        s = m2[m2.design == d]; axs[1, 2].plot(s.duty * 100, s.blocking_pct, "o-", c=col[d], label=f"{d} blocking")
    axs[1, 2].set_xlabel("duty cycle (%) at 150 Erlang"); axs[1, 2].set_ylabel("blocking (%)"); axs[1, 2].set_title("When is the original preferable? (duty sweep)"); axs[1, 2].legend()
    fig.suptitle("Q7 Original (dedicated channel per session) vs Redesign (shared pool + reservation + admission control), 5 replicates")
    save_fig(fig, "q7_original_vs_redesign.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    r3 = m3[m3.design == "redesign"]
    axs[0].plot(r3.rho_max, r3.mean_burst_delay_s * 1000, "o-", label="mean"); axs[0].plot(r3.rho_max, r3.p95_burst_delay_s * 1000, "s--", label="p95"); axs[0].set_yscale("log")
    axs[0].set_xlabel("admission limit rho_max"); axs[0].set_ylabel("burst delay (ms)"); axs[0].set_title("Trade-off: delay vs admission limit (1200 Erlang offered)"); axs[0].legend()
    axs[1].plot(r3.rho_max, r3.blocking_pct, "o-", c="#2a9d8f"); axs[1].set_xlabel("admission limit rho_max"); axs[1].set_ylabel("session blocking (%)"); axs[1].set_title("Blocking vs admission limit")
    save_fig(fig, "q7_tradeoff_rho.png")

    # headline numbers for the defence section
    o = m1[(m1.design == "original") & (m1.offered_erlang == 300)].iloc[0]; r = m1[(m1.design == "redesign") & (m1.offered_erlang == 300)].iloc[0]
    head = dict(at_300_erlang=dict(blocking_orig=o.blocking_pct, blocking_new=r.blocking_pct, airtime_orig=o.carried_airtime_ch_s / T, airtime_new=r.carried_airtime_ch_s / T,
                                   util_orig=o.util_of_occupied_pct, util_new=r.util_of_occupied_pct, msgs_orig=o.signalling_msgs, msgs_new=r.signalling_msgs,
                                   mean_delay_ms=r.mean_burst_delay_s * 1000))
    d100 = m2[m2.duty == 1.0]
    head["continuous_duty100"] = dict(blocking_orig=float(d100[d100.design == "original"].blocking_pct.iloc[0]), blocking_new=float(d100[d100.design == "redesign"].blocking_pct.iloc[0]))
    save_json(head, "q7_headline.json"); print(json_dump(head))


def json_dump(o):
    import json; return json.dumps(o, indent=1, default=float)


if __name__ == "__main__":
    print("Q7 ..."); run()
