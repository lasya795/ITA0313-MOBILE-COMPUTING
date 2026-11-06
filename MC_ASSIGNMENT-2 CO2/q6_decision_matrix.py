"""Q6 - Weighted decision matrix for four deployments + +-20 % sensitivity check.
Ratings (1-5) are traceable to Q2-Q5 evidence: evidence strings are filled at run time from data/q*_summary.json."""
import itertools
from common import *

TECH = ["SDMA", "TDMA", "FDMA", "CDMA", "OFDMA"]
CRIT = ["capacity_continuous", "bursty_small_report_efficiency", "interference_jam_resistance",
        "simplicity_sync_powercontrol", "per_user_QoS_fairness", "scalability_dense_users"]

# rating[tech][criterion], 1 (poor) .. 5 (excellent)
R = {
 "FDMA":  [3, 1, 2, 5, 4, 2],
 "TDMA":  [3, 2, 2, 3, 4, 3],
 "CDMA":  [4, 4, 5, 2, 3, 4],
 "OFDMA": [5, 4, 3, 3, 3, 4],
 "SDMA":  [4, 3, 4, 2, 2, 2],
}
# weights per deployment (sum = 1) and justification
DEPLOY = {
 "1 Wide-area cellular voice": ([.30, .05, .10, .20, .20, .15],
     "Continuous voice -> capacity dominates; per-user QoS (call quality) and implementation cost matter; bursty efficiency barely matters."),
 "2 Jam-resistant tactical link": ([.10, .05, .45, .10, .10, .20],
     "Mission is surviving interference/jamming; raw capacity is secondary; need many nodes -> scalability."),
 "3 High-rate stadium downlink": ([.35, .10, .05, .10, .20, .20],
     "Maximum spectral efficiency for dense users, each seat needs a fair rate; jamming not a concern."),
 "4 5000 low-rate sensors": ([.05, .35, .05, .30, .05, .20],
     "Tiny bursty reports -> efficiency under bursty traffic; cheap/low-power devices -> simplicity (no tight sync, no fast power control); many devices."),
}


def evidence():
    ev = {}
    try:
        q2, q3, q4, q5 = (load_json(f"q{i}_summary.json") for i in (2, 3, 4, 5))
        ev["FDMA/bursty"] = f"Q2: FDMA held-channel utilisation under 10% duty = {q2['bursty_util_fdma']:.1f}% -> rating 1"
        ev["TDMA/bursty"] = f"Q2: TDMA held-channel utilisation under 10% duty = {q2['bursty_util_tdma']:.1f}% (worse overhead) -> rating 2"
        ev["FDMA/capacity"] = f"Q2: useful utilisation (voice, % of nominal) FDMA {q2['voice_util_fdma']:.0f}% vs TDMA {q2['voice_util_tdma']:.0f}% -> FDMA/TDMA both 3"
        ev["CDMA/simplicity"] = (f"Q3: L=64 supports {q3['users_walsh_sync_L64']} users if ideally sync/equal power, but only {q3['users_walsh_jitter_L64']} "
                                 f"with +-1 chip jitter and {q3['users_random_L64']} with random codes -> needs sync + power control -> rating 2; "
                                 f"BER(100x near user, jitter) = {q3['ber_jitter_30dB_near']:.2f}")
        ev["OFDMA/fairness"] = (f"Q4: best-subcarrier Jain={q4['jain']['best_subcarrier']:.2f}, outage {q4['outage']['best_subcarrier']:.0f}% vs "
                                f"with guarantee Jain={q4['jain']['best_with_min_guarantee']:.2f}, outage {q4['outage']['best_with_min_guarantee']:.0f}% -> rating 3 (needs scheduler)")
        ev["OFDMA/capacity"] = f"Q4: channel-aware sum throughput {q4['sum']['best_subcarrier']:.0f} vs interleaved {q4['sum']['interleaved_round_robin']:.0f} bits/symbol -> rating 5"
        ev["SDMA/scalability"] = (f"Q5: N=16 separates a pair only {q5['p_sep_toll_N16']*100:.0f}% of the time at toll-plaza clustering vs "
                                  f"{q5['p_sep_highway_N16']*100:.0f}% on a highway; min separation N=8 = {q5['min_sep_N8_deg']:.0f} deg -> rating 2")
    except FileNotFoundError:
        ev["note"] = "run q2..q5 first to fill evidence strings"
    return ev


def score(w): return {t: float(np.dot(w, R[t])) for t in TECH}


def run():
    ev = evidence()
    pd.DataFrame([dict(technique=t, **dict(zip(CRIT, R[t]))) for t in TECH]).to_csv(os.path.join(DATA, "q6_ratings.csv"), index=False)
    pd.DataFrame([dict(key=k, evidence=v) for k, v in ev.items()]).to_csv(os.path.join(DATA, "q6_evidence_trace.csv"), index=False)
    pd.DataFrame([dict(deployment=d, **dict(zip(CRIT, w)), justification=j) for d, (w, j) in DEPLOY.items()]).to_csv(os.path.join(DATA, "q6_weights.csv"), index=False)
    rows, sens = [], []
    signs = list(itertools.product([0.8, 1.2], repeat=len(CRIT)))      # 64 combinations of +-20 % on every weight
    for d, (w, _) in DEPLOY.items():
        s = score(np.array(w)); order = sorted(s, key=s.get, reverse=True)
        rows.append(dict(deployment=d, **{t: round(s[t], 3) for t in TECH}, winner=order[0], runner_up=order[1], margin=round(s[order[0]] - s[order[1]], 3)))
        wins = {t: 0 for t in TECH}
        for sg in signs:
            w2 = np.array(w) * np.array(sg); w2 /= w2.sum(); s2 = score(w2); wins[max(s2, key=s2.get)] += 1
        wins50 = 0
        for sg in itertools.product([0.5, 1.5], repeat=len(CRIT)):          # stress test: +-50 %
            w2 = np.array(w) * np.array(sg); w2 /= w2.sum(); s2 = score(w2); wins50 += max(s2, key=s2.get) == order[0]
        sens.append(dict(deployment=d, baseline_winner=order[0], stress_pm50_winner_stable_pct=100 * wins50 / 64, **{f"wins_{t}_of_64": wins[t] for t in TECH},
                         winner_stable_pct=100 * wins[order[0]] / len(signs), choice_changes=wins[order[0]] < len(signs)))
    HYB = {"1 Wide-area cellular voice": ("OFDMA (with sector SDMA)", "FDMA carriers x TDMA slots (GSM-style) if legacy/voice-only; OFDMA+sectorised SDMA for new build"),
           "2 Jam-resistant tactical link": ("CDMA (direct sequence)", "CDMA + frequency hopping (FDMA channels) and SDMA null-steering toward jammers"),
           "3 High-rate stadium downlink": ("OFDMA", "OFDMA + SDMA beamforming sectors, with the min-rate-guarantee scheduler from Q4"),
           "4 5000 low-rate sensors": ("OFDMA (narrow sub-bands, random access)", "FDMA narrow-band channels + slotted random access (TDMA-like), shared not dedicated (Q2/Q7)")}
    pd.DataFrame([dict(deployment=d, primary=a, hybrid=b) for d, (a, b) in HYB.items()]).to_csv(os.path.join(DATA, "q6_choices.csv"), index=False)
    sc = pd.DataFrame(rows); se = pd.DataFrame(sens)
    save_csv(sc, "q6_scores.csv"); save_csv(se, "q6_sensitivity.csv")
    print(sc.to_string()); print(se.to_string())
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(len(sc)); w = 0.16
    for i, t in enumerate(TECH): axs[0].bar(x + (i - 2) * w, sc[t], w, label=t)
    axs[0].set_xticks(x); axs[0].set_xticklabels([d[:22] for d in sc.deployment], fontsize=7); axs[0].set_ylabel("weighted score (1-5)"); axs[0].set_title("Q6 weighted scores"); axs[0].legend(ncol=5, fontsize=7); axs[0].set_ylim(1, 5)
    bottom = np.zeros(len(se))
    for t in TECH:
        v = se[f"wins_{t}_of_64"].values / 64 * 100; axs[1].bar(range(len(se)), v, bottom=bottom, label=t); bottom += v
    axs[1].set_xticks(range(len(se))); axs[1].set_xticklabels([d[:22] for d in se.deployment], fontsize=7); axs[1].set_ylabel("% of 64 weight perturbations won")
    axs[1].set_title("Sensitivity: every weight x0.8 or x1.2 (64 combinations)"); axs[1].legend(ncol=5, fontsize=7)
    save_fig(fig, "q6_scores_and_sensitivity.png")
    return sc, se


if __name__ == "__main__":
    print("Q6 ..."); run()
