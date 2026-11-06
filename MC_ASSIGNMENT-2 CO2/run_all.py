"""Run every experiment in order (Q6 needs the Q2-Q5 summaries). Total runtime: a few minutes."""
import time
import q1_mobile_vs_computing as q1, q2_fdma_tdma as q2, q3_cdma as q3, q4_ofdma as q4, q5_sdma as q5, q6_decision_matrix as q6, q7_redesign as q7
from common import *

t0 = time.time()
print("Q1"); q1.part_a(); q1.part_b()
print("Q2"); s = q2.run_datasets(); q2.telemetry(); c = q2.capacity_table()
save_json(dict(C_fdma=int(c.loc[0, "usable_traffic_channels"]), C_tdma=int(c.loc[1, "usable_traffic_channels"]),
               bursty_util_fdma=float(s[(s.dataset == "B_bursty") & (s.scheme == "FDMA")].useful_util_of_held_pct.values[0]),
               bursty_util_tdma=float(s[(s.dataset == "B_bursty") & (s.scheme == "TDMA")].useful_util_of_held_pct.values[0]),
               voice_util_fdma=float(s[(s.dataset == "A_voice") & (s.scheme == "FDMA")].useful_util_of_nominal_pct.values[0]),
               voice_util_tdma=float(s[(s.dataset == "A_voice") & (s.scheme == "TDMA")].useful_util_of_nominal_pct.values[0])), "q2_summary.json")
print("Q3"); df, sup = q3.run()
p = pd.read_csv(os.path.join(DATA, "q3_power_control.csv"))
pick = lambda k: int(sup[(sup.L == 64) & (sup.condition.str.startswith(k))].max_users_meeting_target.values[0])
save_json(dict(users_walsh_sync_L64=pick("A"), users_random_L64=pick("B"), users_walsh_jitter_L64=pick("E"), users_walsh_jitter_10x_L64=pick("F"),
               ber_jitter_30dB_near=float(p[(p.timing == "jitter") & (p.near_user_power_ratio == 100)].BER_other_users.values[0])), "q3_summary.json")
print("Q4"); q4.run()
print("Q5"); q5.run()
print("Q6"); q6.run()
print("Q7"); q7.run()
print(f"Done in {time.time()-t0:.0f} s. Figures in ./figures, CSV in ./data")
