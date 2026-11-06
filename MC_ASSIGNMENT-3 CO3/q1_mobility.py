"""Q1 - Wireless network architecture and mobility analysis.
Part A: handoff loss for pedestrian / slow / fast vehicle (cell diameter 200 m).
Part B: 40 s drive AP-1 -> AP-2 -> AP-3 with Algorithm 1.1 (DecideHandoff),
        delivery-ratio table and handoff efficiency.
"""
import math

PKT_RATE = 50          # packets / s (512 B CBR)
HANDOFF_T = 0.15       # s interruption per handoff
CELL_D = 200           # m
MIN_SIGNAL = -80.0     # dBm  (== edge of a 100 m cell with the model below)
HYSTERESIS = 3.0       # dB
DWELL = 0.5            # s


def rss(d):            # simple log-distance path-loss model (dBm)
    return -40 - 20 * math.log10(max(d, 1.0))


def part_a():
    print("PART A - Mobility vs handoff loss (cell = %d m, %d pkt/s, handoff = %d ms)"
          % (CELL_D, PKT_RATE, HANDOFF_T * 1000))
    print("%-14s %-8s %-10s %-12s %-10s %-10s" %
          ("Scenario", "Speed", "t in cell", "Handoffs/min", "Lost/HO", "Loss ratio"))
    for name, v in (("Pedestrian", 1.5), ("Slow vehicle", 15), ("Fast vehicle", 30)):
        t_cell = CELL_D / v
        ho_min = 60 / t_cell
        lost = HANDOFF_T * PKT_RATE
        sent = t_cell * PKT_RATE
        print("%-14s %-8s %-10s %-12.2f %-10.1f %.2f %%" %
              (name, "%g m/s" % v, "%.1f s" % t_cell, ho_min, lost, 100 * lost / sent))


def decide_handoff(state, t, cur, cand):
    """Algorithm 1.1 - returns STAY / HANDOFF_TO_CANDIDATE / CONNECTION_LOST."""
    r_cur, r_new = rss(cur), rss(cand)
    if r_cur < MIN_SIGNAL and r_new < MIN_SIGNAL:
        return "CONNECTION_LOST"
    if r_new > r_cur + HYSTERESIS and r_new >= MIN_SIGNAL:
        if state.get("dwell_start") is None:
            state["dwell_start"] = t                 # START TIMER dwell
        elif t - state["dwell_start"] >= DWELL:
            state["dwell_start"] = None
            return "HANDOFF_TO_CANDIDATE"
    else:
        state["dwell_start"] = None
    return "STAY"


def part_b_simulation(speed=15.0, dt=0.1):
    aps = {"AP-1": 0, "AP-2": 180, "AP-3": 360}      # 100 m radius, 20 m overlap
    names = list(aps)
    cur, t, x = "AP-1", 0.0, 0.0
    state, events = {}, []
    print("\nPART B - Drive AP-1 -> AP-2 -> AP-3 at %g m/s (Algorithm 1.1)" % speed)
    while x <= 360:
        i = names.index(cur)
        if i + 1 < len(names):
            cand = names[i + 1]
            res = decide_handoff(state, t, abs(x - aps[cur]), abs(x - aps[cand]))
            if res == "HANDOFF_TO_CANDIDATE":
                events.append((t, x, cur, cand))
                print("  t=%5.1f s  x=%6.1f m  HANDOFF %s -> %s   (RSS %s=%.1f dBm, %s=%.1f dBm)" %
                      (t, x, cur, cand, cur, rss(abs(x - aps[cur])), cand, rss(abs(x - aps[cand]))))
                cur = cand
        t += dt
        x += speed * dt
    print("  Total handoffs:", len(events))
    return events


def part_b_table():
    print("\nDelivery ratio over 40 s (50 pkt/s -> 2000 packets sent)")
    print("%-22s %-8s %-9s %-14s %-8s %-10s" %
          ("Mobility level", "Speed", "Handoffs", "Interruption", "Lost", "Delivery"))
    for lvl, v, h in (("Low", 1.5, 0), ("Medium", 15, 2), ("High", 30, 4), ("Very high (ping-pong)", 30, 8)):
        intr = h * 0.15
        lost = round(intr * PKT_RATE)
        print("%-22s %-8s %-9d %-14s %-8d %.2f %%" %
              (lvl, "%g m/s" % v, h, "%.2f s" % intr, lost, 100 * (2000 - lost) / 2000))


def efficiency():
    print("\nHandoff efficiency = 1 - (handoffs/min x interruption)/60  at 4.5 handoffs/min")
    for label, t in (("layer-2 handoff only (0.15 s)", 0.15), ("with Mobile IP registration (0.65 s)", 0.65)):
        print("  %-40s %.1f %%" % (label, 100 * (1 - 4.5 * t / 60)))


if __name__ == "__main__":
    part_a()
    part_b_simulation()
    part_b_table()
    efficiency()
