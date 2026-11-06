"""Q4 - Mobile IP for ambulance / mobile healthcare: Algorithm 4.1 and the
performance-requirement calculations."""


def select_application(req):                                    # Algorithm 4.1
    r = req.lower()
    if "continuous internet" in r:
        return "MOBILE_INTERNET_ACCESS"
    elif "emergency vehicle" in r:
        return "EMERGENCY_COMMUNICATION"
    elif "remote patient" in r or "patient vitals" in r:
        return "MOBILE_HEALTHCARE"
    elif "vehicle" in r or "fleet" in r:
        return "TRANSPORTATION"
    return "GENERAL_MOBILE_ACCESS"


def main():
    print("Algorithm 4.1 - SelectMobileIPApplication")
    for r in ("ambulance sending patient vitals", "continuous internet access while moving",
              "emergency vehicle mobility", "fleet tracking vehicles", "laptop at a cafe"):
        print("  %-42s -> %s" % (r, select_application(r)))

    print("\nCommunication delay budget (target <= 150 ms one-way)")
    parts = {"access network": 30, "core network": 30, "HA detour": 40, "tunnel processing": 5}
    total = sum(parts.values())
    for k, v in parts.items():
        print("  %-18s %3d ms" % (k, v))
    print("  %-18s %3d ms -> %s" % ("TOTAL", total, "MEETS target" if total <= 150 else "FAILS target"))
    print("  with triangular routing (+40..80 ms): %d-%d ms -> %s"
          % (total + 40, total + 80, "within target" if total + 80 <= 150 else "may exceed target at the high end"))

    print("\nPacket delivery (target >= 99 %)")
    rate, interruption, handoffs, sent = 20, 0.5, 4, 12000
    per_ho = rate * interruption
    lost = per_ho * handoffs
    print("  loss per handoff = %.1f s x %d pkt/s = %d packets" % (interruption, rate, per_ho))
    print("  %d handoffs in 10 min, %d packets sent -> lost %d (%.2f %%)"
          % (handoffs, sent, lost, 100 * lost / sent))
    print("  delivery ratio = %.2f %% -> %s" % (100 * (sent - lost) / sent,
          "MEETS" if 100 * (sent - lost) / sent >= 99 else "FAILS"))

    print("\nTunnel overhead for vitals packets (IP-in-IP = 20 B)")
    for size in (50, 100, 200, 1000):
        print("  %4d B payload -> %.1f %% overhead" % (size, 2000 / size))

    print("\nInterruption during movement")
    print("  registration 100-300 ms + address acquisition; with pre-registration + buffering < 100 ms")

    print("\nConclusion: Mobile IP suitable for continuous monitoring; add buffering and multi-link fallback for hard real-time control.")


if __name__ == "__main__":
    main()
