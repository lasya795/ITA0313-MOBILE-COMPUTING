"""Q3 - Cellular vs ad hoc: Algorithm 3.1 (SelectWirelessArchitecture) and a
failure-handling demonstration on both architectures."""
from collections import deque


def select_architecture(infra, deploy, coverage, multihop):     # Algorithm 3.1
    if infra == "AVAILABLE" and coverage == "LARGE":
        return "CELLULAR"
    elif deploy == "IMMEDIATE" and infra == "UNAVAILABLE":
        return "AD_HOC"
    elif multihop == "REQUIRED":
        return "AD_HOC"
    return "CELLULAR"


def reachable(links, src, dst, dead=()):
    adj = {}
    for a, b in links:
        if a in dead or b in dead:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, q = {src}, deque([(src, [src])])
    while q:
        n, path = q.popleft()
        if n == dst:
            return path
        for m in adj.get(n, []):
            if m not in seen:
                seen.add(m)
                q.append((m, path + [m]))
    return None


def main():
    print("Algorithm 3.1 - SelectWirelessArchitecture")
    cases = [
        ("Disaster zone", "UNAVAILABLE", "IMMEDIATE", "SMALL", "REQUIRED"),
        ("City mobile service", "AVAILABLE", "PLANNED", "LARGE", "NOT_REQUIRED"),
        ("Battlefield", "UNAVAILABLE", "IMMEDIATE", "MEDIUM", "REQUIRED"),
        ("Campus Wi-Fi", "AVAILABLE", "PLANNED", "SMALL", "NOT_REQUIRED"),
    ]
    for name, infra, dep, cov, mh in cases:
        print("  %-20s infra=%-11s deploy=%-9s coverage=%-6s multihop=%-12s -> %s"
              % (name, infra, dep, cov, mh, select_architecture(infra, dep, cov, mh)))

    print("\nFailure demonstration (MN1 talks to MN2)")
    cell = [("MN1", "BS1"), ("BS1", "CORE"), ("CORE", "BS2"), ("BS2", "MN2")]
    p = reachable(cell, "MN1", "MN2")
    print("  Cellular  normal   : path =", " -> ".join(p))
    p = reachable(cell, "MN1", "MN2", dead={"BS1"})
    print("  Cellular  BS1 fails: path =", " -> ".join(p) if p else "NONE (coverage area lost)")

    adhoc = [("MN1", "N1"), ("N1", "N2"), ("N2", "MN2"), ("MN1", "N3"), ("N3", "N4"), ("N4", "MN2")]
    p = reachable(adhoc, "MN1", "MN2")
    print("  Ad hoc    normal   : path =", " -> ".join(p), "(%d hops)" % (len(p) - 1))
    p = reachable(adhoc, "MN1", "MN2", dead={"N1"})
    print("  Ad hoc    N1 fails : path =", " -> ".join(p), "(%d hops, rerouted)" % (len(p) - 1))

    print("\nPer-hop capacity in a multi-hop ad hoc route (channel shared by neighbours)")
    for k in (1, 2, 3, 4, 5):
        print("  %d hop(s): relative throughput ~ 1/%d = %.2f" % (k, k, 1 / k))

    print("\nDecision for emergency/disaster environment ->",
          select_architecture("UNAVAILABLE", "IMMEDIATE", "SMALL", "REQUIRED"))


if __name__ == "__main__":
    main()
