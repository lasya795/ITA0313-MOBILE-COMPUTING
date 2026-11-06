"""Q6 - AODV (reactive routing) simulation: RREQ flood with duplicate
suppression and reverse routes, RREP along the reverse path, data forwarding,
RERR on link break and re-discovery (RREQ ID=2).  Also SelectRoutingApproach."""
from collections import deque

LINKS = {("S", "A"), ("A", "D"), ("S", "B"), ("B", "C"), ("C", "D")}


def nbrs(n, links):
    return [b if a == n else a for a, b in links if n in (a, b)]


class Net:
    def __init__(self, links):
        self.links = set(links)
        self.route = {n: {} for n in "SABCD"}       # route[n][dest] = (next, hops, state)
        self.reverse = {n: {} for n in "SABCD"}
        self.seen = {n: set() for n in "SABCD"}
        self.msgs = 0

    def discover(self, src, dst, rid, ttl):
        print("  S broadcasts RREQ(id=%d, dst=%s, TTL=%d)" % (rid, dst, ttl))
        q = deque([(src, None, 0, ttl)])
        self.seen[src].add((src, rid))
        found = None
        while q:
            n, prev, hops, t = q.popleft()
            if prev is not None:
                print("   %s received RREQ from %s (hops=%d) -> reverse route to %s via %s"
                      % (n, prev, hops, src, prev))
                self.reverse[n][src] = prev
            if n == dst:
                found = hops
                print("   %s is destination -> sends RREP (hop count 0)" % n)
                break
            if t == 0:
                print("   %s drops RREQ (TTL expired)" % n)
                continue
            for m in nbrs(n, self.links):
                self.msgs += 1
                if (src, rid) in self.seen[m]:
                    if m != prev:
                        print("   %s discards duplicate RREQ(id=%d) from %s" % (m, rid, n))
                    continue
                self.seen[m].add((src, rid))
                q.append((m, n, hops + 1, t - 1))
        if found is None:
            print("  discovery failed")
            return False
        n, h = dst, 0                                  # RREP travels the reverse path
        path = [dst]
        while n != src:
            p = self.reverse[n][src]
            print("   RREP %s -> %s : install forward route to %s (next=%s, hops=%d)"
                  % (n, p, dst, n, h + 1))
            self.route[p][dst] = (n, h + 1, "VALID")
            n, h = p, h + 1
            path.append(n)
        print("  route installed: %s (%d hops)" % ("-".join(reversed(path)), found))
        return True


def select_routing_approach(delay, traffic, bandwidth, mobility, pattern):
    if delay == "VERY_LOW" and traffic == "HIGH":
        return "PROACTIVE"
    if traffic == "LOW" and bandwidth == "LIMITED":
        return "REACTIVE"
    if mobility == "HIGH" and pattern == "INTERMITTENT":
        return "REACTIVE"
    return "EVALUATE_BOTH"


def main():
    net = Net(LINKS)
    print("Step 1-5: S needs a route to D (no valid route) -> route discovery")
    net.discover("S", "D", 1, 3)
    print("\nStep 6: S sends data S -> A -> D (route ACTIVE)")
    print("\nStep 7: link A-D breaks")
    net.links.discard(("A", "D"))
    net.route["A"]["D"] = ("D", 1, "INVALID")
    print("  A marks route to D INVALID and sends RERR(D, seq+1) to S")
    net.route["S"]["D"] = ("A", 2, "INVALID")
    print("\nStep 8: S receives RERR and still has data -> new discovery")
    net.seen = {n: set() for n in "SABCD"}
    net.discover("S", "D", 2, 4)
    print("\nS route table:", net.route["S"])
    print("Total RREQ transmissions simulated:", net.msgs)

    print("\nSelectRoutingApproach")
    for case in (("VERY_LOW", "HIGH", "ENOUGH", "LOW", "CONTINUOUS"),
                 ("NORMAL", "LOW", "LIMITED", "LOW", "BURSTY"),
                 ("NORMAL", "MEDIUM", "ENOUGH", "HIGH", "INTERMITTENT"),
                 ("NORMAL", "MEDIUM", "ENOUGH", "LOW", "CONTINUOUS")):
        print("  delay=%-8s traffic=%-6s bw=%-8s mobility=%-5s pattern=%-12s -> %s"
              % (case + (select_routing_approach(*case),)))

    print("\nHighly dynamic network -> decision: AODV (overhead only for active routes, fresh routes after breaks)")


if __name__ == "__main__":
    main()
