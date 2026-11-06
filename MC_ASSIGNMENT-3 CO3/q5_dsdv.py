"""Q5 - DSDV (proactive routing) simulation on the sample topology
A-B, B-C, C-D, A-E, E-D.  Reproduces A's routing table before / after the
E-D link break (Algorithm 5.1) and the control-overhead model."""
INF = 10 ** 6
LINKS = {("A", "B"), ("B", "C"), ("C", "D"), ("A", "E"), ("E", "D")}
SEQ0 = {"A": 100, "B": 128, "C": 220, "D": 406, "E": 316}      # even = destination-generated


class Node:
    def __init__(self, name):
        self.name = name
        self.table = {name: dict(next=name, metric=0, seq=SEQ0[name])}

    def update_route(self, dest, hop, seq, neighbor):          # UpdateDSDVRoute
        new_hop = min(hop + 1, INF)
        cur = self.table.get(dest)
        if cur is None or seq > cur["seq"] or (seq == cur["seq"] and new_hop < cur["metric"]):
            self.table[dest] = dict(next=neighbor, metric=new_hop, seq=seq)
            return True
        return False


def neighbors(links, n):
    return [b if a == n else a for a, b in links if n in (a, b)]


def exchange(nodes, links):
    """One round: every node sends its full table to its neighbours."""
    changed = False
    snap = {n: {d: dict(r) for d, r in nodes[n].table.items()} for n in nodes}
    for n in nodes:
        for nb in neighbors(links, n):
            for dest, r in snap[nb].items():
                if dest != n:
                    changed |= nodes[n].update_route(dest, r["metric"], r["seq"], nb)
    return changed


def show(node, title):
    print(title)
    print("  %-12s %-9s %-7s %-10s" % ("Destination", "Next hop", "Metric", "Seq No."))
    for d in sorted(node.table):
        r = node.table[d]
        m = "INF" if r["metric"] >= INF else r["metric"]
        print("  %-12s %-9s %-7s %s-%d" % (d, r["next"], m, d, r["seq"]))
    print()


def handle_link_break(node, failed):                            # HandleLinkBreak
    for d, r in node.table.items():
        if r["next"] == failed and d != node.name:
            r["metric"], r["seq"] = INF, r["seq"] + 1            # odd sequence number
            print("  %s: route to %s via %s INVALID -> metric INF, seq %s-%d (odd)"
                  % (node.name, d, failed, d, r["seq"]))


def main():
    nodes = {n: Node(n) for n in "ABCDE"}
    rounds = 0
    while exchange(nodes, LINKS):
        rounds += 1
    print("Converged after %d update rounds.\n" % rounds)
    show(nodes["A"], "A's routing table BEFORE topology change:")

    print("Topology change: link E-D breaks")
    links = LINKS - {("E", "D")}
    handle_link_break(nodes["E"], "D")
    print()
    for n in nodes:                                              # new advertisement period
        nodes[n].table[n]["seq"] += 2
    r = 0
    while exchange(nodes, links):
        r += 1
    print("Re-converged after %d propagation rounds (D advertises new even seq D-408 via C->B->A)\n" % r)
    show(nodes["A"], "A's routing table AFTER the change:")

    print("Control-overhead model: full dump of N entries x 12 B every T = 15 s  -> O(N^2)")
    print("  %-6s %-16s %-22s %-18s" % ("N", "Per-node (B)", "Network per period (B)", "Overhead (B/s)"))
    for N in (10, 50, 100):
        per = N * 12
        print("  %-6d %-16d %-22d %-18.0f" % (N, per, N * per, N * per / 15))
    print("\nWorst-case convergence ~ diameter x interval = 4 x 15 = 60 s (triggered updates: a few seconds)")


if __name__ == "__main__":
    main()
