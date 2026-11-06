"""Q7 - Mobile emergency communication network (hybrid design).
Ad hoc zone routed with AODV-style on-demand discovery, Mobile IP between the
zone gateways (FA) and the EOC Home Agent, with failure handling
(Algorithm 7.1 - EmergencyForward)."""
import math
from collections import deque

RANGE = 130.0
NODES = {                                   # x, y (m)
    "AMB1": (0, 0), "POL1": (100, 40), "RES1": (200, 0), "MED1": (300, 60),
    "GW1": (400, 0), "GW2": (320, 140), "RES2": (200, 110),
}
GATEWAYS = ["GW1", "GW2"]
HOME = {"AMB1": "10.10.0.5", "POL1": "10.10.0.6", "RES1": "10.10.0.7",
        "MED1": "10.10.0.8", "RES2": "10.10.0.9"}


class Zone:
    def __init__(self):
        self.pos = dict(NODES)
        self.dead = set()
        self.bindings = {}                  # EOC Home Agent: home addr -> (CoA gateway, expiry)
        self.ha_up = True
        self.buffer = []

    def links(self):
        names = [n for n in self.pos if n not in self.dead]
        return {n: [m for m in names if m != n and math.dist(self.pos[n], self.pos[m]) <= RANGE]
                for n in names}

    def route(self, src, dst):              # on-demand route discovery (RREQ/RREP result)
        adj, q, seen = self.links(), deque([[src]]), {src}
        while q:
            p = q.popleft()
            if p[-1] == dst:
                return p
            for m in adj.get(p[-1], []):
                if m not in seen:
                    seen.add(m)
                    q.append(p + [m])
        return None

    def nearest_gateway(self, mn):
        best = None
        for g in GATEWAYS:
            if g in self.dead:
                continue
            r = self.route(mn, g)
            if r and (best is None or len(r) < len(best)):
                best = r
        return best

    def register(self, mn, now):
        r = self.nearest_gateway(mn)
        if not r:
            print("   %s cannot reach any gateway -> no registration (buffer data)" % mn)
            return None
        if not self.ha_up:
            print("   HA unreachable (registration timeout) -> use redundant HA / continue intra-zone only")
            return "HA_DOWN"
        gw = r[-1]
        self.bindings[HOME[mn]] = (gw, now + 60)
        print("   %s registers via %s : RegReq(home=%s, CoA=%s, life=60 s) path %s -> HA binding updated"
              % (mn, gw, HOME[mn], gw, "-".join(r)))
        return gw

    def forward(self, src, dst, now):       # Algorithm 7.1
        if dst == "EOC":
            r = self.nearest_gateway(src)
            if r:
                print("   %s -> EOC : AODV to %s then reverse tunnel (%d hops in zone)"
                      % (src, r[-1], len(r) - 1))
                return "DELIVERED_VIA_GATEWAY"
            self.buffer.append((src, dst))
            print("   %s -> EOC : no route -> store-and-carry-forward (buffered, %d queued)"
                  % (src, len(self.buffer)))
            return "BUFFERED"
        r = self.route(src, dst)
        if r:
            print("   %s -> %s : intra-zone AODV direct, no Mobile IP (%s)" % (src, dst, "-".join(r)))
            return "DELIVERED_DIRECT"
        self.buffer.append((src, dst))
        print("   %s -> %s : partition -> buffered" % (src, dst))
        return "BUFFERED"

    def from_eoc(self, mn, now):
        b = self.bindings.get(HOME[mn])
        if not b or b[1] < now:
            print("   EOC -> %s : binding missing/expired -> DROP" % mn)
            return "FAIL"
        gw = b[0]
        r = self.route(gw, mn) if gw not in self.dead else None
        if not r:
            print("   EOC -> %s : tunnel to %s but no in-zone route -> buffered" % (mn, gw))
            return "BUFFERED"
        print("   EOC -> %s : HA tunnels (20 B header) to %s, then AODV %s" % (mn, gw, "-".join(r)))
        return "DELIVERED"


def main():
    z = Zone()
    print("[1] Mobile nodes join the zone and register through the nearest gateway")
    for mn in HOME:
        z.register(mn, 0)
    print("\n[2] Traffic")
    z.forward("AMB1", "POL1", 1)
    z.forward("AMB1", "EOC", 1)
    z.from_eoc("AMB1", 1)
    print("\n[3] Failure: gateway GW1 goes down -> switch to GW2 and re-register")
    z.dead.add("GW1")
    for mn in ("MED1", "AMB1"):
        z.register(mn, 20)
    z.from_eoc("MED1", 21)
    print("\n[4] Failure: node RES1 fails -> reroute / rediscover")
    z.dead.add("RES1")
    z.forward("AMB1", "MED1", 30)
    z.forward("AMB1", "EOC", 30)
    print("\n[5] Partition: POL1 fails as well -> store-and-carry-forward")
    z.dead.add("POL1")
    z.forward("AMB1", "EOC", 40)
    print("   vehicle repositions to bridge the gap: AMB1 moves to (110, 60)")
    z.pos["AMB1"] = (110, 60)
    z.forward("AMB1", "EOC", 45)
    print("\n[6] HA unreachable")
    z.ha_up = False
    z.register("RES2", 50)
    z.forward("RES2", "MED1", 50)
    print("\n[7] Binding lifetime (60 s): MN must re-register at 50 % (30 s) or the binding expires")
    z.ha_up = True
    z.from_eoc("AMB1", 100)             # last registration was at t=20 -> expired
    z.register("AMB1", 100)             # re-registration
    z.from_eoc("AMB1", 101)

    print("\nPerformance parameters (design targets): control delay < 500 ms, data delay < 150 ms,")
    print("tunnel overhead 20 B/packet, AODV discovery 50-200 ms, HELLO interval 1 s, route lifetime 10 s")


if __name__ == "__main__":
    main()
