"""Q2 - Mobile IP simulation: agent discovery, registration (Algorithm 2.1),
tunnelling (Algorithm 2.2), overhead and handoff-loss figures."""
import random

random.seed(7)
HOME_ADDR, HA_ADDR, FA_ADDR = "10.1.1.50", "10.1.1.1", "172.16.2.1"
CN_ADDR = "192.168.5.20"


class HomeAgent:
    def __init__(self):
        self.bindings = {}                      # home addr -> (CoA, expiry time)

    def register(self, home, coa, lifetime, now):
        self.bindings[home] = (coa, now + lifetime)
        return "ACCEPTED"

    def deregister(self, home):
        self.bindings.pop(home, None)

    def tunnel(self, pkt, now):                 # Algorithm 2.2
        b = self.bindings.get(pkt["dst"])
        if not b or now > b[1]:
            return None, "DELIVERY_FAILURE (no/expired binding - packet dropped)"
        outer = {"src": HA_ADDR, "dst": b[0], "len": pkt["len"] + 20, "inner": pkt}
        return outer, "DELIVERY_SUCCESS"


def location_update(ha, current_net, now, loss_prob=0.5, max_retry=3):   # Algorithm 2.1
    if current_net == "HOME":
        ha.deregister(HOME_ADDR)
        return HOME_ADDR, "HOME"
    coa, retries = FA_ADDR, 0
    while True:
        print("   t=%.1f  RegistrationRequest(home=%s, CoA=%s, lifetime=60, id=%d) -> HA"
              % (now, HOME_ADDR, coa, retries + 1))
        if random.random() > loss_prob:                         # reply received
            status = ha.register(HOME_ADDR, coa, 60, now)
            print("   t=%.1f  RegistrationReply: %s" % (now + 0.2, status))
            return coa, "REGISTERED"
        now += 1.0                                              # regTimer expires
        if retries < max_retry:
            retries += 1
            print("   regTimer expired -> retry %d" % retries)
        else:
            return coa, "FAILED"


def main():
    ha = HomeAgent()
    print("1) Agent discovery: FA broadcasts Agent Advertisement; MN sees a foreign network")
    print("2) Registration (lossy link, retries allowed)")
    coa, status = location_update(ha, "FOREIGN", 0.0)
    print("   -> CoA=%s status=%s ; HA binding table: %s\n" % (coa, status, ha.bindings))

    print("3) Tunnelling of a 1000-byte packet CN -> MN")
    pkt = {"src": CN_ADDR, "dst": HOME_ADDR, "len": 1000}
    outer, res = ha.tunnel(pkt, now=5.0)
    print("   original : %s -> %s (%d B)" % (pkt["src"], pkt["dst"], pkt["len"]))
    print("   tunnelled: %s -> %s (%d B)  [%s]" % (outer["src"], outer["dst"], outer["len"], res))
    print("   FA decapsulates and delivers %d B packet to MN\n" % outer["inner"]["len"])

    print("4) Tunnel overhead = 20 / payload")
    for size, name in ((1000, "1000 B data"), (200, "200 B"), (100, "100 B VoIP")):
        print("   %-12s overhead = %.0f %%" % (name, 20 * 100 / size))

    print("\n5) Binding expiry")
    outer, res = ha.tunnel(pkt, now=65.0)
    print("   packet at t=65 s (lifetime 60 s): %s" % res)

    print("\n6) Challenge figures")
    print("   packets lost while HA not yet updated (0.5 s x 50 pkt/s) = %d" % (0.5 * 50))
    print("   triangular routing: direct 20 ms vs via HA 80 ms -> delay x%.0f" % (80 / 20))
    print("   registration delay: domestic RTT 100 ms + auth -> ~100-300 ms")

    print("\n7) MN returns home")
    coa, status = location_update(ha, "HOME", 100.0)
    print("   CoA=%s status=%s ; bindings=%s" % (coa, status, ha.bindings))


if __name__ == "__main__":
    main()
