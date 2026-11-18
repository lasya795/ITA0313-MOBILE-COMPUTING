"""B4 - Heuristic selection of resolution / bitrate / upload timing for short video clips. Python 3."""
import random, itertools, json

CLIP_S = 15.0
RES = {360: 0.23, 480: 0.41, 720: 0.92, 1080: 2.07}          # megapixels per frame-second factor (relative pixel count, 720p ~ 0.92 MP)
BITRATES = [0.8, 1.5, 3.0, 6.0]                                # Mbps
TIMING = ['NOW', 'WAIT_WIFI', 'WAIT_CHARGING']
P_RADIO = {'wifi': 0.8, '4g': 2.0, '3g': 1.6}                  # W while transferring
QUALITY = {  # (resolution, bitrate) -> perceptual quality 0..100 (illustrative lookup)
    (360, 0.8): 55, (360, 1.5): 58, (360, 3.0): 59, (360, 6.0): 59,
    (480, 0.8): 62, (480, 1.5): 70, (480, 3.0): 73, (480, 6.0): 74,
    (720, 0.8): 60, (720, 1.5): 74, (720, 3.0): 85, (720, 6.0): 88,
    (1080, 0.8): 52, (1080, 1.5): 70, (1080, 3.0): 86, (1080, 6.0): 95}

class Dev:
    def __init__(self, name, cam, cpu, batt, cap_mah, net, mbps, charging=False, wifi_soon=False):
        self.name, self.cam, self.cpu, self.batt, self.cap, self.net, self.mbps = name, cam, cpu, batt, cap_mah, net, mbps
        self.charging, self.wifi_soon = charging, wifi_soon

def estimate(d, res, br, timing):
    enc_t = CLIP_S * RES[res] / (0.92 * d.cpu)                   # seconds to encode (cpu=1.0 -> real-time at 720p)
    enc_e = enc_t * (1.0 + 1.2 * d.cpu)                          # joules (faster SoC burns more power)
    size_mb = br * CLIP_S / 8
    net, mbps = d.net, d.mbps
    if timing == 'WAIT_WIFI':
        if not d.wifi_soon and d.net != 'wifi': return None      # infeasible: no WiFi expected
        net, mbps = 'wifi', 20.0
    up_t = size_mb * 8 / max(mbps, 0.05)
    up_e = up_t * P_RADIO[net]
    if timing == 'WAIT_CHARGING' and not d.charging: return None
    e_j = enc_e + up_e
    wait = {'NOW': 0, 'WAIT_WIFI': 600, 'WAIT_CHARGING': 1800}[timing]
    pct = e_j / (d.cap / 1000 * 3.85 * 3600) * 100 * (0.1 if timing == 'WAIT_CHARGING' else 1.0)  # charging: energy not drawn from battery
    return dict(enc_t=enc_t, up_t=up_t, size_mb=size_mb, energy_j=e_j, battery_pct=pct, delay_s=enc_t + up_t + wait, quality=QUALITY[(res, br)])

def utility(d, e):
    """Objective used by the exhaustive search: maximise quality, penalise energy (more when battery is low) and delay."""
    w_batt = 4.0 if d.batt < 20 and not d.charging else 1.0
    return e['quality'] - w_batt * 6.0 * e['battery_pct'] * 100 / 10 - 0.02 * e['delay_s']

def exhaustive(d):
    best, bcfg, n = None, None, 0
    for res, br, tm in itertools.product([r for r in RES if r <= d.cam], BITRATES, TIMING):
        e = estimate(d, res, br, tm); n += 1
        if e is None or e['enc_t'] > 3 * CLIP_S: continue
        u = utility(d, e)
        if best is None or u > best[0]: best, bcfg = (u, e), (res, br, tm)
    return bcfg, best[1], best[0], n

def heuristic(d):
    """Selection rule (B4.1):
    1. resolution  = highest camera-supported resolution whose encode time <= 2 x clip length;
    2. bitrate     = highest bitrate whose upload time <= 30 s on the current link (cap 3 Mbps for <=480p);
    3. timing      = NOW normally; if battery < 20% and not charging -> WAIT_WIFI when WiFi expected, one resolution level lower;
                     if charging -> NOW."""
    res_ok = [r for r in sorted(RES) if r <= d.cam and CLIP_S * RES[r] / (0.92 * d.cpu) <= 2.0 * CLIP_S]
    res = res_ok[-1] if res_ok else min(RES)
    low = d.batt < 20 and not d.charging
    if low and res_ok and len(res_ok) > 1: res = res_ok[-2]
    br = BITRATES[0]
    for b in BITRATES:
        if b * CLIP_S / d.mbps <= 30 and (res > 480 or b <= 3.0): br = b
    tm = 'WAIT_WIFI' if (low and d.wifi_soon) else 'NOW'
    return (res, br, tm), estimate(d, res, br, tm)

DEVICES = [Dev('D1 flagship, WiFi', 2160, 2.0, 85, 5000, 'wifi', 40),
           Dev('D2 mid-range, 4G', 1080, 1.0, 60, 4000, '4g', 8),
           Dev('D3 low-end, 3G', 720, 0.45, 55, 3000, '3g', 1.2),
           Dev('D4 flagship, 4G, battery 12%', 2160, 2.0, 12, 4500, '4g', 12, wifi_soon=True),
           Dev('D5 mid-range, charging, 4G', 1080, 1.0, 40, 4000, '4g', 10, charging=True),
           Dev('D6 low-end, 3G, battery 9%', 720, 0.45, 9, 3000, '3g', 0.8, wifi_soon=False),
           Dev('D7 old phone, weak 3G', 480, 0.3, 70, 2500, '3g', 0.4),
           Dev('D8 mid-range, WiFi, 18%', 1080, 1.0, 18, 4000, 'wifi', 30)]

if __name__ == '__main__':
    out = {'devices': []}
    print(f'{"Device":32} {"Heuristic choice":28} {"enc(s)":>6} {"up(s)":>6} {"E(J)":>6} {"batt%":>7} | {"Exhaustive choice":28} {"E(J)":>6}')
    same = 0
    for d in DEVICES:
        hc, he = heuristic(d); ec, ee, eu, n = exhaustive(d)
        hu = utility(d, he)
        same += hc == ec
        print(f'{d.name:32} {str(hc):28} {he["enc_t"]:6.1f} {he["up_t"]:6.1f} {he["energy_j"]:6.1f} {he["battery_pct"]:7.4f} | {str(ec):28} {ee["energy_j"]:6.1f}')
        out['devices'].append(dict(name=d.name, h=str(hc), enc=round(he['enc_t'], 1), up=round(he['up_t'], 1), e=round(he['energy_j'], 1),
                                   batt=round(he['battery_pct'], 4), size=round(he['size_mb'], 2), q=he['quality'], ex=str(ec), ex_e=round(ee['energy_j'], 1),
                                   h_u=round(hu, 1), ex_u=round(eu, 1), delay=round(he['delay_s'])))
    print(f'heuristic identical to exhaustive on {same}/{len(DEVICES)} sample devices; exhaustive search size = {n} candidate configurations per device')
    # random population comparison
    rnd = random.Random(2026); N = 2000; match = 0; gaps = []; worst = None
    for k in range(N):
        d = Dev('r', rnd.choice([480, 720, 1080, 2160]), rnd.choice([0.3, 0.45, 0.7, 1.0, 1.5, 2.0]), rnd.choice([8, 12, 18, 25, 40, 60, 90]),
                rnd.choice([2500, 3000, 4000, 5000]), rnd.choice(['wifi', '4g', '3g']), rnd.choice([0.4, 0.8, 1.2, 4, 8, 12, 30]),
                charging=rnd.random() < 0.15, wifi_soon=rnd.random() < 0.4)
        hc, he = heuristic(d); ec, ee, eu, _ = exhaustive(d)
        if he is None or he['enc_t'] > 3 * CLIP_S: gaps.append(None); continue
        hu = utility(d, he)
        if hc == ec: match += 1
        g = eu - hu; gaps.append(g)
        if worst is None or g > worst[0]: worst = (g, d, hc, he, ec, ee, eu, hu)
    vals = [g for g in gaps if g is not None]
    print(f'\nRandom population of {N} devices: identical choice on {match} ({100 * match / N:.1f}%), mean utility gap {sum(vals) / len(vals):.2f}, max gap {max(vals):.2f}')
    g, d, hc, he, ec, ee, eu, hu = worst
    print(f'WORST CASE device: cam={d.cam} cpu={d.cpu} batt={d.batt}% net={d.net} {d.mbps}Mbps charging={d.charging} wifi_soon={d.wifi_soon}')
    print(f'  heuristic  {hc} utility={hu:.1f} E={he["energy_j"]:.1f}J batt={he["battery_pct"]:.4f}% quality={he["quality"]} delay={he["delay_s"]:.0f}s')
    print(f'  exhaustive {ec} utility={eu:.1f} E={ee["energy_j"]:.1f}J batt={ee["battery_pct"]:.4f}% quality={ee["quality"]} delay={ee["delay_s"]:.0f}s')
    out['pop'] = dict(N=N, match=match, pct=round(100 * match / N, 1), mean_gap=round(sum(vals) / len(vals), 2), max_gap=round(max(vals), 2))
    out['worst'] = dict(dev=dict(cam=d.cam, cpu=d.cpu, batt=d.batt, net=d.net, mbps=d.mbps, charging=d.charging, wifi_soon=d.wifi_soon),
                        h=str(hc), h_u=round(hu, 1), h_e=round(he['energy_j'], 1), h_q=he['quality'], h_delay=round(he['delay_s']),
                        ex=str(ec), ex_u=round(eu, 1), ex_e=round(ee['energy_j'], 1), ex_q=ee['quality'], ex_delay=round(ee['delay_s']))
    print('\nExhaustive search cost at scale: 48 configs x 1,000,000 devices = 48,000,000 model evaluations per decision round; heuristic: 1 pass per device.')
    import time
    rt = random.Random(5); devs = [Dev('r', rt.choice([480, 720, 1080, 2160]), rt.choice([0.3, 0.45, 0.7, 1.0, 1.5, 2.0]), rt.choice([8, 12, 18, 25, 40, 60, 90]),
                                      rt.choice([2500, 3000, 4000, 5000]), rt.choice(['wifi', '4g', '3g']), rt.choice([0.4, 0.8, 1.2, 4, 8, 12, 30])) for _ in range(5000)]
    t0 = time.time(); [heuristic(x) for x in devs]; th = time.time() - t0
    t0 = time.time(); [exhaustive(x) for x in devs]; te = time.time() - t0
    print(f'Timing over 5000 devices: heuristic {th / 5000 * 1e6:.1f} us/device, exhaustive {te / 5000 * 1e6:.1f} us/device (ratio {te / th:.1f}x)')
    out['timing'] = dict(h_us=round(th / 5000 * 1e6, 1), e_us=round(te / 5000 * 1e6, 1), ratio=round(te / th, 1))
    json.dump(out, open('out/b4_results.json', 'w'), indent=1)
