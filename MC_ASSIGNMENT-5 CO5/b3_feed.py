"""B3 - Exact vs adaptive (greedy) optimisation for a simplified scrolling-feed renderer. Python 3 + numpy."""
import numpy as np, json, random, itertools, time

CACHES = [0, 8, 16, 32, 64, 128]       # cache size in items (index = c_idx)
MAXP = 4                              # prefetch window 0..4 items
FRAME_BUDGET = 16.0                   # ms

class Device:
    def __init__(self, name, speed, mem_mb, battery):
        self.name, self.speed, self.mem, self.battery = name, speed, mem_mb, battery

def make_feed(n, seed):
    rnd = np.random.RandomState(seed)
    base = rnd.uniform(0.6, 1.8, n)                       # decode complexity (videos higher)
    base[rnd.rand(n) < 0.25] *= 1.6                       # 25 % short videos
    return base

def evaluate(cfg, feed, dev, Q, cache_levels=CACHES):
    """Cost model. Returns dict(cost, drops, energy, avg_ft, mem)."""
    q, ci, p = cfg
    r = (q + 1) / Q
    c = cache_levels[ci]
    hit = c / (c + 24.0)                                   # cache hit probability
    decode = feed * (0.4 + 3.2 * r * r) / dev.speed        # ms to decode at quality q
    ft = 3.0 + decode * (1 - hit) * (1 - 0.12 * p)         # 3 ms layout + residual decode on the UI path
    drops = int((ft > FRAME_BUDGET).sum())
    energy = float((decode * (1 - hit) * 0.9 * (1 + 0.15 * p)).sum() + 0.02 * c * len(feed))   # mJ
    mem = c * (0.5 + 4.0 * r * r) + p * (0.5 + 4.0 * r * r)                                    # MB
    mem_pen = 1000.0 * max(0.0, mem - dev.mem)
    qloss = 0.5 * (1 - r) * len(feed)
    we = 0.05 if dev.battery < 20 else 0.02             # energy weighs more on a low battery
    cost = 10.0 * drops + we * energy + qloss + mem_pen
    return dict(cost=cost, drops=drops, energy=energy, avg_ft=float(ft.mean()), mem=mem)

def exact(feed, dev, Q, cache_levels=CACHES):
    best, bcfg, n = None, None, 0
    for q in range(Q):
        for ci in range(len(cache_levels)):
            for p in range(MAXP + 1):
                n += 1
                e = evaluate((q, ci, p), feed, dev, Q, cache_levels)
                if best is None or e['cost'] < best['cost']: best, bcfg = e, (q, ci, p)
    return bcfg, best, n

def adaptive(feed, dev, Q, cache_levels=CACHES, trace=None, max_iter=200):
    """Greedy / adaptive controller (A5.2): at every step measure the current frame time / energy,
    try the single-step moves q+-1, cache+-1, prefetch+-1, adopt the move with the lowest measured cost,
    and stop at steady state (no single move improves the cost)."""
    q, ci, p = Q - 1, 0, 0                                 # start: best quality, no cache, no prefetch
    cur = evaluate((q, ci, p), feed, dev, Q, cache_levels); evals, qchanges = 1, 0
    for it in range(max_iter):
        best, bcfg = None, None
        for dq, dc, dp in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nq, nc, np_ = q + dq, ci + dc, p + dp
            if not (0 <= nq < Q and 0 <= nc < len(cache_levels) and 0 <= np_ <= MAXP): continue
            m = evaluate((nq, nc, np_), feed, dev, Q, cache_levels); evals += 1
            if best is None or m['cost'] < best['cost']: best, bcfg = m, (nq, nc, np_)
        if trace is not None:
            trace.append((it, (q, ci, p), round(cur['avg_ft'], 2), cur['drops'], round(cur['cost'], 1), bcfg, round(best['cost'], 1)))
        if best['cost'] >= cur['cost'] - 1e-12: break       # steady state: local optimum reached
        if bcfg[0] != q: qchanges += 1
        (q, ci, p), cur = bcfg, best
    return (q, ci, p), cur, evals, qchanges

def lru_evictions(cache_size, n_items, seed=5):
    """Count evictions of an LRU cache on a scroll trace with 30 % back-scrolling."""
    rnd = random.Random(seed); order = []
    pos = 0
    while pos < n_items:
        order.append(pos)
        pos = pos - rnd.randint(1, 3) if (rnd.random() < 0.3 and pos > 3) else pos + 1
    cache, ev = [], 0
    if cache_size == 0: return len(order)
    for x in order:
        if x in cache: cache.remove(x); cache.append(x)
        else:
            cache.append(x)
            if len(cache) > cache_size: cache.pop(0); ev += 1
    return ev

def fmt(cfg): return f'(q={cfg[0]}, cache={CACHES[cfg[1]]}, prefetch={cfg[2]})'

if __name__ == '__main__':
    out = {}
    print('=== B3.1 small instances (Q=4 quality levels, 6 cache sizes, 5 prefetch values = 120 configs) ===')
    insts = [('I1 low-end, 40 items',  Device('low-end', 0.5, 40, 80), 40, 101),
             ('I2 mid-range, 40 items', Device('mid', 1.0, 60, 70), 40, 202),
             ('I3 flagship, 40 items',  Device('flagship', 1.8, 120, 90), 40, 303),
             ('I4 low-end low-battery', Device('low-end/LB', 0.5, 40, 15), 40, 404),
             ('I5 tight memory mid',    Device('mid/tightmem', 1.0, 12, 60), 40, 505)]
    rows = []
    for nm, dev, n, sd in insts:
        feed = make_feed(n, sd)
        ec, em, ne = exact(feed, dev, 4); ac, am, na, qc = adaptive(feed, dev, 4)
        match = abs(em['cost'] - am['cost']) < 1e-9
        gap = (am['cost'] - em['cost']) / em['cost'] * 100
        rows.append(dict(name=nm, exact_cfg=fmt(ec), exact_cost=round(em['cost'], 2), adapt_cfg=fmt(ac),
                         adapt_cost=round(am['cost'], 2), match=match, gap=round(gap, 1)))
        print(f'{nm:26} EXACT {fmt(ec):38} cost={em["cost"]:8.2f} | ADAPT {fmt(ac):38} cost={am["cost"]:8.2f} | match={match} gap={gap:5.1f}%')
    out['small'] = rows

    print('\n=== B3.2 counter-example search (adaptive vs exact, 400 random small instances) ===')
    rnd = random.Random(1); worst = None; mism = 0; tot = 0
    for k in range(400):
        dev = Device('rand', rnd.choice([0.4, 0.5, 0.7, 1.0, 1.4, 1.8]), rnd.choice([10, 20, 40, 80, 120]), rnd.choice([10, 15, 30, 60, 90]))
        n = rnd.choice([30, 40, 50]); sd = rnd.randint(1, 10 ** 6)
        feed = make_feed(n, sd)
        ec, em, _ = exact(feed, dev, 4); ac, am, _, _ = adaptive(feed, dev, 4)
        tot += 1
        if am['cost'] > em['cost'] + 1e-9:
            mism += 1; g = (am['cost'] - em['cost']) / em['cost']
            if worst is None or g > worst[0]: worst = (g, dev, n, sd, ec, ac, em, am)
    print(f'adaptive matched exact on {tot - mism}/{tot} instances; mismatches={mism}')
    g, dev, n, sd, ec, ac, em, am = worst
    print(f'WORST: device speed={dev.speed} mem={dev.mem}MB battery={dev.battery}% items={n} seed={sd}  gap={g * 100:.1f}%')
    print(f'  exact   {fmt(ec)} cost={em["cost"]:.2f} drops={em["drops"]} avg_ft={em["avg_ft"]:.2f}ms mem={em["mem"]:.1f}MB')
    print(f'  adaptive{fmt(ac)} cost={am["cost"]:.2f} drops={am["drops"]} avg_ft={am["avg_ft"]:.2f}ms mem={am["mem"]:.1f}MB')
    feed = make_feed(n, sd); tr = []; adaptive(feed, dev, 4, trace=tr)
    print('  adaptive trace (iter, current cfg, avg frame ms, drops, cost -> best neighbour, its cost):')
    for r in tr: print('   ', r[0], fmt(r[1]), r[2], r[3], r[4], '->', fmt(r[5]), r[6])
    out['counter'] = dict(match_rate=f'{tot - mism}/{tot}', speed=dev.speed, mem=dev.mem, battery=dev.battery, n=n, seed=sd,
                          gap=round(g * 100, 1), exact_cfg=fmt(ec), exact_cost=round(em['cost'], 2), exact_drops=em['drops'],
                          adapt_cfg=fmt(ac), adapt_cost=round(am['cost'], 2), adapt_drops=am['drops'],
                          trace=[(r[0], fmt(r[1]), r[2], r[3], r[4], fmt(r[5]), r[6]) for r in tr])

    print('\n=== B3.3 scaling (operation counts) ===')
    print(f'{"items":>6} {"Q":>3} {"configs":>8} | {"exact evals":>11} {"frame-evals":>12} {"evict":>6} | {"adapt evals":>11} {"frame-evals":>12} {"qchg":>4} {"evict":>6} | {"gap%":>6} {"t_exact(s)":>10} {"t_adapt(s)":>10}')
    dev = Device('mid', 1.0, 60, 70); sc = []
    big_cache = [0, 8, 16, 32, 64, 128, 256, 512]
    for n, Q in [(100, 4), (500, 8), (2000, 16), (5000, 32), (10000, 32)]:
        feed = make_feed(n, 7)
        t0 = time.time(); ec, em, ne = exact(feed, dev, Q, big_cache); te = time.time() - t0
        t0 = time.time(); ac, am, na, qc = adaptive(feed, dev, Q, big_cache); ta = time.time() - t0
        ev_e = lru_evictions(big_cache[ec[1]], n); ev_a = lru_evictions(big_cache[ac[1]], n)
        gap = (am['cost'] - em['cost']) / abs(em['cost']) * 100
        sc.append(dict(n=n, Q=Q, configs=ne, exact_evals=ne, exact_frame_evals=ne * n, evict_exact=ev_e, adapt_evals=na,
                       adapt_frame_evals=na * n, qchg=qc, evict_adapt=ev_a, gap=round(gap, 1), t_exact=round(te, 2), t_adapt=round(ta, 4),
                       exact_cfg=fmt(ec), adapt_cfg=fmt(ac), exact_cost=round(em['cost'], 1), adapt_cost=round(am['cost'], 1)))
        print(f'{n:6} {Q:3} {ne:8} | {ne:11} {ne * n:12} {ev_e:6} | {na:11} {na * n:12} {qc:4} {ev_a:6} | {gap:6.1f} {te:10.2f} {ta:10.4f}')
    out['scale'] = sc
    json.dump(out, open('out/b3_results.json', 'w'), indent=1)
