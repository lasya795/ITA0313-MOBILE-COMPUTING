"""B1 - Constraint-aware background architecture (simulation of WorkManager/Service semantics).
Language: Python 3. Time is discrete: 1 tick = 1 minute of device time.
"""
import math, random, json
from dataclasses import dataclass

CRIT, HIGH, NORM = 0, 1, 2
PNAME = {0: 'CRITICAL', 1: 'HIGH', 2: 'NORMAL'}
THROUGHPUT = {0: 0, 1: 20, 2: 100}      # KB per tick for network state NONE/WEAK/GOOD
MAX_WORKERS, MAX_RETRY = 3, 3
IDLE_DRAIN, JOB_DRAIN, CHARGE_GAIN = 0.05, 0.35, 1.0   # battery % per tick
BATT_LOW, BATT_CRIT, WEAK_MAX_KB = 15.0, 5.0, 60

@dataclass
class Item:
    id: int; kind: str; prio: int; size: int; t_enq: int; ttl: int
    state: str = 'NEW'; remaining: float = 0.0; retries: int = 0
    t_done: int = -1; ever_deferred: bool = False; ever_blocked: bool = False

@dataclass
class Env:
    net: list; battery0: float; doze: list; perm: list; charging: list

def feasible(it, t, env, batt):
    """Feasibility / pruning condition (A1.3). Returns (ok, reason)."""
    if not env.perm[t]:                                    return False, 'PERMISSION'
    if env.doze[t] and it.prio != CRIT:                    return False, 'DOZE'
    chg = env.charging[t]
    if batt < BATT_CRIT and not chg and it.prio != CRIT:   return False, 'BATTERY_CRITICAL'
    if batt < BATT_LOW and not chg and it.prio == NORM:    return False, 'BATTERY_LOW'
    n = env.net[t]
    if n == 0:                                             return False, 'NO_NETWORK'
    if n == 1 and it.size > WEAK_MAX_KB and it.prio != CRIT: return False, 'WEAK_NET_LARGE'
    return True, 'OK'

def score(it, t):
    age = t - it.t_enq
    urgency = 30 if (it.ttl - age) < 5 else 0
    return (2 - it.prio) * 100 + min(age, 40) - it.size / 20.0 + urgency   # priority heuristic (B1.3)

def simulate(items, env, T, policy, log=None):
    items = [Item(**vars(i)) for i in items]
    batt = env.battery0
    running, st = [], dict(enq=0, cancelled=0, expired=0, success=0, retries=0, peak=0,
                           wasted_kb=0.0, checks_failed=0)
    for t in range(T):
        for it in items:                                   # arrivals
            if it.t_enq == t:
                st['enq'] += 1; it.state = 'PENDING'; it.remaining = it.size
        if not env.perm[t]:                                # security boundary: revoke -> cancel + purge
            for it in items:
                if it.state in ('RUNNING', 'PENDING'):
                    if it in running: running.remove(it)
                    it.state = 'CANCELLED'; st['cancelled'] += 1
                    if log is not None: log.append((t, it.id, 'CANCEL+PURGE', 'PERMISSION_REVOKED', env.net[t], round(batt, 1)))
        for it in items:                                   # expiry (stale data is dropped)
            if it.state == 'PENDING' and t - it.t_enq > it.ttl:
                it.state = 'EXPIRED'; st['expired'] += 1
                if log is not None: log.append((t, it.id, 'EXPIRE', 'TTL', env.net[t], round(batt, 1)))
        for it in list(running):                           # mid-flight failures
            broke = env.net[t] == 0 or (env.doze[t] and it.prio != CRIT)
            if broke:
                running.remove(it); st['wasted_kb'] += it.size - it.remaining
                it.retries += 1; st['retries'] += 1; it.remaining = it.size
                it.state = 'CANCELLED' if it.retries > MAX_RETRY else 'PENDING'
                if it.state == 'CANCELLED': st['cancelled'] += 1
                if log is not None: log.append((t, it.id, 'FAIL->RETRY' if it.state == 'PENDING' else 'FAIL->CANCEL', 'NET/DOZE', env.net[t], round(batt, 1)))
        pend = [i for i in items if i.state == 'PENDING']
        slots = MAX_WORKERS - len(running)
        if policy == 'FIFO':
            order = sorted(pend, key=lambda i: (i.t_enq, i.id))
            for k, it in enumerate(order):
                if slots == 0: break
                ok, why = feasible(it, t, env, batt)
                if not ok:
                    it.ever_deferred = True; st['checks_failed'] += 1
                    for blocked in order[k + 1:]: blocked.ever_blocked = True      # head-of-line blocking
                    if log is not None: log.append((t, it.id, 'DEFER(HOL)', why, env.net[t], round(batt, 1)))
                    break
                it.state = 'RUNNING'; running.append(it); slots -= 1
                if log is not None: log.append((t, it.id, 'START', 'OK', env.net[t], round(batt, 1)))
        else:
            order = sorted(pend, key=lambda i: -score(i, t))
            for it in order:
                if slots == 0: break
                ok, why = feasible(it, t, env, batt)
                if not ok:
                    it.ever_deferred = True; st['checks_failed'] += 1
                    if log is not None: log.append((t, it.id, 'DEFER', why, env.net[t], round(batt, 1)))
                    continue
                it.state = 'RUNNING'; running.append(it); slots -= 1
                if log is not None: log.append((t, it.id, 'START', 'OK', env.net[t], round(batt, 1)))
        st['peak'] = max(st['peak'], len(running))
        if running:                                        # progress with shared bandwidth
            share = THROUGHPUT[env.net[t]] / len(running)
            for it in list(running):
                it.remaining -= share
                if it.remaining <= 0:
                    it.state = 'DONE'; it.t_done = t; running.remove(it); st['success'] += 1
                    if log is not None: log.append((t, it.id, 'DONE', 'OK', env.net[t], round(batt, 1)))
        batt += (CHARGE_GAIN if env.charging[t] else 0) - IDLE_DRAIN - JOB_DRAIN * len(running)
        batt = max(0.0, min(100.0, batt))
    pending_left = sum(1 for i in items if i.state in ('PENDING', 'RUNNING'))
    crit = [i for i in items if i.prio == CRIT and i.state == 'DONE']
    crit_all = [i for i in items if i.prio == CRIT and i.state != 'NEW']
    lat = sum(i.t_done - i.t_enq for i in crit) / len(crit) if crit else float('nan')
    st.update(deferred_items=sum(1 for i in items if i.ever_deferred),
              blocked_items=sum(1 for i in items if i.ever_blocked and not i.ever_deferred),
              backlog=pending_left, final_batt=round(batt, 1),
              crit_delivered=f'{len(crit)}/{len(crit_all)}', crit_latency=round(lat, 2) if crit else None,
              delivered_kb=sum(i.size for i in items if i.state == 'DONE'),
              all_delivered=(st['success'] == st['enq']))
    return st

def mk_items(spec):
    return [Item(id=i + 1, kind=k, prio=p, size=s, t_enq=te, ttl=tl) for i, (k, p, s, te, tl) in enumerate(spec)]

def scenario_s1():
    T = 40; sizes = [40, 120, 60, 30, 150, 80, 25, 90, 45, 110, 35, 70]
    spec = []
    for i, s in enumerate(sizes):
        spec.append(('UPLOAD', NORM if i % 3 else HIGH, s, i * 2, 40))
    spec += [('ALERT', CRIT, 2, 9, 10), ('ALERT', CRIT, 2, 21, 10)]
    env = Env([2] * T, 90.0, [False] * T, [True] * T, [False] * T)
    return mk_items(spec), env, T

def scenario_s2():
    T = 60; rnd = random.Random(7); spec = []
    for i in range(18):
        spec.append(('UPLOAD', NORM if i % 4 else HIGH, rnd.choice([30, 50, 80, 120, 200]), i * 3, 45))
    spec += [('ALERT', CRIT, 2, 14, 12), ('ALERT', CRIT, 2, 30, 12)]
    net = [2] * 45 + [1] * 15
    doze = [10 <= t < 35 for t in range(T)]
    env = Env(net, 16.0, doze, [True] * T, [False] * T)
    return mk_items(spec), env, T

def scenario_s3():
    T = 40; spec = []
    for i in range(14):
        spec.append(('UPLOAD', NORM if i % 3 else HIGH, 400 if i == 5 else [60, 140, 90, 50][i % 4], i * 2, 40))   # item 6 (400 KB) is mid-transfer at t=12
    spec.append(('ALERT', CRIT, 2, 15, 10))
    perm = [not (12 <= t < 24) for t in range(T)]        # permission revoked at t=12, restored at t=24
    env = Env([2] * T, 80.0, [False] * T, perm, [False] * T)
    return mk_items(spec), env, T

def scenario_s4(seed=11, n=60):
    T = 120; rnd = random.Random(seed); spec = []
    for i in range(n):
        r = rnd.random()
        prio = CRIT if r < 0.10 else (HIGH if r < 0.40 else NORM)
        size = 2 if prio == CRIT else rnd.choice([10, 20, 40, 80, 150, 300])
        ttl = {CRIT: 25, HIGH: 40, NORM: 60}[prio]
        spec.append(('ALERT' if prio == CRIT else 'UPLOAD', prio, size, rnd.randint(0, 85), ttl))
    cyc = [2] * 8 + [1] * 6 + [0] * 4
    net = [cyc[t % len(cyc)] for t in range(T)]
    doze = [(30 <= t < 50) or (90 <= t < 100) for t in range(T)]
    env = Env(net, 60.0, doze, [True] * T, [False] * T)
    return mk_items(spec), env, T

SCEN = {'S1': scenario_s1, 'S2': scenario_s2, 'S3': scenario_s3, 'S4': scenario_s4}

if __name__ == '__main__':
    res = {}
    for name, fn in SCEN.items():
        items, env, T = fn()
        print(f'=== Scenario {name}: {len(items)} items, T={T} ticks ===')
        res[name] = {}
        for pol in ('FIFO', 'PRIORITY'):
            lg = []
            st = simulate(items, env, T, pol, lg)
            res[name][pol] = st
            print(f'[{pol:8}] enq={st["enq"]} success={st["success"]} deferred_items={st["deferred_items"]} '
                  f'blocked_items={st["blocked_items"]} cancelled={st["cancelled"]} expired={st["expired"]} '
                  f'retries={st["retries"]} peak_workers={st["peak"]} backlog={st["backlog"]} '
                  f'crit={st["crit_delivered"]} crit_latency={st["crit_latency"]} wasted_kb={st["wasted_kb"]:.0f} '
                  f'final_batt={st["final_batt"]}')
            if name == 'S2' and pol == 'PRIORITY':
                with open('out/b1_s2_log.txt', 'w') as f:
                    for r in lg: f.write('t=%3d item=%2d %-13s %-17s net=%d batt=%.1f\n' % r)
            if name == 'S3' and pol == 'PRIORITY':
                with open('out/b1_s3_log.txt', 'w') as f:
                    for r in lg: f.write('t=%3d item=%2d %-13s %-17s net=%d batt=%.1f\n' % r)
    json.dump(res, open('out/b1_results.json', 'w'), indent=1)
