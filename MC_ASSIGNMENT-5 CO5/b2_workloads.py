"""B2 - Two workloads, same size (N=40) and same constraint thresholds, run on the B1 scheduler (PRIORITY policy)."""
import random, json
from b1_background import *

def workload_A(n=40, T=100, seed=3):
    """Easy: small records, steady good network, healthy battery, no Doze."""
    rnd = random.Random(seed); spec = []
    for i in range(n):
        prio = HIGH if i % 4 == 0 else NORM
        spec.append(('UPLOAD', prio, rnd.choice([20, 30, 40, 60]), i * 2, 60))
    env = Env([2] * T, 85.0, [False] * T, [True] * T, [False] * T)
    return mk_items(spec), env, T

def workload_B(n=40, T=100, seed=3):
    """Hard: large records, flapping network (3 good / 2 none), battery near the thresholds, two Doze windows."""
    rnd = random.Random(seed); spec = []
    for i in range(n):
        prio = HIGH if i % 4 == 0 else NORM
        spec.append(('UPLOAD', prio, rnd.choice([80, 120, 160, 200]), i * 2, 60))
    cyc = [2, 2, 2, 0, 0]
    net = [cyc[t % 5] for t in range(T)]
    doze = [(20 <= t < 40) or (60 <= t < 75) for t in range(T)]
    env = Env(net, 30.0, doze, [True] * T, [False] * T)
    return mk_items(spec), env, T

if __name__ == '__main__':
    out = {}
    for nm, fn in (('A', workload_A), ('B', workload_B)):
        items, env, T = fn()
        st = simulate(items, env, T, 'PRIORITY')
        out[nm] = st
        total_kb = sum(i.size for i in items)
        print(f'Workload {nm}: N={len(items)} total_kb={total_kb} T={T}')
        print(f'  success={st["success"]} deferred_items={st["deferred_items"]} checks_failed={st["checks_failed"]} '
              f'retries={st["retries"]} wasted_kb={st["wasted_kb"]:.0f} expired={st["expired"]} cancelled={st["cancelled"]} '
              f'backlog={st["backlog"]} final_batt={st["final_batt"]} delivered_kb={st["delivered_kb"]}')
        out[nm]['total_kb'] = total_kb
    json.dump(out, open('out/b2_results.json', 'w'), indent=1)
