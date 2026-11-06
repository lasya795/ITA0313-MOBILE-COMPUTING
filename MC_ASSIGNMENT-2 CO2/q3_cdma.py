"""Q3 - Direct-sequence CDMA uplink simulation (BPSK, AWGN) with Walsh vs random codes,
power imbalance, asynchronism and chip-timing jitter. Lengths 8, 16, 64."""
import math
from common import *

EBN0_DB = 10.0       # thermal noise level (kept low so errors are dominated by multiple-access interference)
NBITS = 2000         # bits per user per trial
TRIALS = 8           # independent draws of codes/timing/data per (L, K, condition) point -> 16000 bits per user
BER_TARGET = 1e-2    # our definition of acceptable voice-grade BER


def hadamard(n):
    H = np.array([[1]])
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H


def qfunc(x): return 0.5 * math.erfc(x / math.sqrt(2))


def simulate(L, K, code_type, timing, power_ratio, rg, nbits=NBITS, ebn0_db=EBN0_DB):
    """timing: 'sync' | 'jitter' (+-1 chip) | 'async' (random 0..L-1 chip offset).
    power_ratio: power of user 0 relative to others (1 = equal). Returns BER over the *other* users
    (all users when power_ratio == 1), mean desired correlation, interference std."""
    codes = hadamard(L)[:K] if code_type == "walsh" else rg.choice([-1, 1], size=(K, L))
    bits = rg.choice([-1, 1], size=(K, nbits))
    amp = np.ones(K); amp[0] = math.sqrt(power_ratio)
    off = {"sync": np.zeros(K, int), "jitter": rg.integers(-1, 2, K), "async": rg.integers(0, L, K)}[timing]
    sig = np.zeros(nbits * L)
    for k in range(K):
        s = (bits[k][:, None] * codes[k][None, :]).ravel() * amp[k]
        sig += np.roll(s, off[k]) if off[k] else s
    ebn0 = 10 ** (ebn0_db / 10)
    r = sig + math.sqrt(L / (2 * ebn0)) * rg.standard_normal(sig.size)
    users = range(1, K) if power_ratio != 1 and K > 1 else range(K)
    errs = tot = 0; corr = []; istd = []
    for j in users:
        rj = np.roll(r, -off[j]) if off[j] else r
        z = rj.reshape(nbits, L) @ codes[j] / L          # despread (correlate) and normalise by processing gain
        dec = np.where(z >= 0, 1, -1)
        errs += int((dec != bits[j]).sum()); tot += nbits
        corr.append(float((z * bits[j]).mean())); istd.append(float((z - bits[j] * amp[j]).std()))
    return errs / tot, tot, float(np.mean(corr)), float(np.mean(istd)), codes


def max_cross_corr(codes):
    L = codes.shape[1]; c = np.abs(codes @ codes.T / L); np.fill_diagonal(c, 0); return float(c.max()), float(c.sum() / max(1, c.size - len(c)))


CONDS = [  # name, code_type, timing, power_ratio
    ("A Walsh, sync, equal power", "walsh", "sync", 1),
    ("B Random codes, sync, equal power", "random", "sync", 1),
    ("C Walsh, sync, user0 10x power", "walsh", "sync", 10),
    ("D Walsh, async (random chip offsets)", "walsh", "async", 1),
    ("E Walsh, +-1 chip timing jitter", "walsh", "jitter", 1),
    ("F Walsh, jitter, user0 10x power", "walsh", "jitter", 10),
]
KGRID = {8: [1, 2, 3, 4, 5, 6, 7, 8], 16: [1, 2, 4, 6, 8, 10, 12, 14, 16], 64: [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]}


def run():
    rows = []
    for L, ks in KGRID.items():
        for ci, (cname, ct, tm, pr) in enumerate(CONDS):
            for K in ks:
                if pr != 1 and K < 2: continue
                res = [simulate(L, K, ct, tm, pr, rng(3000 + L * 10 + ci * 1000 + K * 7 + t)) for t in range(TRIALS)]
                tot = sum(r[1] for r in res); ber = sum(r[0] * r[1] for r in res) / tot
                corr = float(np.mean([r[2] for r in res])); istd = float(np.mean([r[3] for r in res]))
                mx, mean_cc = max_cross_corr(res[0][4])
                rows.append(dict(L=L, K=K, condition=cname, bits_tested=tot, bit_errors=round(ber * tot), BER=ber,
                                 processing_gain=L, processing_gain_dB=round(10 * math.log10(L), 2),
                                 mean_desired_correlation=corr, interference_std=istd,
                                 max_abs_cross_corr_zero_lag=mx, mean_abs_cross_corr_zero_lag=mean_cc,
                                 theory_gauss_random_BER=qfunc(1 / math.sqrt((K - 1) / L + 1 / (2 * 10 ** (EBN0_DB / 10)))) if K > 0 else float("nan")))
    df = pd.DataFrame(rows); save_csv(df, "q3_cdma_results.csv")

    # --- BER vs users figure
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), sharey=True)
    for ax, L in zip(axs, KGRID):
        for cname, *_ in CONDS:
            d = df[(df.L == L) & (df.condition == cname)]
            floor = 0.5 / d.bits_tested.max()
            ax.plot(d.K, np.maximum(d.BER, floor), "o-", ms=4, label=cname)
        d = df[(df.L == L) & (df.condition == CONDS[1][0])]
        ax.plot(d.K, np.maximum(d.theory_gauss_random_BER, 1e-6), "k:", label="Gaussian theory (random)")
        ax.axhline(BER_TARGET, color="gray", ls="--", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("simultaneous users K"); ax.set_title(f"Code length L = {L} (processing gain {10*math.log10(L):.1f} dB)")
    axs[0].set_ylabel(f"BER ({NBITS*TRIALS} bits/user; floor = 0 errors)"); axs[0].legend(fontsize=6)
    fig.suptitle(f"Q3 DS-CDMA uplink, Eb/N0 = {EBN0_DB:.0f} dB; dashed = BER target {BER_TARGET}")
    save_fig(fig, "q3_ber_vs_users.png")

    # --- supported users table
    sup = []
    for L in KGRID:
        for cname, *_ in CONDS:
            d = df[(df.L == L) & (df.condition == cname) & (df.BER <= BER_TARGET)]
            sup.append(dict(L=L, condition=cname, max_users_meeting_target=int(d.K.max()) if len(d) else 0,
                            BER_target=BER_TARGET, grid_resolution_note="largest K on tested grid with BER<=target"))
    sup = pd.DataFrame(sup); save_csv(sup, "q3_supported_users.csv")

    # --- power control figure
    pr_list = [1, 2, 4, 10, 30, 100]; rows2 = []
    L, K = 64, 32
    for tm in ["sync", "jitter", "async"]:
        for pr in pr_list:
            rs = [simulate(L, K, "walsh", tm, pr, rng(9000 + pr * 3 + len(tm) + t)) for t in range(TRIALS)]; tot = sum(r[1] for r in rs); ber = sum(r[0] * r[1] for r in rs) / tot
            rows2.append(dict(L=L, K=K, timing=tm, near_user_power_ratio=pr, power_dB=10 * math.log10(pr), BER_other_users=ber, bits_tested=tot))
    pc = pd.DataFrame(rows2); save_csv(pc, "q3_power_control.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for tm in ["sync", "jitter", "async"]:
        d = pc[pc.timing == tm]; ax.plot(d.power_dB, np.maximum(d.BER_other_users, 0.5 / d.bits_tested.max()), "o-", label=f"Walsh, {tm}")
    ax.set_yscale("log"); ax.set_xlabel("near user power advantage (dB)"); ax.set_ylabel("BER of the other users")
    ax.set_title("Q3 near-far: L=64, K=32 users"); ax.legend()
    save_fig(fig, "q3_power_control.png")

    # --- correlation matrices
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    W = hadamard(16)[:16]; R = rng(7).choice([-1, 1], size=(16, 16))
    for ax, M, t in zip(axs[:2], [W, R], ["Walsh L=16 (zero lag)", "Random L=16 (zero lag)"]):
        im = ax.imshow(M @ M.T / 16, vmin=-1, vmax=1, cmap="coolwarm"); ax.set_title(t); ax.grid(False)
    plt.colorbar(im, ax=axs[:2], fraction=0.025)
    lags = np.arange(-8, 9); cc = [abs(np.dot(W[3], np.roll(W[5], l))) / 16 for l in lags]
    axs[2].stem(lags, cc); axs[2].set_xlabel("chip lag"); axs[2].set_ylabel("|cross-correlation|")
    axs[2].set_title("Walsh rows 3 vs 5: orthogonal only at lag 0")
    save_fig(fig, "q3_correlations.png")
    return df, sup


if __name__ == "__main__":
    print("Q3 ...")
    df, sup = run(); print(sup.to_string())
    pick = lambda c: int(sup[(sup.L == 64) & (sup.condition.str.startswith(c))].max_users_meeting_target.values[0])
    p = pd.read_csv(os.path.join(DATA, "q3_power_control.csv"))
    save_json(dict(users_walsh_sync_L64=pick("A"), users_random_L64=pick("B"), users_walsh_jitter_L64=pick("E"),
                   users_walsh_jitter_10x_L64=pick("F"),
                   ber_jitter_30dB_near=float(p[(p.timing == "jitter") & (p.near_user_power_ratio == 100)].BER_other_users.values[0])),
              "q3_summary.json")
