"""Q4 - OFDMA: orthogonality, cyclic prefix, and 3(+1) subcarrier-allocation schemes.
8 users, 64 subcarriers, frequency-selective Rayleigh fading, 1000 realisations, fixed seed."""
from common import *

N_SC, N_USERS, N_REAL = 64, 8, 1000
USER_SNR_DB = np.linspace(22, 6, N_USERS)    # average SNR per user (near -> far in the hall)
R_MIN = 10.0                                  # min rate in bits per OFDM symbol (justified in README)
DF = 15e3                                     # subcarrier spacing (assumed, LTE-like) -> Tsym = 1/DF = 66.7 us


# ------------------------------------------------------------ Part 1: orthogonality + cyclic prefix
def orthogonality():
    T = 1 / DF; ovs = 16; t = np.arange(N_SC * ovs) / (N_SC * ovs) * T
    ks = [3, 4, 5, 10, 3.5]                  # 3.5 = non-integer (non-orthogonal) control
    X = np.array([np.exp(2j * np.pi * k * DF * t) for k in ks])
    G = (X.conj() @ X.T) / X.shape[1]        # (1/T) * integral of x_i^* x_j over one symbol
    df = pd.DataFrame(np.abs(G), index=[f"k={k}" for k in ks], columns=[f"k={k}" for k in ks])
    df.to_csv(os.path.join(DATA, "q4_orthogonality_inner_products.csv")); print("  csv    -> q4_orthogonality_inner_products.csv")
    # FFT check: modulate subcarriers 3,4,5 with different symbols and recover exactly
    Xf = np.zeros(N_SC, complex); Xf[[3, 4, 5]] = [1 + 1j, -1 + 1j, 1 - 1j]
    rec = np.fft.fft(np.fft.ifft(Xf))
    fft_err = float(np.abs(rec - Xf).max())
    return df, fft_err


def cp_experiment(cp=16, nsym=2000):
    rg = rng(4001)
    h = np.zeros(8, complex); h[[0, 3, 6]] = [1, 0.6, 0.35]; h /= np.linalg.norm(h)       # delay spread 6 samples < CP
    H = np.fft.fft(h, N_SC)
    q = (rg.choice([-1, 1], (nsym, N_SC)) + 1j * rg.choice([-1, 1], (nsym, N_SC))) / np.sqrt(2)
    td = np.fft.ifft(q, axis=1) * np.sqrt(N_SC)
    out = {}
    for name, c in [("with_CP", cp), ("no_CP", 0)]:
        blk = np.concatenate([td[:, -c:], td], axis=1) if c else td
        rx = np.convolve(blk.ravel(), h)[:blk.size].reshape(nsym, N_SC + c)[:, c:]
        Y = np.fft.fft(rx, axis=1) / np.sqrt(N_SC) / H
        out[name] = Y
    res = []
    for name, Y in out.items():
        evm = np.sqrt(np.mean(np.abs(Y - q) ** 2) / np.mean(np.abs(q) ** 2))
        ser = np.mean((np.sign(Y.real) != np.sign(q.real)) | (np.sign(Y.imag) != np.sign(q.imag)))
        res.append(dict(case=name, cp_samples=cp if name == "with_CP" else 0, EVM=evm, QPSK_symbol_error_rate=ser,
                        cp_overhead_pct=100 * cp / (N_SC + cp) if name == "with_CP" else 0))
    return pd.DataFrame(res), out


# ------------------------------------------------------------ Part 2: allocation
def channel_gains(n_users, snr_db, nreal, rg, taps=8, decay=0.6):
    pdp = decay ** np.arange(taps); pdp /= pdp.sum()
    h = (rg.standard_normal((nreal, n_users, taps)) + 1j * rg.standard_normal((nreal, n_users, taps))) / np.sqrt(2) * np.sqrt(pdp)
    H = np.fft.fft(h, N_SC, axis=2)
    return np.abs(H) ** 2 * (10 ** (np.asarray(snr_db) / 10))[None, :, None]     # (R,U,N) linear SNR


def rate_of(g): return np.log2(1 + g)


def per_user(rate, owner, U):
    sel = np.take_along_axis(rate, owner[:, None, :], axis=1)[:, 0, :]
    return (sel[:, :, None] * (owner[:, :, None] == np.arange(U))).sum(1)


def alloc(rate, scheme, rmin=R_MIN):
    R, U, N = rate.shape
    if scheme == "fixed_contiguous": return np.broadcast_to(np.arange(N) // (N // U), (R, N)).copy()
    if scheme == "interleaved_round_robin": return np.broadcast_to(np.arange(N) % U, (R, N)).copy()
    if scheme == "best_subcarrier": return rate.argmax(axis=1)
    if scheme == "best_with_min_guarantee":            # greedy: lift the poorest user to rmin first, then max-sum for the rest
        owner = -np.ones((R, N), int)
        for r in range(R):
            cur = np.zeros(U); free = np.ones(N, bool)
            while free.any():
                u = int(cur.argmin())
                if cur[u] >= rmin: break
                idx = np.where(free)[0]; k = idx[rate[r, u, idx].argmax()]
                owner[r, k] = u; cur[u] += rate[r, u, k]; free[k] = False
            idx = np.where(free)[0]
            owner[r, idx] = rate[r][:, idx].argmax(axis=0)
        return owner
    raise ValueError(scheme)


def jain(x): x = np.asarray(x, float); return float(x.sum() ** 2 / (len(x) * (x ** 2).sum())) if x.sum() > 0 else 0.0


def run():
    inner, fft_err = orthogonality(); print("  FFT recovery error (3 subcarriers):", fft_err)
    cpdf, cons = cp_experiment(); save_csv(cpdf, "q4_cyclic_prefix.csv"); print(cpdf.to_string())
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    im = axs[0].imshow(inner.values, cmap="viridis", vmin=0, vmax=1); axs[0].set_xticks(range(5)); axs[0].set_xticklabels(inner.columns, fontsize=7)
    axs[0].set_yticks(range(5)); axs[0].set_yticklabels(inner.index, fontsize=7); axs[0].grid(False)
    axs[0].set_title("|inner product| over one symbol T=1/Δf\n(k=3.5 is a non-integer control)")
    for i in range(5):
        for j in range(5): axs[0].text(j, i, f"{inner.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    for ax, key in zip(axs[1:], ["with_CP", "no_CP"]):
        Y = cons[key][:200].ravel(); ax.scatter(Y.real, Y.imag, s=2, alpha=.4); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ev = cpdf[cpdf.case == key].EVM.values[0]; ax.set_title(f"QPSK after FFT + 1-tap equaliser: {key} (EVM={ev:.3f})"); ax.set_aspect("equal")
    save_fig(fig, "q4_orthogonality_and_cp.png")

    rg = rng(4100)
    g = channel_gains(N_USERS, USER_SNR_DB, N_REAL, rg); rate = rate_of(g)
    schemes = ["fixed_contiguous", "interleaved_round_robin", "best_subcarrier", "best_with_min_guarantee"]
    rows = []; thr_store = {}
    for s in schemes:
        thr = per_user(rate, alloc(rate, s), N_USERS); thr_store[s] = thr
        mean_u = thr.mean(0)
        inst_j = np.mean([jain(t) for t in thr])
        row = dict(scheme=s, realisations=N_REAL, sum_throughput_mean=thr.sum(1).mean(), jain_longterm=jain(mean_u), jain_instant_mean=inst_j,
                   min_user_mean_throughput=mean_u.min(), prob_user_below_Rmin_pct=100 * (thr < R_MIN).mean(),
                   prob_all_users_meet_Rmin_pct=100 * (thr >= R_MIN).all(1).mean())
        row.update({f"user{u+1}_mean_thr": mean_u[u] for u in range(N_USERS)}); rows.append(row)
    res = pd.DataFrame(rows); save_csv(res, "q4_allocation_results.csv")
    print(res.round(3).iloc[:, :9].to_string())

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    xs = np.arange(N_USERS); w = 0.2
    for i, s in enumerate(schemes): axs[0].bar(xs + (i - 1.5) * w, thr_store[s].mean(0), w, label=s)
    axs[0].axhline(R_MIN, color="k", ls="--", lw=1, label=f"R_min = {R_MIN}"); axs[0].set_xticks(xs); axs[0].set_xticklabels([f"U{u+1}\n{USER_SNR_DB[u]:.0f} dB" for u in xs], fontsize=7)
    axs[0].set_ylabel("mean throughput (bits / OFDM symbol)"); axs[0].set_title("Per-user throughput (1000 realisations)"); axs[0].legend(fontsize=6)
    axs[1].bar(range(4), res.sum_throughput_mean, color="#457b9d"); axs[1].set_xticks(range(4)); axs[1].set_xticklabels(["fixed", "interleaved", "best", "best+min"], fontsize=8)
    axs[1].set_ylabel("sum throughput"); axs[1].set_title("Sum throughput"); a2 = axs[1].twinx(); a2.plot(range(4), res.jain_longterm, "ro-"); a2.set_ylabel("Jain index (red)"); a2.set_ylim(0, 1); a2.grid(False)
    axs[2].bar(range(4), res.prob_user_below_Rmin_pct, color="#e76f51"); axs[2].set_xticks(range(4)); axs[2].set_xticklabels(["fixed", "interleaved", "best", "best+min"], fontsize=8)
    axs[2].set_ylabel("% of (user, realisation) below R_min"); axs[2].set_title(f"Minimum-rate outage (R_min={R_MIN})")
    save_fig(fig, "q4_allocation_results.png")

    # multiuser diversity vs number of users (equal average SNR 15 dB)
    mu = []
    for U in [1, 2, 4, 8, 16, 32]:
        r_ = rate_of(channel_gains(U, [15] * U, N_REAL, rng(4200 + U)))
        a = per_user(r_, alloc(r_, "interleaved_round_robin"), U).sum(1).mean(); b = per_user(r_, alloc(r_, "best_subcarrier"), U).sum(1).mean()
        fb = per_user(r_, alloc(r_, "best_subcarrier"), U).mean(0)
        mu.append(dict(users=U, sum_interleaved=a, sum_best=b, gain_pct=100 * (b / a - 1), jain_best_longterm=jain(fb)))
    mu = pd.DataFrame(mu); save_csv(mu, "q4_multiuser_diversity.csv")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(mu.users, mu.gain_pct, "o-"); ax.set_xscale("log", base=2)
    ax.set_xlabel("number of users sharing 64 subcarriers"); ax.set_ylabel("gain of best-subcarrier over interleaved (%)")
    ax.set_title("Q4 multiuser diversity (equal mean SNR 15 dB)"); save_fig(fig, "q4_multiuser_diversity.png")
    save_json(dict(jain=dict(zip(res.scheme, res.jain_longterm)), min_user=dict(zip(res.scheme, res.min_user_mean_throughput)),
                   outage=dict(zip(res.scheme, res.prob_user_below_Rmin_pct)), sum=dict(zip(res.scheme, res.sum_throughput_mean))), "q4_summary.json")
    return res, mu


if __name__ == "__main__":
    print("Q4 ..."); res, mu = run(); print(mu.round(3).to_string())
