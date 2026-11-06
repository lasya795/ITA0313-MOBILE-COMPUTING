"""Q5 - SDMA with a uniform linear array (ULA).
Array factor (d = lambda/2, angle theta from broadside, beam steered at theta0):
    AF(theta) = sum_{n=0}^{N-1} exp(j n psi),   psi = 2*pi*(d/lambda)*(sin theta - sin theta0)
    |AF|/N    = | sin(N psi / 2) / (N sin(psi / 2)) |
Main-lobe: first nulls at sin(theta)-sin(theta0) = +-lambda/(N d); HPBW ~ 0.886*lambda/(N d cos theta0) rad
Largest side-lobe -> -13.26 dB for large N (sinc-like), i.e. independent of N."""
from common import *

D = 0.5                      # element spacing in wavelengths
NS = [2, 4, 8, 16]
SIR_THR_DB = [10, 15, 20]    # candidate SIR thresholds; 10 dB is the primary one (assumed)
PRIMARY = 10   # primary SIR threshold: coded QPSK works at ~10 dB; a plain ULA side-lobe floor of ~-13 dB makes >13 dB unattainable without null steering


def af_power(theta_deg, theta0_deg, N):
    psi = 2 * np.pi * D * (np.sin(np.radians(theta_deg)) - np.sin(np.radians(theta0_deg)))
    s = np.sin(psi / 2)
    num = np.sin(N * psi / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.where(np.abs(s) < 1e-12, 1.0, num / (N * s))
    return af ** 2


def beam_metrics(N, th0):
    th = np.arange(-90, 90.001, 0.01); p = af_power(th, th0, N); i0 = np.argmin(np.abs(th - th0))
    # half-power beamwidth around main lobe
    l = i0
    while l > 0 and p[l] >= 0.5: l -= 1
    r = i0
    while r < len(th) - 1 and p[r] >= 0.5: r += 1
    hpbw = th[r] - th[l]
    # first nulls
    l2 = i0
    while l2 > 0 and p[l2] > 1e-3 and not (p[l2 - 1] > p[l2] and p[l2] < 1e-3): l2 -= 1
    # simpler: walk until local minimum
    l2 = i0
    while l2 > 0 and p[l2 - 1] <= p[l2]: l2 -= 1
    r2 = i0
    while r2 < len(th) - 1 and p[r2 + 1] <= p[r2]: r2 += 1
    out = np.ones(len(th), bool); out[l2:r2 + 1] = False
    sll = 10 * np.log10(p[out].max()) if out.any() and p[out].max() > 1e-4 else np.nan
    return hpbw, th[r2] - th[l2], sll


def run():
    # --- pattern metrics
    rows = []
    for th0 in [0, 45]:
        for N in NS:
            hpbw, fnbw, sll = beam_metrics(N, th0)
            rows.append(dict(N=N, steer_deg=th0, HPBW_deg=round(hpbw, 2), HPBW_theory_deg=round(np.degrees(0.886 / (N * D * np.cos(np.radians(th0)))), 2),
                             first_null_BW_deg=round(fnbw, 2), peak_sidelobe_dB=round(sll, 2) if not np.isnan(sll) else "none"))
    met = pd.DataFrame(rows)

    # --- SIR vs separation (A at theta0, B at theta0+delta), equal powers
    seps = np.arange(5, 60.01, 0.5); sir_rows = []; mins = []
    for th0 in [0, 45]:
        for N in NS:
            sir = -10 * np.log10(af_power(th0 + seps, th0, N) + 1e-12)
            for sd, sv in zip(seps, sir): sir_rows.append(dict(N=N, steer_deg=th0, separation_deg=sd, SIR_dB=sv))
            for thr in SIR_THR_DB:
                ok = sir >= thr
                first = seps[ok][0] if ok.any() else np.nan
                robust = next((seps[i] for i in range(len(seps)) if ok[i:].all()), np.nan)
                mins.append(dict(N=N, steer_deg=th0, SIR_threshold_dB=thr, first_sep_meeting_thr_deg=first, robust_min_sep_deg=robust))
    sirdf = pd.DataFrame(sir_rows); save_csv(sirdf, "q5_sir_vs_separation.csv")
    mins = pd.DataFrame(mins); save_csv(mins, "q5_min_separation.csv"); save_csv(met, "q5_beam_metrics.csv")
    print(met.to_string()); print(mins[mins.steer_deg == 0].to_string())

    # --- figure 1: patterns
    th = np.arange(-90, 90.01, 0.1)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for N in NS: axs[0].plot(th, 10 * np.log10(af_power(th, 0, N) + 1e-8), label=f"N={N}")
    axs[0].set_ylim(-50, 3); axs[0].set_xlabel("angle (deg)"); axs[0].set_ylabel("normalised pattern (dB)"); axs[0].set_title("Q5 ULA patterns, steered to 0°"); axs[0].legend()
    for N in NS: axs[1].plot(th, 10 * np.log10(af_power(th, 45, N) + 1e-8), label=f"N={N}")
    axs[1].set_ylim(-50, 3); axs[1].set_xlabel("angle (deg)"); axs[1].set_title("Steered to 45° (beam broadens by 1/cos θ0)"); axs[1].legend()
    save_fig(fig, "q5_patterns.png")

    # --- figure 2: SIR vs separation incl. omni / sector baselines, beamwidth scaling
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for N in NS:
        d = sirdf[(sirdf.N == N) & (sirdf.steer_deg == 0)]; axs[0].plot(d.separation_deg, d.SIR_dB, label=f"ULA N={N}")
    axs[0].plot(seps, np.zeros_like(seps), "k--", label="omni (0 dB)"); axs[0].plot(seps, np.zeros_like(seps) + 0.0, "k:", label="120° sector, same sector (0 dB)")
    for thr in SIR_THR_DB: axs[0].axhline(thr, color="gray", lw=0.6)
    axs[0].set_ylim(-2, 50); axs[0].set_xlabel("angular separation A–B (deg)"); axs[0].set_ylabel("SIR at B (dB)"); axs[0].set_title("SIR at B when beam is steered at A"); axs[0].legend(fontsize=7)
    m0 = met[met.steer_deg == 0]
    axs[1].loglog(m0.N, m0.HPBW_deg, "o-", label="measured HPBW"); axs[1].loglog(m0.N, m0.HPBW_theory_deg, "k--", label="0.886 λ/(N d)")
    mm = mins[(mins.steer_deg == 0) & (mins.SIR_threshold_dB == PRIMARY)]
    axs[1].loglog(mm.N, mm.robust_min_sep_deg, "s-", label=f"min separation for {PRIMARY} dB SIR (robust)")
    axs[1].loglog(mm.N, mm.first_sep_meeting_thr_deg, "^--", label="first separation meeting it")
    axs[1].set_xlabel("elements N"); axs[1].set_ylabel("degrees"); axs[1].set_title("Beamwidth and required separation vs N"); axs[1].legend()
    save_fig(fig, "q5_sir_and_beamwidth.png")

    # --- Monte-Carlo: pair separability vs angular clustering, with steering error
    rg = rng(5000); M = 20000; steer_err = 1.0; rows = []
    for sig in [1, 2, 3, 5, 8, 12, 20, 30, 60]:
        thA = np.clip(rg.normal(0, sig, M), -60, 60); thB = np.clip(rg.normal(0, sig, M), -60, 60)
        eA = rg.normal(0, steer_err, M); eB = rg.normal(0, steer_err, M)
        for N in NS:
            sirB = 10 * np.log10((af_power(thB, thB + eB, N) + 1e-12) / (af_power(thA, thB + eB, N) + 1e-12))
            sirA = 10 * np.log10((af_power(thA, thA + eA, N) + 1e-12) / (af_power(thB, thA + eA, N) + 1e-12))
            rows.append(dict(cluster_sigma_deg=sig, N=N, steering_error_deg=steer_err, P_pair_separable=float(np.mean(np.minimum(sirA, sirB) >= PRIMARY))))
    mc = pd.DataFrame(rows); save_csv(mc, "q5_cluster_montecarlo.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for N in NS:
        d = mc[mc.N == N]; ax.plot(d.cluster_sigma_deg, d.P_pair_separable, "o-", label=f"N={N}")
    ax.set_xscale("log"); ax.axvline(3, color="r", ls=":"); ax.text(3.1, 0.05, "toll plaza\n(σ≈3°)", color="r", fontsize=8)
    ax.axvline(30, color="g", ls=":"); ax.text(31, 0.05, "highway\n(wide spread)", color="g", fontsize=8)
    ax.set_xlabel("angular spread σ of user directions (deg)"); ax.set_ylabel(f"P(both users get SIR ≥ {PRIMARY} dB)")
    ax.set_title("Q5 limit of SDMA: clustered users (1° steering error)"); ax.legend()
    save_fig(fig, "q5_cluster_limit.png")
    pick = lambda s, N: float(mc[(mc.cluster_sigma_deg == s) & (mc.N == N)].P_pair_separable.values[0])
    save_json(dict(min_sep_N8_deg=float(mm[mm.N == 8].robust_min_sep_deg.values[0]), min_sep_N16_deg=float(mm[mm.N == 16].robust_min_sep_deg.values[0]),
                   p_sep_toll_N16=pick(3, 16), p_sep_highway_N16=pick(30, 16), p_sep_toll_N8=pick(3, 8), p_sep_highway_N8=pick(30, 8)), "q5_summary.json")
    return met, mins, mc


if __name__ == "__main__":
    print("Q5 ..."); met, mins, mc = run(); print(mc[mc.N == 16].to_string())
