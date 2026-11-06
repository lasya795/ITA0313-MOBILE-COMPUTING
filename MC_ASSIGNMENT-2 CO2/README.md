# Mobile Computing – Assignment 2 (CO2): Multiple Access Techniques

Python simulations for Q1–Q7. Every experiment uses fixed random seeds (`SEED = 42` in `common.py`; each
experiment adds a documented offset), so re-running reproduces the CSV files and figures exactly.

## Run
```bash
pip install -r requirements.txt
python run_all.py            # all questions, ~1 minute -> data/*.csv, data/*.json, figures/*.png
python q3_cdma.py            # or run a single question
```
Run order matters only for Q6, which reads `data/q2..q5_summary.json` (run_all.py handles this).

## Layout
| File | Question | Main outputs (`data/`, `figures/`) |
|---|---|---|
| `q1_mobile_vs_computing.py` | Q1 | `q1_classification.csv`, `q1_sim_*.csv`, `q1_classification_table.png`, `q1_simulation.png` |
| `q2_fdma_tdma.py` | Q2 | `q2_capacity_overheads.csv`, `q2_raw_all_replicates.csv`, `q2_summary_mean.csv`, `q2_telemetry_decision.csv`, `q2_fdma_vs_tdma.png`, `q2_telemetry_decision.png` |
| `q3_cdma.py` | Q3 | `q3_cdma_results.csv`, `q3_supported_users.csv`, `q3_power_control.csv`, `q3_ber_vs_users.png`, `q3_power_control.png`, `q3_correlations.png` |
| `q4_ofdma.py` | Q4 | `q4_allocation_results.csv`, `q4_orthogonality_inner_products.csv`, `q4_cyclic_prefix.csv`, `q4_multiuser_diversity.csv`, 3 figures |
| `q5_sdma.py` | Q5 | `q5_beam_metrics.csv`, `q5_sir_vs_separation.csv`, `q5_min_separation.csv`, `q5_cluster_montecarlo.csv`, 3 figures |
| `q6_decision_matrix.py` | Q6 | `q6_ratings/weights/scores/sensitivity/choices.csv`, `q6_evidence_trace.csv`, `q6_scores_and_sensitivity.png` |
| `q7_redesign.py` | Q7 (A–F) | `q7_prediction_written_before_run.txt`, `q7_raw_all_replicates.csv`, `q7_summary_mean.csv`, `q7_headline.json`, 2 figures |

## Seeds
`rng(offset)` = `numpy.random.default_rng(42 + offset)`. Offsets: Q1 `link_trace` int(100·avail)·1000+rep; Q2 2000+; Q3 3000+; Q4 4000–4200; Q5 5000; Q7 7000/7500.

## Key assumptions (state these in your report, with citations for standard values)
* **Q2** GSM-like numbers: 8 slots/carrier, 156.25 bit-periods per slot with 114 payload bits, 26-multiframe (24 traffic + SACCH + idle) → TDMA payload efficiency 0.673 (3GPP TS 45.001/45.002). Guard channels, control channels and set-up times (0.25 s FDMA, 0.45 s TDMA) are **assumed** (see top of `q2_fdma_tdma.py`).
* **Q3** BPSK, AWGN, Eb/N0 = 10 dB, 16 000 bits/user/point (8 trials × 2 000), BER target 10⁻². "Ten times the power" = amplitude √10. Timing conditions: sync, ±1-chip jitter, fully asynchronous.
* **Q4** 15 kHz spacing (LTE-like, assumed), 8-tap exponential PDP Rayleigh channel, user average SNR 22→6 dB, Shannon rate per subcarrier, 1000 realisations, **R_min = 10 bits/OFDM-symbol** (≈150 kbit/s at 15 kHz — enough for a low-resolution lecture stream; change `R_MIN` to test your own justification).
* **Q5** d = λ/2, equal powers, primary SIR threshold 10 dB (a plain ULA has a ≈ −13 dB side-lobe floor, so higher thresholds cannot be met robustly without null steering), 1° steering error in the Monte-Carlo.
* **Q6** Ratings are **your judgement** (1–5, `R` dict). Edit them and the weights; `q6_evidence_trace.csv` links selected ratings to numbers from Q2–Q5.
* **Q7** Chosen limitation: dedicated channels idle under bursty traffic (Q2). Redesign = shared channel pool + per-burst reservation + admission control. Note that "utilisation of occupied channel-time" in the redesign is high by construction (a channel is only held while sending); the fairer comparison is **carried useful airtime** and **session blocking**, plus the new costs (signalling, delay).

## Important – before you submit
1. **Write your predictions before running** (Q1–Q6 require this) and keep a timestamped copy (e.g. a git commit of `PREDICTIONS.md`). Q1's predictions are in the code; the Q7 prediction is written to `data/q7_prediction_written_before_run.txt`.
2. **Disclose AI assistance** in the report/viva, and make sure you can explain every design choice and number — you may be asked to defend and modify the code.
3. Cite sources for any standards parameters and state your own assumptions.
