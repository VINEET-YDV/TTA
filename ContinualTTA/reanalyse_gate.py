# =============================================================================
# Re-analyse the saved gate log with the CORRECT batches-per-block.
#
# Bug: the analysis script used BATCHES_PER_BLOCK = 10000 // 32 = 312.
#      10000/32 = 312.5, so each block actually yields 313 batches
#      (the last is a partial batch of 16 images).
#      Your log has 23475 batches = 313 x 75. Confirmed.
#
# CONSEQUENCE: assumed boundaries sat at multiples of 312 while real
#      boundaries are at multiples of 313, so the assumed position drifted
#      by one batch per block -- 75 batches off by the end. With a
#      tolerance of 1-5 batches this misses essentially every real
#      boundary and yields lift ~1 and severity-recall 0 REGARDLESS of
#      whether the gate works.
#
# This script recomputes everything from the `t` column, which is correct,
# and ignores the block / is_boundary columns, which are not.
#
# Run:
#   python reanalyse_gate.py
#   python reanalyse_gate.py --log path\to\gate_log_batches.csv
# =============================================================================

import os
import csv
import math
import argparse
import statistics as st

DEFAULT_LOG = (r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA"
               r"\results\gate_analysis\gate_log_batches.csv")

N_CORRUPTIONS = 15
N_SEVERITIES  = 5
N_BLOCKS      = N_CORRUPTIONS * N_SEVERITIES     # 75
IMAGES        = 10000
BATCH_SIZE    = 32
TAU           = 0.04


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                t=int(r["t"]),
                js=float(r["js"]) if r["js"] not in ("", "nan") else float("nan"),
                adapted=bool(int(r["adapted"])),
                n_reliable=int(r["n_reliable"]),
            ))
    return sorted(rows, key=lambda r: r["t"])


def infer_bpb(n_rows):
    """Recover batches-per-block from the actual log length."""
    exact = n_rows / N_BLOCKS
    bpb = round(exact)
    return bpb, exact


def analyse(rows, bpb, tolerance=1):
    n_total  = len(rows)
    n_uncond = sum(1 for r in rows if r["js"] != r["js"])
    fire_all = sum(r["adapted"] for r in rows) / n_total

    dec = [r for r in rows if r["js"] == r["js"]]
    n   = len(dec)

    # true boundary = first batch of each block after the first
    bpos = {b * bpb for b in range(1, N_BLOCKS)}
    near = lambda t: any(abs(t - bp) < tolerance for bp in bpos)

    # severity jumps: blocks 15, 30, 45, 60
    spos = {b * bpb for b in range(N_CORRUPTIONS, N_BLOCKS, N_CORRUPTIONS)}
    near_sev = lambda t: any(abs(t - bp) < tolerance for bp in spos)

    tp = sum(1 for r in dec if r["adapted"] and near(r["t"]))
    fp = sum(1 for r in dec if r["adapted"] and not near(r["t"]))
    fn = sum(1 for r in dec if not r["adapted"] and near(r["t"]))
    tn = n - tp - fp - fn

    nb, nnb = tp + fn, fp + tn
    p_b  = tp / nb  if nb  else 0.0
    p_nb = fp / nnb if nnb else 0.0
    lift = p_b / p_nb if p_nb > 0 else float("inf")

    sev_hits = sum(1 for r in dec if r["adapted"] and near_sev(r["t"]))
    sev_tot  = sum(1 for r in dec if near_sev(r["t"]))

    js_b  = [r["js"] for r in dec if near(r["t"])]
    js_nb = [r["js"] for r in dec if not near(r["t"])]

    return dict(
        n_total=n_total, n_uncond=n_uncond, fire_rate=fire_all,
        tolerance=tolerance, tp=tp, fp=fp, fn=fn, tn=tn,
        boundary_recall=p_b, within_rate=p_nb, lift=lift,
        sev_hits=sev_hits, sev_tot=sev_tot,
        med_js_b=st.median(js_b) if js_b else float("nan"),
        med_js_nb=st.median(js_nb) if js_nb else float("nan"),
        max_js_b=max(js_b) if js_b else float("nan"),
        frac_b_above_tau=sum(1 for j in js_b if j > TAU)/len(js_b) if js_b else 0,
        frac_nb_above_tau=sum(1 for j in js_nb if j > TAU)/len(js_nb) if js_nb else 0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        print(f"Log not found: {args.log}")
        return

    rows = load(args.log)
    bpb, exact = infer_bpb(len(rows))

    print("=" * 66)
    print("RE-ANALYSIS WITH CORRECTED BLOCK SIZE")
    print("=" * 66)
    print(f"  rows in log            : {len(rows)}")
    print(f"  rows / 75 blocks       : {exact:.3f}")
    print(f"  batches per block      : {bpb}   "
          f"(script previously assumed {IMAGES // BATCH_SIZE})")
    print(f"  ceil(10000/32)         : {math.ceil(IMAGES/BATCH_SIZE)}")
    if bpb != IMAGES // BATCH_SIZE:
        drift = (bpb - IMAGES // BATCH_SIZE) * (N_BLOCKS - 1)
        print(f"  -> previous run's boundary positions drifted by {drift} "
              f"batches by block 75")
    print()

    print(f"{'tol':>4} {'bnd recall':>11} {'within-blk':>11} {'lift':>9} "
          f"{'caught':>10} {'sev jumps':>11}")
    print("-" * 62)
    for tol in (1, 2, 3, 5, 10):
        m = analyse(rows, bpb, tol)
        lift = "inf" if m["lift"] == float("inf") else f"{m['lift']:.2f}"
        print(f"{tol:>4} {m['boundary_recall']:11.3f} {m['within_rate']:11.4f} "
              f"{lift:>9} {m['tp']:>4}/{N_BLOCKS-1:<5} "
              f"{m['sev_hits']:>4}/{m['sev_tot']:<6}")

    m = analyse(rows, bpb, 1)
    print()
    print(f"  overall fire rate      : {100*m['fire_rate']:.2f}%")
    print(f"  median JS at boundary  : {m['med_js_b']:.5f}")
    print(f"  median JS within block : {m['med_js_nb']:.5f}")
    print(f"  max JS at boundary     : {m['max_js_b']:.5f}")
    print(f"  tau                    : {TAU}")
    print(f"  % boundaries above tau : {100*m['frac_b_above_tau']:.1f}%")
    print(f"  % within-blk above tau : {100*m['frac_nb_above_tau']:.1f}%")

    print()
    print("=" * 66)
    print("VERDICT")
    print("=" * 66)
    L = m["lift"]
    if L == float("inf") or L > 3:
        print("  Lift >> 1. Firing IS boundary-localised.")
        print("  The paper's timing claim is supported. The earlier lift~0.9")
        print("  was the off-by-one artifact.")
    elif L > 1.5:
        print("  Lift moderate. Boundary-biased but not sharply so.")
        print("  Soften 'silent within stable periods' but keep the claim.")
    else:
        print("  Lift still ~1 after the fix. Firing is NOT boundary-localised.")
        print("  The accuracy gain comes from adapting LESS, not from adapting")
        print("  at the right TIME. This is a real finding and the paper's")
        print("  mechanism claim needs rewriting.")
        print()
        print("  Note the median JS values: if boundary and within-block JS are")
        print("  nearly equal, the batch marginal simply does not shift much at")
        print("  a corruption boundary on CIFAR-10-C -- the gate cannot see")
        print("  something that is not in its input signal.")


if __name__ == "__main__":
    main()