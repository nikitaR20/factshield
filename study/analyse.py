"""Statistics for the study. Runs on synthetic data until the pilot is done.

    python -m study.analyse --synthetic             # one dataset, known effects
    python -m study.analyse --synthetic --reps 50   # coverage over many datasets
    python -m study.analyse --scored study/results/scored.csv   # real data, after the pilot

Models, per outcome (correct, confidently_wrong, committed):

1. Mixed-effects logistic regression, on run 0 only (repeats are not
   independent observations; they feed the noise baseline, item 5)
       outcome ~ condition * claim_set + model + domain + retrievable
       + random intercept for source + random intercept for claim
   Claim ids are unique across sources, so the claim intercept is nested in
   source. Fitted with statsmodels' BinomialBayesMixedGLM.
   READ THIS: that is statsmodels' only binomial mixed model, and it is
   Bayesian. It reports posterior means and SDs under a N(0, 2^2) prior on
   fixed effects, not maximum-likelihood estimates and not p-values. The
   intervals printed are mean +/- 1.96 SD of an approximate posterior.

   The fit method matters. On synthetic data where this exact model is the
   true model (`--synthetic --reps 100`, run 0, 120 claims in 40 sources):
     fit_vb  (variational): "95%" intervals covered the true condition effects
             69-76% of the time; a true null was called non-zero in 25-31%.
     fit_map (Laplace, SDs from the inverse Hessian): covered 96-99%; a true
             null was called non-zero in 1%. Slightly conservative.
   So `laplace` is the default (config: analysis.mixed_method); `vb` is kept
   only to reproduce that comparison. At that synthetic size the Laplace fit
   detected a -0.6 log-odds condition effect in 65% of datasets and a -0.4
   interaction in 20%: the design is underpowered for interactions of that
   size. The claim-level SD was often estimated at 0 (a boundary estimate
   with 3 claims per source). For a conventional ML fit, lme4::glmer in R is
   the reference implementation.

2. GEE (run 0), clustered on source, exchangeable working correlation, robust SEs.
   A frequentist cross-check with p-values. Its coefficients are
   population-averaged, so they are smaller in magnitude than the mixed
   model's subject-specific ones by design; compare sign and significance,
   not size.

3. Per-model paired comparisons between conditions on run 0: McNemar (exact),
   difference in proportions with a cluster bootstrap CI that resamples
   SOURCES, since claims from one source are not independent. Holm-Bonferroni
   within each family (one family = one outcome, all model x condition-pair
   tests). Reported as descriptive.

4. Format vs free: difference with CI. No equivalence test (see the report).

5. Every condition effect printed beside the run-to-run change rate.
"""

from __future__ import annotations

import argparse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from .common import config, path, provenance

OUTCOMES = ("correct", "confidently_wrong", "committed")
REF_CONDITION = "free"


# ------------------------------------------------------------------ synthetic

TRUTH = {
    # log-odds of `correct`, relative to: free, set A, model m1, retrievable=0
    "intercept": 0.8,
    "model[m2]": -0.3,
    "model[m3]": 0.2,
    "set[B]": -0.5,
    "condition[format]": 0.0,          # a TRUE NULL: the check must not "find" it
    "condition[citation]": -0.6,
    "condition[format]:set[B]": 0.0,
    "condition[citation]:set[B]": -0.4,
    "retrievable": 0.3,
    "sd_source": 0.8,
    "sd_claim": 0.6,
}


def synthetic(seed: int) -> pd.DataFrame:
    sc = config()["analysis"]["synthetic"]
    rng = np.random.default_rng(seed)
    labels = np.array(["supported", "refuted", "not_enough_evidence"])
    rows = []
    for s in range(sc["n_sources"]):
        cset = "A" if s < sc["n_sources"] // 2 else "B"
        domain = ["economics", "health", "politics"][s % 3]
        u_src = rng.normal(0, TRUTH["sd_source"])
        for k in range(sc["claims_per_source"]):
            cid = f"syn{s:03d}_{k}"
            u_cl = rng.normal(0, TRUTH["sd_claim"])
            retr = int(rng.random() < 0.7)
            gold = labels[rng.integers(3)]
            for m in ("m1", "m2", "m3"):
                for cond in ("free", "format", "citation"):
                    for r in range(sc["n_repeats"]):
                        eta = (TRUTH["intercept"] + TRUTH.get(f"model[{m}]", 0)
                               + (TRUTH["set[B]"] if cset == "B" else 0)
                               + TRUTH.get(f"condition[{cond}]", 0)
                               + (TRUTH.get(f"condition[{cond}]:set[B]", 0) if cset == "B" else 0)
                               + TRUTH["retrievable"] * retr + u_src + u_cl)
                        correct = rng.random() < 1 / (1 + np.exp(-eta))
                        pred = gold if correct else rng.choice([l for l in labels if l != gold])
                        committed = pred in ("supported", "refuted")
                        rows.append({"claim_id": cid, "claim_set": cset, "domain": domain,
                                     "source_id": f"src{s:03d}", "retrievable": retr, "model": m,
                                     "condition": cond, "run_index": r, "gold": gold, "pred": pred,
                                     "correct": int(correct), "committed": int(committed),
                                     "confidently_wrong": int(committed and not correct)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ models

def _formula(df: pd.DataFrame, outcome: str) -> str:
    terms = [f"C(condition, Treatment('{REF_CONDITION}'))"]
    if df["claim_set"].nunique() > 1:
        terms[0] = f"{terms[0]} * C(claim_set)"
    for col, term in (("model", "C(model)"), ("domain", "C(domain)"), ("retrievable", "retrievable")):
        if col in df and df[col].nunique() > 1:
            terms.append(term)
    return f"{outcome} ~ " + " + ".join(terms)


def _pretty(name: str) -> str:
    return (name.replace(f"C(condition, Treatment('{REF_CONDITION}'))[T.", "condition[")
                .replace("C(claim_set)[T.", "set[").replace("C(model)[T.", "model[")
                .replace("C(domain)[T.", "domain[").replace("Intercept", "intercept"))


def fit_mixed(df: pd.DataFrame, outcome: str, method: str | None = None) -> pd.DataFrame:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    method = method or config()["analysis"]["mixed_method"]
    f = _formula(df, outcome)
    vc = {"source": "0 + C(source_id)", "claim": "0 + C(claim_id)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BinomialBayesMixedGLM.from_formula(f, vc, df)
        if method == "laplace":
            res = model.fit_map()
        elif method == "vb":
            res = model.fit_vb()
        else:
            raise ValueError(f"mixed_method must be laplace or vb, not {method!r}")
    fe = pd.DataFrame({"term": [_pretty(n) for n in model.exog_names],
                       "mean": res.fe_mean, "sd": res.fe_sd})
    fe["lo"] = fe["mean"] - 1.96 * fe["sd"]
    fe["hi"] = fe["mean"] + 1.96 * fe["sd"]
    # vcp_mean are log standard deviations of the random effects
    vcs = pd.DataFrame({"term": [f"sd_{n}" for n in model.vcp_names],
                        "mean": np.exp(res.vcp_mean), "sd": np.nan, "lo": np.nan, "hi": np.nan})
    out = pd.concat([fe, vcs], ignore_index=True)
    out.attrs["formula"] = f
    out.attrs["method"] = method
    return out


def fit_gee(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    f = _formula(df, outcome)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.gee(f, "source_id", df, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = res.conf_int()
    return pd.DataFrame({"term": [_pretty(n) for n in res.params.index], "coef": res.params.values,
                         "lo": ci[0].values, "hi": ci[1].values, "p": res.pvalues.values})


# ------------------------------------------------------------------ paired

def _cluster_index(claims: pd.Series, sources: pd.Series) -> list[np.ndarray]:
    idx = {}
    for i, s in enumerate(sources):
        idx.setdefault(s, []).append(i)
    return [np.array(v) for v in idx.values()]


def paired(df: pd.DataFrame, outcome: str, reps: int, seed: int) -> pd.DataFrame:
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests

    d0 = df[df["run_index"] == df["run_index"].min()]
    conds = [c for c in config()["conditions"] if c in set(d0["condition"])]
    rng = np.random.default_rng(seed)
    out = []
    for m in sorted(d0["model"].unique()):
        wide = d0[d0["model"] == m].pivot_table(index=["claim_id", "source_id"], columns="condition",
                                                 values=outcome, aggfunc="first").dropna().reset_index()
        clusters = _cluster_index(wide["claim_id"], wide["source_id"])
        for a, b in combinations(conds, 2):
            if a not in wide or b not in wide:
                continue
            x, y = wide[a].to_numpy(), wide[b].to_numpy()
            table = [[int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())],
                     [int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum())]]
            p = mcnemar(table, exact=True).pvalue
            diff = y.mean() - x.mean()
            boots = np.empty(reps)
            for i in range(reps):
                pick = np.concatenate([clusters[j] for j in rng.integers(len(clusters), size=len(clusters))])
                boots[i] = y[pick].mean() - x[pick].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append({"outcome": outcome, "model": m, "comparison": f"{b} - {a}", "n_claims": len(x),
                        "diff": diff, "lo": lo, "hi": hi, "discordant": table[0][1] + table[1][0],
                        "p_mcnemar": p})
    res = pd.DataFrame(out)
    if len(res):
        res["p_holm"] = multipletests(res["p_mcnemar"], method="holm")[1]
    return res


def format_vs_free(df: pd.DataFrame, outcome: str, reps: int, seed: int) -> dict:
    """Pooled over models, paired by claim x model on run 0, cluster bootstrap
    over sources. A difference and its CI — no equivalence claim."""
    d0 = df[df["run_index"] == df["run_index"].min()]
    wide = d0.pivot_table(index=["claim_id", "source_id", "model"], columns="condition",
                          values=outcome, aggfunc="first").dropna().reset_index()
    if "format" not in wide or "free" not in wide:
        return {}
    x, y = wide["free"].to_numpy(), wide["format"].to_numpy()
    clusters = _cluster_index(wide["claim_id"], wide["source_id"])
    rng = np.random.default_rng(seed)
    boots = np.array([
        (lambda pick: y[pick].mean() - x[pick].mean())(
            np.concatenate([clusters[j] for j in rng.integers(len(clusters), size=len(clusters))]))
        for _ in range(reps)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"outcome": outcome, "diff_format_minus_free": y.mean() - x.mean(), "lo": lo, "hi": hi,
            "n_pairs": len(x), "ci_half_width": (hi - lo) / 2}


def run_to_run(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "condition", "claim_id"])["pred"].nunique()
    reps = df.groupby(["model", "condition", "claim_id"])["run_index"].nunique()
    changed = (g[reps > 1] > 1).groupby(level=["model", "condition"]).mean()
    return changed.rename("run_to_run_change").reset_index()


# ------------------------------------------------------------------ reporting

def report(df: pd.DataFrame, *, reps: int, seed: int, truth: dict | None = None,
           method: str | None = None) -> dict:
    pd.set_option("display.width", 160)
    out = {}
    rr = run_to_run(df) if df["run_index"].nunique() > 1 else pd.DataFrame()
    for outcome in OUTCOMES:
        if outcome not in df or df[outcome].nunique() < 2:
            continue
        print(f"\n{'=' * 78}\nOUTCOME: {outcome}\n{'=' * 78}")
        # Models are fitted on ONE run. Repeats of the same claim x model x
        # condition are not independent observations (at temperature 0 they are
        # often identical), so pooling them would overstate precision. Repeats
        # are used only for the run-to-run noise baseline below.
        first = df[df["run_index"] == df["run_index"].min()]
        mixed = fit_mixed(first, outcome, method)
        print(f"Mixed model ({mixed.attrs['method']} posterior, run {first['run_index'].iloc[0]} only, "
              f"n={len(first)}; {mixed.attrs['formula']} + (1|source) + (1|claim)):")
        show = mixed.copy()
        if truth and outcome == "correct":
            show["truth"] = show["term"].map(truth)
            show["covers"] = np.where(show["truth"].notna() & show["lo"].notna(),
                                      (show["lo"] <= show["truth"]) & (show["truth"] <= show["hi"]), None)
        print(show.round(3).to_string(index=False))
        gee = fit_gee(first, outcome)
        print("\nGEE cross-check (population-averaged, clustered on source):")
        print(gee[gee["term"].str.startswith("condition")].round(4).to_string(index=False))
        pr = paired(df, outcome, reps, seed)
        if len(pr):
            print("\nPaired per model, run 0 (cluster-bootstrap CI over sources; Holm within outcome):")
            print(pr.round(4).to_string(index=False))
        fv = format_vs_free(df, outcome, reps, seed)
        if fv:
            print(f"\nFormat vs free, pooled: {fv['diff_format_minus_free']:+.3f} "
                  f"[{fv['lo']:+.3f}, {fv['hi']:+.3f}] (n={fv['n_pairs']} pairs). "
                  f"CI half-width {fv['ci_half_width']:.3f}: differences smaller than this "
                  f"cannot be distinguished from zero in this design.")
        if len(rr) and len(pr):
            noise = rr.groupby("model")["run_to_run_change"].mean()
            print("\nCondition effects beside run-to-run noise (same claim, model, condition, "
                  "different run):")
            for _, row in pr.iterrows():
                print(f"  {row['model']:<8}{row['comparison']:<20} |diff|={abs(row['diff']):.3f}   "
                      f"run-to-run verdict change={noise.get(row['model'], float('nan')):.3f}")
        out[outcome] = {"mixed": mixed, "gee": gee, "paired": pr, "format_vs_free": fv}
    return out


def coverage_study(n_reps: int, seed: int, method: str | None = None) -> None:
    """Fit many synthetic datasets and measure what the intervals actually do."""
    method = method or config()["analysis"]["mixed_method"]
    keys = ["condition[format]", "condition[citation]", "condition[citation]:set[B]",
            "condition[format]:set[B]"]
    rec = {k: {"cover": 0, "excl0": 0, "est": []} for k in keys}
    for i in range(n_reps):
        df = synthetic(seed + 1000 + i)
        df = df[df["run_index"] == 0]
        m = fit_mixed(df, "correct", method).set_index("term")
        for k in keys:
            row = m.loc[k]
            rec[k]["cover"] += row["lo"] <= TRUTH[k] <= row["hi"]
            rec[k]["excl0"] += not (row["lo"] <= 0 <= row["hi"])
            rec[k]["est"].append(row["mean"])
    print(f"\nCoverage of the {method} fit over {n_reps} synthetic datasets (nominal 95%):")
    print(f"  {'term':<30}{'truth':>7}{'mean est':>10}{'bias':>8}{'covered':>9}{'CI excl. 0':>12}")
    for k in keys:
        est = np.mean(rec[k]["est"])
        print(f"  {k:<30}{TRUTH[k]:>7.2f}{est:>10.3f}{est - TRUTH[k]:>8.3f}"
              f"{rec[k]['cover'] / n_reps:>9.0%}{rec[k]['excl0'] / n_reps:>12.0%}")
    print("  'CI excl. 0' is power for real effects and the false-positive rate for the null ones.")


def main() -> None:
    cfg = config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--reps", type=int, default=0, help="with --synthetic: coverage study")
    ap.add_argument("--method", choices=["laplace", "vb"], default=None,
                    help="mixed-model fit; default from config")
    ap.add_argument("--scored", default=None)
    ap.add_argument("--i-have-finished-the-pilot", action="store_true", dest="pilot_done")
    args = ap.parse_args()
    seed = cfg["seed"]
    boot = cfg["analysis"]["bootstrap_reps"]
    print(f"provenance: {provenance()}")
    if args.synthetic:
        df = synthetic(seed)
        print(f"SYNTHETIC DATA: {len(df)} rows, {df['claim_id'].nunique()} claims, "
              f"{df['source_id'].nunique()} sources. True effects: {TRUTH}")
        report(df, reps=boot, seed=seed, truth=TRUTH, method=args.method)
        if args.reps:
            coverage_study(args.reps, seed, args.method)
        return
    if not args.pilot_done:
        raise SystemExit("Real-data analysis is locked until the pilot is complete and the "
                         "analysis plan is frozen. Use --synthetic, or pass "
                         "--i-have-finished-the-pilot deliberately.")
    df = pd.read_csv(args.scored or path("results") / "scored.csv")
    report(df, reps=boot, seed=seed, method=args.method)


if __name__ == "__main__":
    main()
