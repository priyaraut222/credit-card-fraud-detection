# Findings

Notes on the more interesting and/or counterintuitive results from this project,
written up separately from the README because they're worth reading as a
narrative, not just a metrics table.

## SMOTE vs. class-weighting: SMOTE did not clearly win

SMOTE (Synthetic Minority Oversampling) is often treated as a default
best-practice for imbalanced classification — it's the first technique most
tutorials reach for. This project tested that assumption directly instead of
assuming it, by training each classifier three ways: unmodified, with
class-weighting, and with SMOTE.

On the initial (random-split) run:

| Model | PR-AUC |
|---|---|
| Random Forest (baseline) | 0.861 |
| Random Forest (class-weighted) | 0.850 |
| Random Forest (SMOTE) | 0.868 |
| Logistic Regression (baseline) | 0.744 |
| Logistic Regression (class-weighted) | 0.719 |
| Logistic Regression (SMOTE) | 0.725 |

SMOTE gave Random Forest a small edge over the baseline, but **class-weighting
actually underperformed the untouched baseline for both models** — and for
Logistic Regression, neither imbalance strategy beat doing nothing at all.

**Why this is worth noting, not just reporting:** `V1`–`V28` are PCA
components, meaning "between two fraud examples" in that space isn't
guaranteed to be "a realistic fraud pattern." SMOTE generates synthetic
minority-class points by interpolating between real ones — which works well
when the feature space has meaningful geometric structure, but is a weaker
assumption in a PCA-transformed space where the axes don't correspond to
anything interpretable. The gain from SMOTE here is real but modest, and not
large enough to justify defaulting to it without checking — which was the
entire point of testing it against the alternative rather than assuming the
answer.

**Practical takeaway:** for this dataset, a plain baseline classifier or
light class-weighting is a perfectly reasonable starting point. SMOTE is
worth trying, not worth assuming.

## Time-based validation changes the story

The random split above is the easier, more common way to validate a fraud
model — and it's also optimistic in a way that matters. `Time` is a real
temporal signal in this dataset (seconds since the first transaction), and a
random split allows the model to train on transactions that occur
chronologically *after* some of the transactions it's evaluated on. That
never happens in production: you can't train tomorrow's fraud detector on
data from the day after tomorrow.

This project's final pipeline instead sorts by `Time` and holds out the last
20% of transactions chronologically — a stricter, more realistic test of
forward-looking performance. Expect PR-AUC to come in somewhat lower under
this split than under the random one above; that gap *is* the point. It's a
more honest estimate of how the model would perform if deployed.

(Run `train_models.py` and check `models/metrics.json` for the exact
time-based numbers — they intentionally aren't hardcoded here, since they're
a direct output of the pipeline, not a claim made separately from it.)

## Unsupervised methods trade recall for robustness to novelty

Isolation Forest and the Autoencoder consistently score lower on PR-AUC than
the supervised models — expected, since they never see fraud labels during
training. The practical argument for including them isn't "they win on
paper," it's that supervised models can only recognize fraud patterns that
looked like the labeled fraud they were trained on. A genuinely novel fraud
pattern — one that doesn't resemble anything in the training set — is
exactly the case where an unsupervised method has a chance of catching
something a supervised model would confidently wave through.

The ensemble (averaging normalized Isolation Forest and Autoencoder scores)
was built to test whether combining two different "unusualness" signals
beats either alone — check `models/metrics.json` for whether it did on your
training run; the two methods reason about anomalies differently (feature
isolation vs. reconstruction fidelity) so there's a reasonable case they
catch partially different things.

## What would change this analysis

- **More recent data.** This dataset is from September 2013. Fraud tactics
  evolve; a model this good on 2013 patterns says little about current ones.
- **Unmasked features.** If `V1`–`V28` corresponded to real, interpretable
  transaction attributes instead of anonymized PCA components, SMOTE's
  interpolation assumption would likely hold up better, and feature
  engineering could meaningfully improve every model here.
- **A real cost model.** The \$500 / \$5 cost assumption used for threshold
  tuning is a reasonable placeholder, not a number pulled from an actual
  card issuer's loss data. Real deployment would need that number sourced
  from the business, not assumed.
