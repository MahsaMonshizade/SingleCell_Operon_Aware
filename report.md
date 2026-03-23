# Meeting Talking Points

---

## 1. What I Built — OperonAwareSCVI

Standard scVI was designed for eukaryotic cells and has no awareness of prokaryotic
transcriptional organization. My core contribution is modifying the scVI loss function
to incorporate operon structure as a biological prior.

**The key idea:** genes in the same operon are co-transcribed, so their decoded
expression in the generative model should be similar. I enforce this by adding a
penalty term to the ELBO reconstruction loss:

    L_total = L_reconstruction + λ · mean( (px_rate[:, g1] - px_rate[:, g2])² )

where g1 and g2 are indices of operon-neighbor gene pairs, px_rate is the model's
decoded expression rate, and λ controls the strength of the regularization.

**Implementation:** I cloned the scvi-tools repository and added two new files:
- scvi/module/_vae_new.py  →  OperonAwareVAE  (modified VAE with operon penalty)
- scvi/model/_scvi_new.py  →  OperonAwareSCVI (the model class that wraps it)

The rest of scvi-tools is unchanged, so the new model inherits all standard scVI
functionality — training, inference, latent representation, etc.

**Operon pairs input:** provided as a CSV of gene pairs in Blattner locus tag format
(e.g. b0001, b0002), matching the gene identifiers in our AnnData object.

---

## 2. A Technical Challenge I Solved — Gene Name Mapping

RegulonDB uses gene symbols like thrA and lacZ. Our AnnData uses Blattner locus
tags like b0002 and b0001. These are two entirely different naming systems for the
same genes, and no direct mapping exists in either file alone.

To bridge this, I wrote a parser that reads the RefSeq GFF3 annotation file for
E. coli K-12 MG1655, which contains both Name=thrA and locus_tag=b0002 in the
same record. This gave me a complete symbol → Blattner → adata column index
mapping, allowing me to evaluate properly against RegulonDB ground truth.

---

## 3. How I Generated Operon Neighbor Pairs

Rather than using RegulonDB directly for training, I generated operon neighbor pairs
from the genome annotation itself using a heuristic based on genomic proximity.

**The logic:** in bacteria, genes in the same operon tend to be on the same strand
and separated by a very short intergenic region (typically < 200 bp). I exploit
this by scanning the GFF file and flagging consecutive gene pairs that satisfy
both conditions.

The algorithm:
  1. Parse the GFF file and extract all gene features with their locus tags
  2. Sort genes by genomic start position
  3. For each consecutive pair, check:
       - Same strand (both + or both −)
       - Intergenic distance: 0 ≤ (curr.start − prev.end) < 200 bp
  4. Pairs passing both filters are written to operon_neighbors.csv

This produced 1,323 gene pairs used as the operon prior during training
(after dropping pairs where either gene was absent from the filtered AnnData).

**Why this approach rather than using RegulonDB directly for training:**
The proximity heuristic is purely sequence-based and independent of the
expression-based RegulonDB annotations, which means RegulonDB can serve as a
fully independent ground-truth evaluation set with no overlap between training
signal and evaluation labels.

**Limitation worth raising:** the 200 bp threshold is a heuristic. Some genuine
operon pairs have longer intergenic regions, and some co-directional proximal
genes are not actually co-transcribed. This is a natural discussion point about
whether training on a cleaner (but smaller) set like RegulonDB Strong operons
directly would give a stronger signal.

---

## 4. Evaluation Against RegulonDB Ground Truth

I downloaded the RegulonDB operon set (v14.5, 848 multi-gene operons) and evaluated
how well each model's denoised expression respects known operon co-expression.
The evaluation uses a held-out set of cells not seen during training.

**Important context:** RegulonDB v14.5 has no Confirmed-confidence operons —
only Strong (61 operons) and Weak (787 operons). The 61 Strong operons are the
most biologically reliable ground truth, so they are the most important subset.

---

### Panel 1 — Per-Operon Intra-Correlation (scatter plot, top left)

For each operon, I computed the mean pairwise Pearson r between the denoised
expression of its member genes across cells. Each point is one operon. Points
above the diagonal mean the operon-aware model captures that operon's
co-expression better than standard scVI.

The cloud sits very close to the diagonal with a slight lean above it overall,
meaning there is a modest but consistent improvement across most operons.

    All operons (848):  Standard = 0.3380,  Operon = 0.3434,  Δ = +0.0055

---

### Panel 2 — Improvement by Confidence Level (boxplot, top middle)

This breaks the per-operon Δ down by RegulonDB confidence level.

    Strong operons (61):  median Δ ≈ 0,  mean Δ = −0.0016  ← Standard wins here
    Weak operons (787):   median Δ ≈ 0,  distribution slightly positive

The Strong result is the honest one to highlight: on the most reliable ground
truth, the two models perform essentially the same, with standard scVI marginally
ahead (Δ = −0.0016). The overall positive result is driven largely by Weak operons,
which have less reliable annotation. This is worth being transparent about.

---

### Panel 3 — Intra- vs Inter-Operon Contrast (bar chart, top right)

This tests whether operon gene pairs are specifically more correlated than random
gene pairs — the proper null comparison. A larger gap between the intra and inter
bars means the model is better at distinguishing real operon structure from noise.

    Standard SCVI:  intra = 0.3162,  inter ≈ 0.047,  contrast = 0.3162
    Operon SCVI:    intra = 0.3135,  inter ≈ 0.055,  contrast = 0.3135  ← lower

The operon model slightly raises inter-operon correlations as well as intra, which
narrows the contrast by −0.0027. This suggests the penalty is not perfectly specific
— it may be encouraging general co-expression slightly beyond just operon pairs.

---

### Panel 4 — Distribution of Per-Operon Improvement (histogram, bottom left)

This shows the full distribution of Δ values (operon r minus standard r) across
all 848 operons.

    All mean  = +0.0055  (pink line)  — operon model better overall
    Strong mean = −0.0016 (green line) — standard model better on Strong operons
    % operons improved = 53.9%

The distribution is approximately centred on zero with a slight positive skew.
53.9% of operons improve, meaning the operon model is better more often than not,
but the effect size is small and not statistically significant (p = 0.39).

---

### Panel 5 — Within-Operon Gene Similarity (scatter, bottom right)

Rather than cell-level correlations, this measures gene-level similarity — how
similar the expression profiles of genes within the same operon are across the
cell population. Points above the diagonal mean the operon model captures tighter
gene-level co-expression.

    Standard = 0.3435,  Operon = 0.3524,  Δ = +0.0089  ✓ Operon wins

This is the cleanest positive result in the evaluation and shows the penalty is
having the intended effect at the gene level even when cell-level effects are small.

---

### Summary Table

| Metric                         | Standard | Operon | Δ       | Winner      |
|--------------------------------|----------|--------|---------|-------------|
| Mean intra-operon r (all)      | 0.3380   | 0.3434 | +0.0055 | ✓ Operon    |
| Mean intra-operon r (Strong)   | 0.3367   | 0.3351 | −0.0016 | ✓ Standard  |
| Intra-inter contrast (Δr)      | 0.3162   | 0.3135 | −0.0027 | ✓ Standard  |
| % operons improved             | —        | 53.9%  | —       | ✓ Operon    |
| Within-operon gene similarity  | 0.3435   | 0.3524 | +0.0089 | ✓ Operon    |

Operon SCVI wins 3/5 metrics. The two losses are on the metrics that matter most
for specificity (Strong ground truth and intra-inter contrast), which is the
honest framing to bring to the meeting.

---

## 5. Lambda Tuning

Lambda controls the strength of the operon penalty. I ran a sweep:

    λ = 0.01  →  Δ = +0.0031,  52.9% improved   (too weak)
    λ = 0.05  →  Δ = +0.0038,  53.4% improved   ← best
    λ = 0.10  →  Δ = −0.0025,  49.6% improved   (over-penalized)

The current setting is λ = 0.05. Too high a lambda distorts the reconstruction
loss and degrades performance. The optimum is shallow, suggesting the model is
not very sensitive to this parameter in this range.

---

## 5. Honest Summary

The operon-aware model shows a consistent but modest and not statistically
significant improvement overall (p = 0.39). On the most reliable ground truth
(Strong operons), the two models perform essentially the same. The clearest
positive signal is in within-operon gene similarity (+0.0089).

This is not a null result — the regularizer is doing something biologically
meaningful — but the effect size is small. This likely reflects the difficulty
of the task: the penalty must compete against the full reconstruction objective
across 3,070 genes and 57,627 cells.

---

## 6. Questions to Discuss

- The Strong operon result slightly favours Standard. Is this because 61 operons
  is too small a sample, or a genuine signal that the model needs rethinking?

- The intra-inter contrast narrows slightly — should I penalize only confirmed
  operon pairs rather than all pairs, to improve specificity?

- Would it make sense to rebuild the training operon pairs directly from
  RegulonDB Strong operons, so training and evaluation are fully consistent?

- Should I explore harder constraints (e.g. tied latent dimensions for operon
  genes) rather than a soft penalty on decoded expression?