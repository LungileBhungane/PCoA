"""
PCoA biplot for MZmine feature tables + GNPS-style metadata.

Edit the CONFIG block below, then run the file (Run button / F5 / `python pcoa_biplot.py`).
Requires: pandas numpy scipy matplotlib scikit-learn
    pip install pandas numpy scipy matplotlib scikit-learn
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# =============================================================================
# CONFIG  --  everything you normally change lives between these lines
# =============================================================================

# ---- files ------------------------------------------------------------------
FEATURE_TABLE = "featuretable_reformated.csv"
METADATA      = "merged_metadata.csv"
OUTPUT_DIR    = "pcoa_output"

# How to match feature-table columns to metadata rows.
# Feature table columns look like "Acid-Dam-C18-1.mzML Peak area"
# Metadata filenames look like    "Acid-Dam-C18-1.mzML"
SAMPLE_COL_SUFFIX = " Peak area"
METADATA_FILENAME_COL = "filename"

# ---- which samples go into the ordination ------------------------------------
TYPE_COLUMN   = "ATTRIBUTE_type"   # column holding SAMPLE / BLANK / QC
BLANK_LABELS  = ["BLANK"]          # values in TYPE_COLUMN that mark blanks
KEEP_TYPES    = ["SAMPLE"]         # types plotted in the ordination

# ---- 1. BLANK SUBTRACTION ----------------------------------------------------
# A feature is DROPPED when:  max intensity across blanks  >=  mean across
# samples / BLANK_RATIO.  Equivalently, a feature is KEPT only when it is more
# than BLANK_RATIO-fold more abundant in samples than in the worst blank.
BLANK_SUBTRACTION = True
BLANK_RATIO       = 10.0     # the "10th of the average in the samples" rule
BLANK_DROP_IF_ABSENT_IN_SAMPLES = True   # also drop features that are 0 in all samples

# ---- 2. FEATURE PRESENCE FILTER (optional, applied after blank subtraction) ---
MIN_PREVALENCE = 2          # feature must be non-zero in at least this many samples
                            # set to 0 to disable

# ---- 3. IMPUTATION -----------------------------------------------------------
# What to do with zeros (below-detection values) before transforming.
#   "none"       -> leave zeros as they are
#   "small"      -> replace 0 with IMPUTE_FRACTION * (min non-zero of that feature)
#   "half_min"   -> replace 0 with half the global minimum non-zero value
#   "knn"        -> k-nearest-neighbour imputation (zeros treated as missing)
IMPUTATION      = "small"
IMPUTE_FRACTION = 0.5
KNN_NEIGHBOURS  = 5

# ---- 4. SAMPLE-CENTRIC NORMALIZATION -----------------------------------------
# Puts samples on a comparable scale (corrects injection / extraction differences).
#   "none"   -> raw intensities
#   "tic"    -> total ion current: divide each sample by its column sum
#   "median" -> divide each sample by its median non-zero intensity
#   "pqn"    -> probabilistic quotient normalization (robust, MS-standard)
NORMALIZATION = "tic"

# ---- 5. TRANSFORMATION -------------------------------------------------------
#   "none" | "log10" (log10(x+1)) | "sqrt" | "fourth_root"
TRANSFORM = "log10"

# ---- 6. FEATURE SCALING ------------------------------------------------------
# Puts features on a comparable scale (stops huge peaks dominating).
#   "none"   -> no scaling
#   "auto"   -> centre + divide by SD (unit variance / z-score)
#   "pareto" -> centre + divide by sqrt(SD)   (mild, common in metabolomics)
#   "range"  -> centre + divide by (max - min)
#   "center" -> centre only
FEATURE_SCALING = "none"

# ---- 7. DISTANCE METRIC ------------------------------------------------------
# Non-negative-only metrics: braycurtis, canberra, jaccard
# Work with negatives:       euclidean, cosine, correlation, cityblock, chebyshev
DISTANCE_METRIC = "braycurtis"
BINARY_PRESENCE_ABSENCE = False   # True -> convert to 0/1 before distance (for jaccard)

# ---- 8. PLOTTING -------------------------------------------------------------
COLOR_BY      = "ATTRIBUTE_source"     # metadata column used for marker colour
SHAPE_BY      = "ATTRIBUTE_resin"      # metadata column for marker shape; None to disable
LABEL_POINTS  = False                  # write sample names next to markers
AXES          = (1, 2)                 # which PCoA axes to plot (1-based)
DRAW_ELLIPSES = True                   # 95% confidence ellipse per COLOR_BY group

# ---- 9. BIPLOT ARROWS --------------------------------------------------------
SHOW_BIPLOT_ARROWS = True
N_TOP_FEATURES     = 15       # number of feature vectors to draw
ARROW_SCALE        = 0.9      # 1.0 = arrows fill the plot area
FEATURE_LABEL      = "mz_rt"  # "mz_rt" | "row_id"

# ---- 10. PERMANOVA (tests whether COLOR_BY groups actually differ) ------------
RUN_PERMANOVA  = True
N_PERMUTATIONS = 999
RANDOM_SEED    = 42

FIG_SIZE = (11, 8)
DPI      = 300

# =============================================================================
# END OF CONFIG
# =============================================================================


def log(msg):
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# Load and align
# -----------------------------------------------------------------------------
def load_data():
    log("=" * 70)
    log("LOADING")
    log("=" * 70)

    ft = pd.read_csv(FEATURE_TABLE)
    md = pd.read_csv(METADATA)

    # Drop stray unnamed columns MZmine sometimes leaves behind
    ft = ft.loc[:, ~ft.columns.str.startswith("Unnamed")]

    sample_cols = [c for c in ft.columns if c.endswith(SAMPLE_COL_SUFFIX)]
    if not sample_cols:
        raise ValueError(f"No columns ending in '{SAMPLE_COL_SUFFIX}' found.")

    # Feature identity columns we keep for labelling arrows
    id_cols = {}
    for src, dest in [("row ID", "row_id"),
                      ("row m/z", "mz"),
                      ("row retention time", "rt")]:
        if src in ft.columns:
            id_cols[dest] = ft[src].values

    feature_ids = (ft["row ID"].astype(str).values
                   if "row ID" in ft.columns
                   else np.arange(len(ft)).astype(str))

    # X: features (rows) x samples (columns)
    X = ft[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X.index = feature_ids
    X.columns = [c[: -len(SAMPLE_COL_SUFFIX)] for c in sample_cols]

    feature_info = pd.DataFrame(id_cols, index=feature_ids)

    md = md.set_index(METADATA_FILENAME_COL)
    shared = [c for c in X.columns if c in md.index]
    missing = [c for c in X.columns if c not in md.index]
    if missing:
        log(f"  ! {len(missing)} table columns have no metadata row and are dropped: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")

    X = X[shared]
    md = md.loc[shared]

    log(f"  feature table : {X.shape[0]} features x {X.shape[1]} matched columns")
    log(f"  metadata      : {md.shape[0]} rows, columns {list(md.columns)}")
    if TYPE_COLUMN in md.columns:
        log(f"  {TYPE_COLUMN} breakdown: {md[TYPE_COLUMN].value_counts().to_dict()}")
    return X, md, feature_info


# -----------------------------------------------------------------------------
# Blank subtraction
# -----------------------------------------------------------------------------
def blank_subtract(X, md):
    log("=" * 70)
    log("BLANK SUBTRACTION")
    log("=" * 70)

    if not BLANK_SUBTRACTION:
        log("  disabled")
        return X

    if TYPE_COLUMN not in md.columns:
        log(f"  ! '{TYPE_COLUMN}' not in metadata - skipping")
        return X

    blank_cols = md.index[md[TYPE_COLUMN].isin(BLANK_LABELS)].tolist()
    sample_cols = md.index[md[TYPE_COLUMN].isin(KEEP_TYPES)].tolist()

    if not blank_cols:
        log("  ! no blank columns found in the feature table - skipping")
        return X

    log(f"  blanks  ({len(blank_cols)}): {blank_cols}")
    log(f"  samples ({len(sample_cols)})")

    blank_max   = X[blank_cols].max(axis=1)
    sample_mean = X[sample_cols].mean(axis=1)

    # keep only if sample_mean > BLANK_RATIO * blank_max
    keep = sample_mean > (BLANK_RATIO * blank_max)
    # features absent from every blank are always kept (unless absent everywhere)
    keep |= (blank_max == 0) & (sample_mean > 0)

    if BLANK_DROP_IF_ABSENT_IN_SAMPLES:
        keep &= sample_mean > 0

    log(f"  rule: keep if mean(samples) > {BLANK_RATIO:g} x max(blanks)")
    log(f"  kept {int(keep.sum())} / {len(keep)} features "
        f"({100 * keep.sum() / len(keep):.1f}%)")

    return X.loc[keep]


# -----------------------------------------------------------------------------
# Processing steps
# -----------------------------------------------------------------------------
def prevalence_filter(X):
    if MIN_PREVALENCE and MIN_PREVALENCE > 0:
        keep = (X > 0).sum(axis=1) >= MIN_PREVALENCE
        log(f"  prevalence filter (>= {MIN_PREVALENCE} non-zero samples): "
            f"kept {int(keep.sum())} / {len(keep)}")
        X = X.loc[keep]
    # always drop features that are entirely zero or constant
    keep = X.std(axis=1) > 0
    dropped = int((~keep).sum())
    if dropped:
        log(f"  dropped {dropped} zero-variance features")
    return X.loc[keep]


def impute(X):
    log(f"  imputation: {IMPUTATION}")
    if IMPUTATION == "none":
        return X

    Xv = X.values.astype(float).copy()

    if IMPUTATION == "small":
        for i in range(Xv.shape[0]):
            row = Xv[i]
            nz = row[row > 0]
            if nz.size:
                row[row == 0] = IMPUTE_FRACTION * nz.min()
    elif IMPUTATION == "half_min":
        gmin = Xv[Xv > 0].min() if (Xv > 0).any() else 1.0
        Xv[Xv == 0] = gmin / 2.0
    elif IMPUTATION == "knn":
        from sklearn.impute import KNNImputer
        Xn = np.where(Xv == 0, np.nan, Xv)
        # KNNImputer works on rows = observations, so transpose to samples x features
        imp = KNNImputer(n_neighbors=KNN_NEIGHBOURS)
        Xv = imp.fit_transform(Xn.T).T
        Xv = np.nan_to_num(Xv, nan=0.0)
    else:
        raise ValueError(f"Unknown IMPUTATION: {IMPUTATION}")

    return pd.DataFrame(Xv, index=X.index, columns=X.columns)


def normalize_samples(X):
    log(f"  sample normalization: {NORMALIZATION}")
    if NORMALIZATION == "none":
        return X

    Xv = X.values.astype(float).copy()

    if NORMALIZATION == "tic":
        sums = Xv.sum(axis=0)
        sums[sums == 0] = 1.0
        Xv = Xv / sums * np.median(sums)
    elif NORMALIZATION == "median":
        meds = np.array([np.median(col[col > 0]) if (col > 0).any() else 1.0
                         for col in Xv.T])
        meds[meds == 0] = 1.0
        Xv = Xv / meds * np.median(meds)
    elif NORMALIZATION == "pqn":
        # integral-normalize first, then divide by the median quotient to a
        # reference spectrum (the median sample)
        sums = Xv.sum(axis=0)
        sums[sums == 0] = 1.0
        Xi = Xv / sums * np.median(sums)
        ref = np.median(Xi, axis=1)
        quot = np.full(Xi.shape[1], 1.0)
        for j in range(Xi.shape[1]):
            mask = (ref > 0) & (Xi[:, j] > 0)
            if mask.sum() > 0:
                quot[j] = np.median(Xi[mask, j] / ref[mask])
        quot[quot == 0] = 1.0
        Xv = Xi / quot
    else:
        raise ValueError(f"Unknown NORMALIZATION: {NORMALIZATION}")

    return pd.DataFrame(Xv, index=X.index, columns=X.columns)


def transform(X):
    log(f"  transform: {TRANSFORM}")
    if TRANSFORM == "none":
        return X
    Xv = X.values.astype(float)
    if TRANSFORM == "log10":
        Xv = np.log10(Xv + 1.0)
    elif TRANSFORM == "sqrt":
        Xv = np.sqrt(np.clip(Xv, 0, None))
    elif TRANSFORM == "fourth_root":
        Xv = np.power(np.clip(Xv, 0, None), 0.25)
    else:
        raise ValueError(f"Unknown TRANSFORM: {TRANSFORM}")
    return pd.DataFrame(Xv, index=X.index, columns=X.columns)


def scale_features(X):
    log(f"  feature scaling: {FEATURE_SCALING}")
    if FEATURE_SCALING == "none":
        return X

    Xv = X.values.astype(float)
    mean = Xv.mean(axis=1, keepdims=True)
    sd = Xv.std(axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0

    if FEATURE_SCALING == "center":
        Xv = Xv - mean
    elif FEATURE_SCALING == "auto":
        Xv = (Xv - mean) / sd
    elif FEATURE_SCALING == "pareto":
        Xv = (Xv - mean) / np.sqrt(sd)
    elif FEATURE_SCALING == "range":
        rng = Xv.max(axis=1, keepdims=True) - Xv.min(axis=1, keepdims=True)
        rng[rng == 0] = 1.0
        Xv = (Xv - mean) / rng
    else:
        raise ValueError(f"Unknown FEATURE_SCALING: {FEATURE_SCALING}")

    return pd.DataFrame(Xv, index=X.index, columns=X.columns)


# -----------------------------------------------------------------------------
# Distance + PCoA
# -----------------------------------------------------------------------------
NONNEGATIVE_METRICS = {"braycurtis", "canberra", "jaccard", "dice", "kulsinski"}


def distance_matrix(X):
    log("=" * 70)
    log("DISTANCE")
    log("=" * 70)

    M = X.values.T.astype(float)  # samples x features

    if BINARY_PRESENCE_ABSENCE:
        M = (M > 0).astype(float)
        log("  converted to presence/absence")

    if DISTANCE_METRIC in NONNEGATIVE_METRICS and M.min() < 0:
        warnings.warn(
            f"'{DISTANCE_METRIC}' needs non-negative data but scaling/centering "
            f"produced negatives. Shifting each feature to be non-negative. "
            f"Consider FEATURE_SCALING='none' or a metric like 'euclidean'."
        )
        M = M - M.min(axis=0, keepdims=True)

    metric = DISTANCE_METRIC
    kwargs = {}
    if metric == "jaccard":
        M = (M > 0).astype(bool)

    D = squareform(pdist(M, metric=metric, **kwargs))
    log(f"  metric: {metric}   matrix: {D.shape}")
    log(f"  distance range: {D[D > 0].min():.4f} - {D.max():.4f}")
    return D


def pcoa(D):
    log("=" * 70)
    log("PCoA")
    log("=" * 70)

    n = D.shape[0]
    A = -0.5 * (D ** 2)
    J = np.eye(n) - np.ones((n, n)) / n
    G = J @ A @ J
    G = (G + G.T) / 2.0   # enforce symmetry

    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    pos = vals > 1e-10
    coords = vecs[:, pos] * np.sqrt(vals[pos])

    neg_sum = vals[vals < 0].sum()
    if neg_sum < -1e-8:
        log(f"  note: negative eigenvalues present (sum {neg_sum:.4g}) - normal for "
            f"non-Euclidean metrics like Bray-Curtis")

    explained = vals[pos] / vals[pos].sum() * 100.0
    for i in range(min(5, len(explained))):
        log(f"  PCo{i + 1}: {explained[i]:5.2f}%")
    return coords, explained


def permanova(D, groups, n_perm=999, seed=42):
    """One-way PERMANOVA (Anderson 2001) on a distance matrix."""
    rng = np.random.default_rng(seed)
    g = pd.Series(groups).values
    n = len(g)
    levels = pd.unique(g)
    a = len(levels)
    if a < 2:
        return None

    D2 = D ** 2
    SS_total = D2[np.triu_indices(n, 1)].sum() / n

    def within_ss(labels):
        ss = 0.0
        for lv in levels:
            idx = np.where(labels == lv)[0]
            if len(idx) > 1:
                sub = D2[np.ix_(idx, idx)]
                ss += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
        return ss

    SS_w = within_ss(g)
    SS_a = SS_total - SS_w
    F_obs = (SS_a / (a - 1)) / (SS_w / (n - a))

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(g)
        ssw = within_ss(perm)
        F = ((SS_total - ssw) / (a - 1)) / (ssw / (n - a))
        if F >= F_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    R2 = SS_a / SS_total
    return F_obs, R2, p


# -----------------------------------------------------------------------------
# Biplot feature vectors
# -----------------------------------------------------------------------------
def feature_vectors(X, coords, ax1, ax2, feature_info):
    """Correlate each feature with the two plotted axes (vegan-style species scores)."""
    M = X.values.T.astype(float)        # samples x features
    c1, c2 = coords[:, ax1], coords[:, ax2]

    def corr(v, axis):
        vs = v.std(axis=0)
        ok = vs > 0
        r = np.zeros(v.shape[1])
        if axis.std() > 0:
            r[ok] = ((v[:, ok] - v[:, ok].mean(0)) *
                     (axis - axis.mean())[:, None]).sum(0) / (
                        (v.shape[0] - 1) * vs[ok] * axis.std(ddof=1))
        return r

    r1 = corr(M, c1)
    r2 = corr(M, c2)
    length = np.sqrt(r1 ** 2 + r2 ** 2)

    top = np.argsort(length)[::-1][:N_TOP_FEATURES]

    labels = []
    for i in top:
        fid = X.index[i]
        if FEATURE_LABEL == "mz_rt" and {"mz", "rt"}.issubset(feature_info.columns):
            row = feature_info.loc[fid]
            labels.append(f"{row['mz']:.3f}/{row['rt']:.2f}")
        else:
            labels.append(str(fid))

    return r1[top], r2[top], labels, length[top]


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or cov[0, 0] == 0 or cov[1, 1] == 0:
        return
    pear = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx = np.sqrt(1 + pear)
    ry = np.sqrt(1 - pear)
    ell = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    sx = np.sqrt(cov[0, 0]) * n_std
    sy = np.sqrt(cov[1, 1]) * n_std
    tr = (transforms.Affine2D()
          .rotate_deg(45)
          .scale(sx, sy)
          .translate(np.mean(x), np.mean(y)))
    ell.set_transform(tr + ax.transData)
    ax.add_patch(ell)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot(coords, explained, md, X, feature_info, permanova_result):
    ax1, ax2 = AXES[0] - 1, AXES[1] - 1
    if coords.shape[1] <= max(ax1, ax2):
        raise ValueError(f"Only {coords.shape[1]} axes available; AXES={AXES}")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    x = coords[:, ax1]
    y = coords[:, ax2]

    groups = md[COLOR_BY].astype(str).values if COLOR_BY in md.columns else np.array(["all"] * len(x))
    levels = sorted(pd.unique(groups))
    cmap = plt.get_cmap("tab10" if len(levels) <= 10 else "tab20")
    colors = {lv: cmap(i % cmap.N) for i, lv in enumerate(levels)}

    if SHAPE_BY and SHAPE_BY in md.columns:
        shapes = md[SHAPE_BY].astype(str).values
        slevels = sorted(pd.unique(shapes))
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
        mmap = {lv: markers[i % len(markers)] for i, lv in enumerate(slevels)}
    else:
        shapes = np.array([""] * len(x))
        mmap = {"": "o"}

    if DRAW_ELLIPSES:
        for lv in levels:
            m = groups == lv
            confidence_ellipse(x[m], y[m], ax, n_std=2.0,
                               facecolor=colors[lv], alpha=0.10,
                               edgecolor=colors[lv], linestyle="--", linewidth=1.2)

    for g in levels:
        for s in pd.unique(shapes):
            m = (groups == g) & (shapes == s)
            if not m.any():
                continue
            ax.scatter(x[m], y[m], c=[colors[g]], marker=mmap[s],
                       s=95, edgecolors="black", linewidths=0.6,
                       alpha=0.9, zorder=3)

    if LABEL_POINTS:
        for xi, yi, name in zip(x, y, md.index):
            ax.annotate(name, (xi, yi), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    # ---- biplot arrows ----
    if SHOW_BIPLOT_ARROWS:
        r1, r2, labels, _ = feature_vectors(X, coords, ax1, ax2, feature_info)
        span = max(np.abs(x).max(), np.abs(y).max()) * ARROW_SCALE
        for a, b, lab in zip(r1, r2, labels):
            ax.arrow(0, 0, a * span, b * span,
                     color="#444444", alpha=0.65, width=span * 0.0015,
                     head_width=span * 0.022, length_includes_head=True, zorder=2)
            ax.annotate(lab, (a * span * 1.06, b * span * 1.06),
                        fontsize=7, color="#222222", ha="center", va="center", zorder=4)

    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.axvline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel(f"PCo{AXES[0]} ({explained[ax1]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PCo{AXES[1]} ({explained[ax2]:.1f}%)", fontsize=12)

    title = f"PCoA biplot - {DISTANCE_METRIC}"
    sub = (f"norm={NORMALIZATION} | transform={TRANSFORM} | scaling={FEATURE_SCALING} | "
           f"imputation={IMPUTATION} | blank-sub={'on' if BLANK_SUBTRACTION else 'off'} | "
           f"{X.shape[0]} features")
    if permanova_result:
        F, R2, p = permanova_result
        sub += f"\nPERMANOVA on {COLOR_BY}: R2={R2:.3f}, p={p:.3f}"
    ax.set_title(title + "\n" + sub, fontsize=10)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[lv],
                      markeredgecolor="black", markersize=9, label=lv) for lv in levels]
    leg1 = ax.legend(handles=handles, title=COLOR_BY, loc="upper left",
                     bbox_to_anchor=(1.01, 1), fontsize=9, title_fontsize=10)
    ax.add_artist(leg1)

    if SHAPE_BY and SHAPE_BY in md.columns:
        handles2 = [Line2D([0], [0], marker=mmap[lv], color="black", linestyle="",
                           markerfacecolor="white", markersize=9, label=lv)
                    for lv in sorted(mmap)]
        ax.legend(handles=handles2, title=SHAPE_BY, loc="lower left",
                  bbox_to_anchor=(1.01, 0), fontsize=9, title_fontsize=10)

    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = f"pcoa_{DISTANCE_METRIC}_{NORMALIZATION}_{TRANSFORM}_{FEATURE_SCALING}"
    png = os.path.join(OUTPUT_DIR, stem + ".png")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, stem + ".pdf"), bbox_inches="tight")
    log(f"  saved {png}")

    # scree plot
    fig2, ax2b = plt.subplots(figsize=(6, 4))
    k = min(10, len(explained))
    ax2b.bar(np.arange(1, k + 1), explained[:k], color="#4C72B0")
    ax2b.set_xlabel("Axis")
    ax2b.set_ylabel("% variation explained")
    ax2b.set_title("Scree plot")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "scree.png"), dpi=DPI)

    return fig


# -----------------------------------------------------------------------------
def main():
    X, md, feature_info = load_data()

    X = blank_subtract(X, md)

    # keep only the sample types we want to ordinate
    keep_cols = md.index[md[TYPE_COLUMN].isin(KEEP_TYPES)].tolist() \
        if TYPE_COLUMN in md.columns else list(X.columns)
    X = X[keep_cols]
    md = md.loc[keep_cols]
    log(f"  ordinating {X.shape[1]} samples")

    log("=" * 70)
    log("PROCESSING")
    log("=" * 70)
    X = prevalence_filter(X)
    X = impute(X)
    X = normalize_samples(X)
    X = transform(X)
    X = scale_features(X)
    log(f"  final matrix: {X.shape[0]} features x {X.shape[1]} samples")

    D = distance_matrix(X)
    coords, explained = pcoa(D)

    perm = None
    if RUN_PERMANOVA and COLOR_BY in md.columns:
        perm = permanova(D, md[COLOR_BY].astype(str).values,
                         n_perm=N_PERMUTATIONS, seed=RANDOM_SEED)
        if perm:
            F, R2, p = perm
            log(f"  PERMANOVA ({COLOR_BY}): F={F:.3f}  R2={R2:.3f}  p={p:.4f} "
                f"({N_PERMUTATIONS} permutations)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = pd.DataFrame(coords[:, :5],
                       index=md.index,
                       columns=[f"PCo{i+1}" for i in range(min(5, coords.shape[1]))])
    out = out.join(md)
    out.to_csv(os.path.join(OUTPUT_DIR, "pcoa_coordinates.csv"))

    log("=" * 70)
    log("PLOTTING")
    log("=" * 70)
    plot(coords, explained, md, X, feature_info, perm)

    log("Done. Files are in ./" + OUTPUT_DIR)
    plt.show()


if __name__ == "__main__":
    main()
