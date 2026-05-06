"""Decode each Destrieux parcel against Neurosynth meta-analytic terms.

Direct vectorized implementation: for each parcel ROI, find studies that
report any coordinate inside the ROI, then score each cognitive term by the
t-statistic of its tf-idf weight in in-ROI vs out-ROI studies. This matches
the spirit of Neurosynth's ROI association without the heavy MKDA pass.

Outputs tribe_ui/destrieux_neurosynth.json mapping
"<HEMI>_<atlas_label>" -> [{term, z}, ...] (top N terms by z).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import datasets
from nimare.extract import fetch_neurosynth


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "tribe_ui" / "destrieux_neurosynth.json"
DATA_DIR = ROOT / ".cache" / "neurosynth"
TOP_N = 12
MIN_STUDIES = 30  # require at least this many in-ROI studies for stable z

# Curated cognitive / mental-state vocabulary. Anatomical/sensory terms are
# kept minimally so output is interpretable in psychological terms.
COGNITIVE_TERMS = {
    "emotion", "emotional", "fear", "anxiety", "anger", "sad", "sadness", "happy",
    "happiness", "reward", "value", "loss", "punishment", "motivation", "incentive",
    "attention", "salience", "salient", "inhibition", "conflict", "control", "executive",
    "working memory", "memory", "episodic", "autobiographical", "encoding",
    "retrieval", "recall", "recognition", "semantic", "language", "speech",
    "phonological", "syntactic", "reading", "comprehension", "narrative", "word",
    "social", "mentalizing", "empathy", "self", "self referential", "moral",
    "face", "faces", "body", "biological motion", "agency", "intention", "intentions",
    "decision", "choice", "risk", "uncertainty", "prediction", "prediction error",
    "pain", "interoception", "disgust", "taste", "olfactory", "smell",
    "spatial", "navigation", "imagery", "mental imagery",
    "action", "motor", "movement", "imitation", "observation", "reaching",
    "default mode", "salience network",
    "anticipation", "expectation", "regulation", "appraisal", "arousal",
    "music", "auditory", "visual", "tactile", "touch", "rhythm",
    "reasoning", "inference", "abstract", "concept", "category",
    "fearful", "happy", "anger", "anxiety", "stress", "threat",
    "empathic", "compassion", "trust", "cooperation", "competition",
    "speech production", "speech perception", "voice", "voices",
    "object", "objects", "scene", "scenes", "place", "places",
    "number", "numerical", "magnitude", "quantity",
    "rest", "resting state", "mind wandering",
}


def load_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] Neurosynth -> {DATA_DIR}", flush=True)
    result = fetch_neurosynth(
        data_dir=str(DATA_DIR),
        version="7",
        overwrite=False,
        return_type="dataset",
        source="abstract",
        vocab="terms",
    )
    ds = result[0] if isinstance(result, list) else result
    print(f"[load] dataset ready ({len(ds.ids)} studies)", flush=True)
    return ds


def build_term_matrix(ds):
    annot = ds.annotations
    term_cols = [c for c in annot.columns if c.startswith("terms_abstract_tfidf__")]
    keep_idx = []
    keep_names = []
    for i, col in enumerate(term_cols):
        clean = col.split("__")[-1].strip().lower()
        if clean in COGNITIVE_TERMS:
            keep_idx.append(i)
            keep_names.append(clean)
    print(f"[load] keeping {len(keep_names)} cognitive terms", flush=True)
    tfidf = annot[[term_cols[i] for i in keep_idx]].values.astype(np.float32)
    study_ids = annot["study_id"].astype(str).values
    return tfidf, keep_names, study_ids


def coords_to_voxels(coords_df, affine, shape):
    xyz = coords_df[["x", "y", "z"]].values.astype(np.float64)
    inv = np.linalg.inv(affine)
    vox = nib.affines.apply_affine(inv, xyz).round().astype(int)
    valid = (
        (vox[:, 0] >= 0) & (vox[:, 0] < shape[0])
        & (vox[:, 1] >= 0) & (vox[:, 1] < shape[1])
        & (vox[:, 2] >= 0) & (vox[:, 2] < shape[2])
    )
    return vox, valid


def parcel_iter(atlas_data, labels):
    for idx, raw in enumerate(labels):
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
        if idx == 0 or name.lower() in {"background", "unknown"}:
            continue
        parts = name.split(" ", 1)
        if len(parts) != 2 or parts[0] not in {"L", "R"}:
            continue
        hemi, label = parts
        if label == "Medial_wall":
            continue
        mask = (atlas_data == idx)
        if mask.sum() < 5:
            continue
        yield f"{hemi}_{label}", mask


def main():
    ds = load_dataset()
    tfidf, term_names, ann_study_ids = build_term_matrix(ds)
    coords = ds.coordinates
    coord_studies = coords["study_id"].astype(str).values

    print("[atlas] fetching volumetric Destrieux", flush=True)
    atlas = datasets.fetch_atlas_destrieux_2009(verbose=0)
    img = nib.load(atlas["maps"])
    atlas_data = np.asarray(img.get_fdata(), dtype=np.int32)
    labels = list(atlas["labels"])

    print("[atlas] mapping Neurosynth coords to atlas voxels", flush=True)
    vox, valid = coords_to_voxels(coords, img.affine, atlas_data.shape)

    # Map each annotation study_id row to its index for fast lookup
    study_to_row = {sid: i for i, sid in enumerate(ann_study_ids)}

    results: dict[str, list[dict]] = {}
    for key, mask in parcel_iter(atlas_data, labels):
        # find coordinates inside this ROI
        in_coord = np.zeros(len(coords), dtype=bool)
        v = vox[valid]
        hits = mask[v[:, 0], v[:, 1], v[:, 2]]
        in_coord[valid] = hits
        if not in_coord.any():
            results[key] = []
            print(f"[skip] {key}: no coords", flush=True)
            continue
        # studies with at least one coord in ROI
        in_study_ids = np.unique(coord_studies[in_coord])
        sel = np.zeros(len(ann_study_ids), dtype=bool)
        for sid in in_study_ids:
            r = study_to_row.get(sid)
            if r is not None:
                sel[r] = True
        n_in = int(sel.sum())
        if n_in < MIN_STUDIES:
            results[key] = []
            print(f"[skip] {key}: only {n_in} studies", flush=True)
            continue
        n_out = len(ann_study_ids) - n_in
        in_mean = tfidf[sel].mean(axis=0)
        out_mean = tfidf[~sel].mean(axis=0)
        in_var = tfidf[sel].var(axis=0)
        out_var = tfidf[~sel].var(axis=0)
        se = np.sqrt(in_var / n_in + out_var / max(n_out, 1))
        z = (in_mean - out_mean) / np.where(se > 0, se, 1.0)
        order = np.argsort(-z)
        top = []
        for j in order[: TOP_N * 3]:
            term = term_names[j]
            zv = float(z[j])
            if not np.isfinite(zv) or zv <= 0:
                continue
            top.append({"term": term, "z": round(zv, 3)})
            if len(top) >= TOP_N:
                break
        results[key] = top
        print(f"[ok] {key} ({n_in} studies): {[t['term'] for t in top[:6]]}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {OUT_PATH} ({len(results)} parcels)", flush=True)


if __name__ == "__main__":
    main()
