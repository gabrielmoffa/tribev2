# TRIBE v2 UI Fork

This adds a FastAPI + vanilla-JS review UI around TRIBE v2 inference. The
backend is `review_app.py`; the frontend lives in `review_static/`.

## Local demo

```bash
pip install -e ".[plotting,ui]"
uvicorn review_app:app --reload
```

Then open http://127.0.0.1:8000. The app can load an existing output folder
such as `outputs/img_3578_av`, or run a new video through TRIBE.

## Hugging Face Jobs

The local UI submits inference to Hugging Face Jobs. Authenticate once with
`hf auth login` or export `HF_TOKEN`; the app creates a private
`<your-username>/tribe-job-artifacts` artifact store for job inputs and outputs.
Set `HF_NAMESPACE` only if you intentionally want jobs and artifacts under an
organization namespace.

## UI Flow

1. Upload a video.
2. Optionally enable transcript text.
3. Run full TRIBE inference.
4. Review the normalized video next to the 3D cortical activation movie.
5. Inspect the activation timeline and top cortical parcels.

## Functional Associations

The "Active Functional Associations" panel offers two views per parcel:

- **Anatomy** — hand-curated cognitive/anatomical descriptions in
  `tribe_ui/destrieux_functions.py`.
- **Neurosynth decoded** — meta-analytic terms over-represented in fMRI
  papers that report activation peaks inside the parcel. Built once via
  `python scripts/build_neurosynth_decoding.py`, which downloads the
  Neurosynth v7 corpus (~14k studies) and writes
  `tribe_ui/destrieux_neurosynth.json`.

For each parcel, the decoded score for a term is roughly
`(mean tf-idf in in-ROI papers − mean tf-idf in out-of-ROI papers) / SE`,
i.e. a t-statistic of how much more often that term appears in abstracts
of papers activating this parcel vs. the baseline across the literature.
With ~14k studies, idiosyncratic per-paper noise averages out — only
systematic over-representation survives.

### Caveats for the decoded view

- **Forward inference, not reverse.** A high score for "auditory" means
  "papers activating this parcel mention 'auditory' more than baseline,"
  not "activity here implies auditory processing." Many regions show up
  for many terms.
- **Crude term granularity.** "Memory" lumps working / episodic /
  autobiographical, etc.
- **Publication bias.** Regions tied to popular paradigms (faces,
  language, reward) are over-represented relative to less-studied
  cognitive domains.
- **Peak coordinates, not full maps.** Neurosynth indexes reported
  activation peaks, so spatial precision is roughly ~1 cm and a parcel
  is "in" if any peak from a study falls inside it.
