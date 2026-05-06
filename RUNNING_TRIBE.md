# Running TRIBE v2 for Social Video Brain-Response Checks

This repo is Meta/Facebook Research's TRIBE v2 code. It predicts fMRI-like brain responses from naturalistic video, audio, and text, mapped to the fsaverage5 cortical surface. The pretrained checkpoint is hosted as `facebook/tribev2` on Hugging Face.

## What This Can and Cannot Tell You

TRIBE v2 predicts an average-subject cortical response. It does not directly output "boring" or "engaging." For social media videos, treat it as an exploratory signal:

- Compare videos or edits against each other, not as an absolute truth.
- Look at response strength and dynamics over time.
- Use it alongside retention, watch time, rewatches, likes, comments, and human review.
- Do not treat predicted activation as medical or psychological diagnosis.

The model license is CC-BY-NC-4.0, so commercial use needs careful license review.

## Local Reality Check

A weak laptop can probably run short clips, but slowly. The first run downloads:

- TRIBE checkpoint from Hugging Face, about 709 MB.
- Feature extractor weights for video/audio/text, which can add multiple GB.
- WhisperX models if you use speech transcription.

Best local workflow:

1. Test on 5-15 second clips first.
2. Start without transcript/text features.
3. Render only the first few timesteps until the setup works.
4. Use Google Colab or a rented GPU for longer videos or batch scoring.

## Setup

Use Python 3.11, not the default Python 3.13:

```bash
uv venv --python /opt/homebrew/bin/python3.11 .venv
source .venv/bin/activate
uv pip install -e ".[plotting]"
```

If you want full text features, log into Hugging Face and make sure your account has access to the gated LLaMA model used by the TRIBE config:

```bash
huggingface-cli login
```

## Fast Local Smoke Test

This skips speech transcription and text features, using video/audio only:

```bash
source .venv/bin/activate
python scripts/run_tribe_video.py /path/to/video.mp4 \
  --out-dir outputs/test-video \
  --max-timesteps 15 \
  --render-mp4
```

Outputs:

- `outputs/test-video/events.tsv`
- `outputs/test-video/preds.npy`
- `outputs/test-video/segments.pkl`
- `outputs/test-video/summary.json`
- `outputs/test-video/brain_activity.mp4` if `--render-mp4` is used

## Full Multimodal Run

This runs WhisperX transcription and enables text events. It is slower and more likely to need a GPU or patience:

```bash
source .venv/bin/activate
python scripts/run_tribe_video.py /path/to/video.mp4 \
  --out-dir outputs/full-video \
  --with-transcript \
  --render-mp4
```

On CPU, the local code has been patched to call WhisperX with `int8` and a smaller batch size. On CUDA it still uses `float16`.

## Recommended Cloud Path

For practical work, run inference in Google Colab with a GPU:

```bash
uv pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
```

Then upload videos to Colab, run the notebook-style inference, and download the `preds.npy` and rendered MP4. This is the most realistic option for multiple social videos.

## Interpreting "Boring"

A pragmatic first score can be:

- Mean absolute activation over time.
- Temporal variance, meaning how much the response changes second to second.
- Peak count or peak density.
- Drop-off after the first 3-5 seconds.
- Comparison against your best-performing videos.

Do not use a single activation threshold. Build a small dataset of your own videos with real performance metrics, then correlate TRIBE-derived features with retention and engagement.
