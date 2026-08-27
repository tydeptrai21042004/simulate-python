# 2026-08-27 update: official auto-data + accuracy-guarded LTH export

## One-command official run

```bash
pip install -r requirements-esp32-train.txt
pip install -e .
./RUN_ESP32_OFFICIAL.sh
```

Windows:

```bat
RUN_ESP32_OFFICIAL.bat
```

On the first official run, the pipeline creates `data/uwb_original_standard.mat` automatically by downloading the public GitHub repository and the separately hosted `Dyn_CIR_VAR.mat` referenced by the official README.

## What changed

- Official-data converter fixed for 4x3 XYZ anchors, actual `Dyn_var_CIRxx` / `Bg_var_CIRxx` variable names, and complex CIR magnitude.
- Structured Lottery Ticket selection now prunes Conv channels **and FC hidden neurons**.
- Channel ranking is dependency-aware across the compact physical graph.
- Surviving parameters are rewound to their original supernet initialization before retraining.
- Candidate architectures are tried smallest-first.
- Export requires quality preservation against the supernet: relative/absolute ToF MAE, NLL, and outlier BCE limits.
- `best_student.pt` is always a structured rewound LTH ticket.
- Random compact training remains as a scientific control but is blocked from normal export.
- Direct `export_esp32.py` also rejects non-LTH checkpoints unless the explicit debug override is provided.
- INT8 calibration now samples across clean and corrupted training examples rather than simply taking the first clean block.

## Verified smoke result

The short smoke run is not a final accuracy experiment, but confirms the complete mechanism:

- selected model type: structured rewound LTH ticket;
- architecture: Conv `[6, 8, 8]`, FC hidden `8`;
- trainable parameters: `827`;
- raw INT8 weight/bias blob: `904 bytes`;
- random compact control: trained/reported but not exported;
- automated tests: `61/61 passed`.

Use `configs/esp32s3_official.yaml` for final experimental results. It has a strict quality guard and automatically fetches the official dataset when needed.
