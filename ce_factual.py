"""
CE Step 5a — Factual CE Plots + Narratives

Generates 6 factual CE plots (3 at-risk + 3 normal) per model with:
  - Feature weights with Venn-Abers confidence intervals (bar chart)
  - Human-readable narrative explanation (_narrative.txt)

Output : MAIN_CE_OUTPUT/factual/<model_key>/
Runtime: ~20-30 min for all 10 models

Run this FIRST — gives you visual per-patient results quickly.
After this, run: ce_cce.py → ce_global.py → ce_mondrian.py
"""

import numpy as np
from calibrated_explanations import CalibratedExplainer

from ce_utils import (
    Config, MODEL_REGISTRY,
    load_data, _load_single_model,
    _save_plot, _save_narrative,
    CE_AVAILABLE,
)


# =============================================================================
# PER-MODEL FACTUAL CE
# =============================================================================

def run_factual_for_model(model_key, model_name, model,
                          X_cal, y_cal, X_test, y_test, features, config):
    print(f"\n{'='*70}")
    print(f"  Factual CE: {model_name}")
    print(f"{'='*70}")

    out_dir = config.output_dir / 'factual' / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Initializing CalibratedExplainer (n_cal={len(X_cal)})...")
    try:
        ce = CalibratedExplainer(
            model, X_cal, y_cal,
            feature_names=features,
            mode='classification',
        )
        print(f"  [OK] CalibratedExplainer ready")
    except Exception as e:
        print(f"  [ERROR] Init failed: {e}")
        return

    rng         = np.random.default_rng(config.random_state)
    at_risk_idx = np.where(y_test == 1)[0]
    normal_idx  = np.where(y_test == 0)[0]
    n_each      = config.n_factual_plots // 2

    plot_idx = np.concatenate([
        rng.choice(at_risk_idx, min(n_each, len(at_risk_idx)), replace=False),
        rng.choice(normal_idx,  min(n_each, len(normal_idx)),  replace=False),
    ])
    labels = (
        ['at_risk'] * min(n_each, len(at_risk_idx)) +
        ['normal']  * min(n_each, len(normal_idx))
    )

    print(f"  Generating {len(plot_idx)} factual CE plots...")
    saved = 0
    for idx, label in zip(plot_idx, labels):
        try:
            exp    = ce.explain_factual(X_test[[idx]])
            single = exp[0]

            path = out_dir / f'ce_factual_{label}_{idx}.png'
            ok   = _save_plot(single, str(path))
            _save_narrative(single, str(path))
            if ok:
                saved += 1
                print(f"    [saved] {path.name}")
            else:
                print(f"    [warn]  plot not saved for instance {idx}")
        except Exception as e:
            print(f"    [warn]  instance {idx}: {e}")

    print(f"  Done: {saved}/{len(plot_idx)} plots → {out_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not CE_AVAILABLE:
        return

    config = Config()
    _, _, X_cal, y_cal, X_test, y_test, features = load_data(config)

    print(f"\n  Running factual CE for {len(MODEL_REGISTRY)} models "
          f"({config.n_factual_plots} plots each)...")

    for model_key, model_name, fmt in MODEL_REGISTRY:
        model = _load_single_model(model_key, model_name, fmt, config)
        if model is None:
            continue
        try:
            run_factual_for_model(
                model_key, model_name, model,
                X_cal, y_cal, X_test, y_test, features, config,
            )
        finally:
            del model   # release RAM before loading next model

    print("\n" + "="*70)
    print("[DONE] ce_factual.py complete.")
    print(f"Outputs: {config.output_dir / 'factual'}")
    print("Next   : python ce_cce.py")
    print("="*70)


if __name__ == '__main__':
    main()
