"""
CE Step 5b — Counterfactual CCE + Ensured CCE Plots + Narratives

Generates 6 CCE instances (3 at-risk + 3 normal) per model with:
  - CCE plot: "what-if" alternatives with uncertainty intervals
  - Ensured CCE plot: filtered — only reliable alternatives kept
    (Lofstrom et al., arXiv:2410.05479)
  - Narrative .txt for each

Output : MAIN_CE_OUTPUT/counterfactual/<model_key>/
Runtime: ~25-40 min for all 10 models

Run after ce_factual.py. Can run in parallel with ce_global.py.
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
# PER-MODEL CCE
# =============================================================================

def run_cce_for_model(model_key, model_name, model,
                      X_cal, y_cal, X_test, y_test, features, config):
    print(f"\n{'='*70}")
    print(f"  CCE: {model_name}")
    print(f"{'='*70}")

    out_dir = config.output_dir / 'counterfactual' / model_key
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
    n_each      = config.n_counterfactual // 2

    ccf_idx = np.concatenate([
        rng.choice(at_risk_idx, min(n_each, len(at_risk_idx)), replace=False),
        rng.choice(normal_idx,  min(n_each, len(normal_idx)),  replace=False),
    ])
    labels = (
        ['at_risk'] * min(n_each, len(at_risk_idx)) +
        ['normal']  * min(n_each, len(normal_idx))
    )

    print(f"  Generating {len(ccf_idx)} CCE instances...")
    for idx, label in zip(ccf_idx, labels):
        try:
            counter = ce.explore_alternatives(X_test[[idx]])
            single  = counter[0]

            # Standard CCE
            cce_path = out_dir / f'cce_{label}_{idx}.png'
            ok = _save_plot(single, str(cce_path))
            _save_narrative(single, str(cce_path))
            status = "[saved]" if ok else "[plot warn]"
            print(f"    {status} cce_{label}_{idx}.png")

            # Ensured CCE — only reliable alternatives
            try:
                ensured      = single.ensured_explanations()
                ensured_path = out_dir / f'cce_ensured_{label}_{idx}.png'
                ok2 = _save_plot(ensured, str(ensured_path))
                _save_narrative(ensured, str(ensured_path))
                status2 = "[saved]" if ok2 else "[plot warn]"
                print(f"    {status2} cce_ensured_{label}_{idx}.png")
            except Exception as ee:
                print(f"    [warn] Ensured CCE instance {idx}: {ee}")

        except Exception as e:
            print(f"    [warn] CCE instance {idx}: {e}")

    print(f"  Done → {out_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not CE_AVAILABLE:
        return

    config = Config()
    _, _, X_cal, y_cal, X_test, y_test, features = load_data(config)

    print(f"\n  Running CCE for {len(MODEL_REGISTRY)} models "
          f"({config.n_counterfactual} instances each)...")

    for model_key, model_name, fmt in MODEL_REGISTRY:
        model = _load_single_model(model_key, model_name, fmt, config)
        if model is None:
            continue
        try:
            run_cce_for_model(
                model_key, model_name, model,
                X_cal, y_cal, X_test, y_test, features, config,
            )
        finally:
            del model   # release RAM before loading next model

    print("\n" + "="*70)
    print("[DONE] ce_cce.py complete.")
    print(f"Outputs: {config.output_dir / 'counterfactual'}")
    print("Next   : python ce_global.py  (or  python ce_mondrian.py)")
    print("="*70)


if __name__ == '__main__':
    main()
