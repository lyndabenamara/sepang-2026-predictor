# run_weekend.py: the whole Sepang prediction in one go.
#   ./.venv/bin/python run_weekend.py
# Re-measures tyres from every practice session so far, rebuilds the strategies,
# pulls qualifying (or a stand-in if it hasn't happened), then runs the prediction.

import runpy

for step in ["sepang_tyres", "sepang_strategies", "sepang_weekend", "predict_sepang"]:
    print(f"\n{'#' * 70}\n#  {step}.py\n{'#' * 70}", flush=True)
    runpy.run_path(f"{step}.py", run_name="__main__")
