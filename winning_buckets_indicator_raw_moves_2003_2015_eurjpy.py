from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).parent
IN_PATH = OUT_DIR / "winning_buckets_indicator_lift_2003_2015_eurjpy.csv"
MIN_TRADES = 30


def main():
    df = pd.read_csv(IN_PATH)
    df = df[df["grp_trades"] >= MIN_TRADES].copy()

    # Keep only raw indicator behavior (no lift fields in the main ranking)
    raw = df[
        [
            "side",
            "bucket",
            "indicator",
            "indicator_bucket",
            "grp_trades",
            "avg_pips",
            "pivot_first_rate",
            "base_avg_pips",
            "base_pivot_first",
        ]
    ].copy()
    raw = raw.sort_values(["avg_pips", "grp_trades"], ascending=[False, False]).reset_index(drop=True)

    # Also provide per-distance-bucket best indicator states by raw avg_pips.
    best_per_bucket = (
        raw.sort_values(["side", "bucket", "avg_pips", "grp_trades"], ascending=[True, True, False, False])
        .groupby(["side", "bucket"], as_index=False)
        .head(5)
        .reset_index(drop=True)
    )

    out = {
        "scope": "winning distance buckets only",
        "note": "raw avg_pips per indicator state (no lift subtraction)",
        "min_trades": MIN_TRADES,
        "top_overall_raw": raw.head(50).to_dict(orient="records"),
        "top5_per_bucket_raw": best_per_bucket.to_dict(orient="records"),
    }

    out_json = OUT_DIR / "winning_buckets_indicator_raw_moves_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "winning_buckets_indicator_raw_moves_2003_2015_eurjpy.csv"
    out_bucket_csv = OUT_DIR / "winning_buckets_indicator_raw_moves_top5_per_bucket_2003_2015_eurjpy.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    raw.to_csv(out_csv, index=False)
    best_per_bucket.to_csv(out_bucket_csv, index=False)

    print("TOP RAW OVERALL")
    print(raw.head(20).to_string(index=False))
    print("\nTOP 5 PER BUCKET")
    print(best_per_bucket.to_string(index=False))
    print(f"saved={out_json}")
    print(f"saved={out_csv}")
    print(f"saved={out_bucket_csv}")


if __name__ == "__main__":
    main()

