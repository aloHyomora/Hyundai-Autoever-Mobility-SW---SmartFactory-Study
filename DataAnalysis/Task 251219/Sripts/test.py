import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DATAPATH = "DataAnalysis/Task 251219/Data/data251219.xlsx"


def _normalize_colname(name: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(name).strip().lower())


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    norm_to_actual = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_colname(cand)
        if key in norm_to_actual:
            return norm_to_actual[key]
    raise KeyError(
        f"컬럼을 찾지 못했습니다. candidates={candidates}, df.columns={list(df.columns)}"
    )


def plot_all_lots_one_param(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    outpath: Path | None,
    show: bool,
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6.5))
    ax.scatter(df[x_col], df[y_col], s=14, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Time index (every 5 seconds)" if x_col == "_t_idx" else "Time (seconds)")
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=160)

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Group by (Date, LoT), build time index within each group, then plot 3 global scatter plots (pH/Temperature/ProcessRate)."
    )
    parser.add_argument("--data", default=DEFAULT_DATAPATH, help="Excel file path")
    parser.add_argument("--sheet", default="data", help="Sheet name (default: data)")

    parser.add_argument("--date-col", default=None, help="Date column name (optional)")
    parser.add_argument("--lot-col", default=None, help="LoT column name (optional)")
    parser.add_argument("--time-col", default=None, help="Optional time column for sorting within group")

    parser.add_argument("--interval-sec", type=int, default=5, help="Sampling interval in seconds (default: 5)")
    parser.add_argument("--expected-n", type=int, default=69, help="Expected rows per (Date, LoT) group (default: 69)")
    parser.add_argument("--strict", action="store_true", help="Fail if any group size != expected-n")

    parser.add_argument("--x", choices=["idx", "sec"], default="idx", help="x-axis: idx(1..n) or sec(0..)")
    parser.add_argument("--outdir", default="plots", help="Output directory for png files")
    parser.add_argument("--no-save", action="store_true", help="Do not save png files")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    datapath = Path(args.data)
    print("Read data from:", datapath)

    df = pd.read_excel(datapath, sheet_name=args.sheet)

    date_col = args.date_col or _pick_column(df, ["Date", "date", "DATE", "일자", "날짜"])
    lot_col = args.lot_col or _pick_column(df, ["LoT", "LOT", "Lot", "lot", "로트"])
    ph_col = _pick_column(df, ["pH", "PH", "ph"])
    temp_col = _pick_column(df, ["Temperature", "TEMP", "Temp", "temperature"])
    rate_col = _pick_column(df, ["ProcessRate", "Process Rate", "processrate", "Rate", "process_rate"])

    # 그룹 내 시간 정렬: time_col 있으면 사용, 없으면 원본 행 순서
    if args.time_col and args.time_col in df.columns:
        order_col = args.time_col
    else:
        order_col = "_row_order"
        df[order_col] = range(len(df))

    df = df.sort_values([date_col, lot_col, order_col], kind="mergesort")

    # 각 (Date, LoT) 그룹에서 time index 생성: 1..n
    df["_t_idx"] = df.groupby([date_col, lot_col]).cumcount() + 1
    df["_t_sec"] = (df["_t_idx"] - 1) * int(args.interval_sec)
    x_col = "_t_idx" if args.x == "idx" else "_t_sec"

    # (Date, LoT)별 69개 확인
    sizes = df.groupby([date_col, lot_col]).size().reset_index(name="n")
    bad = sizes[sizes["n"] != int(args.expected_n)]
    print(f"Total groups: {len(sizes)}")
    print(f"Expected rows per group: {args.expected_n}")
    if len(bad) > 0:
        print(f"WARNING: {len(bad)} groups have unexpected size.")
        print(bad.head(20).to_string(index=False))
        if args.strict:
            raise ValueError("Group size mismatch detected (strict mode).")

    outdir = Path(args.outdir)
    save = not args.no_save

    # 전체 LoT를 합쳐서 파라미터별(=3장)만 그림
    plots = [
        (ph_col, "pH"),
        (temp_col, "Temperature"),
        (rate_col, "ProcessRate"),
    ]

    for y_col, label in plots:
        outpath = None
        if save:
            outpath = outdir / f"{label}.png"

        title = f"All LoTs over time ({label}) | x={args.x} | groups={len(sizes)}"
        plot_all_lots_one_param(
            df,
            x_col=x_col,
            y_col=y_col,
            title=title,
            outpath=outpath,
            show=args.show,
        )

    print(f"Done. Saved to: {outdir.resolve() if save else '(not saved)'}")


if __name__ == "__main__":
    main()