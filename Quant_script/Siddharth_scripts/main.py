from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import List

import pandas as pd

import Quant_script.Siddharth_scripts.all_strategy as all_strategy


def parse_symbols(value: str) -> List[str]:
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def parse_strategy_names(value: str) -> List[str]:
    names = [s.strip() for s in value.split(",") if s.strip()]
    if not names:
        return sorted(all_strategy.strategy_registry().keys())
    return names


def parse_ddmmyyyy(value: str) -> date:
    try:
        return datetime.strptime(value.strip(), "%d/%m/%Y").date()
    except ValueError as exc:
        raise ValueError("--as-of-date must be in dd/mm/yyyy format") from exc


def resolve_anchor_date(value: str) -> date:
    if value.strip():
        return parse_ddmmyyyy(value)

    user_value = input("Enter anchor date (dd/mm/yyyy): ").strip()
    if not user_value:
        raise ValueError("Anchor date is required.")
    return parse_ddmmyyyy(user_value)


def write_strategy_outputs(
    output_dir: Path,
    run_date: date,
    execution: all_strategy.StrategyExecution,
    write_detail_file: bool,
) -> dict[str, Path]:
    date_label = run_date.strftime("%d_%m_%Y")

    bull_file = output_dir / f"{execution.name}_bull_{date_label}.csv"
    bear_file = output_dir / f"{execution.name}_bear_{date_label}.csv"
    detail_file = output_dir / f"{execution.name}_all_{date_label}.csv"

    # Output label mapping is intentionally swapped to match expected portal semantics.
    output_bull = execution.bearish
    output_bear = execution.bullish

    output_bull.to_csv(bull_file, index=False)
    output_bear.to_csv(bear_file, index=False)
    if write_detail_file:
        execution.results.to_csv(detail_file, index=False)

    return {
        "bull": bull_file,
        "bear": bear_file,
        "all": detail_file if write_detail_file else Path(""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrator for running multiple strategy scripts and exporting standardized outputs"
    )
    parser.add_argument(
        "--strategies",
        default="",
        help=(
            "Comma-separated strategy names. Defaults to all from all_strategy.py. "
            "Example: weekly_vs_daily_sweep,inside_bar_pattern_daily_sweep,double_sweep,daily_fvg_sweep,ema5_sweep"
        ),
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols. If empty, runs all NSE futures stock symbols.",
    )
    parser.add_argument(
        "--as-of-date",
        default="",
        help="Anchor date in dd/mm/yyyy format (default: today)",
    )
    parser.add_argument(
        "--output-dir",
        default="strategy_outputs",
        help="Directory where output CSV files are stored",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-symbol strategy logs where supported",
    )
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Print extracted intermediate values where supported",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution across selected strategies",
    )
    parser.add_argument(
        "--skip-detail-files",
        action="store_true",
        help="Do not write per-strategy *_all_* CSV files",
    )
    parser.add_argument(
        "--skip-combined-file",
        action="store_true",
        help="Do not write all_strategy_matches_* combined CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    as_of_date = resolve_anchor_date(args.as_of_date)

    strategy_names = parse_strategy_names(args.strategies)

    if args.symbols.strip():
        symbols = parse_symbols(args.symbols)
    else:
        symbols = all_strategy.load_default_symbols()

    if not symbols:
        raise ValueError("No symbols resolved. Provide --symbols or check NSE symbol loader.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running strategies: {', '.join(strategy_names)}")
    print(f"Symbols count: {len(symbols)}")
    print(f"As-of date: {as_of_date.strftime('%d/%m/%Y')}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel strategies: {not args.no_parallel}")

    executions = all_strategy.run_strategies(
        strategy_names=strategy_names,
        symbols=symbols,
        as_of_date=as_of_date,
        verbose=args.verbose,
        print_values=args.print_values,
        parallel=not args.no_parallel,
    )

    summary_rows = []
    combined_rows = []

    for execution in executions:
        output_bull = execution.bearish
        output_bear = execution.bullish
        files = write_strategy_outputs(
            output_dir,
            as_of_date,
            execution,
            write_detail_file=not args.skip_detail_files,
        )

        summary_rows.append(
            {
                "strategy": execution.name,
                "total_symbols": len(execution.results),
                "bull_count": len(output_bull),
                "bear_count": len(output_bear),
                "bull_file": str(files["bull"]),
                "bear_file": str(files["bear"]),
                "all_file": str(files["all"]) if str(files["all"]) else "",
            }
        )

        if not args.skip_combined_file and not output_bull.empty:
            tmp = output_bull.copy()
            tmp.insert(0, "signal_type", "bull")
            tmp.insert(0, "strategy", execution.name)
            combined_rows.append(tmp)

        if not args.skip_combined_file and not output_bear.empty:
            tmp = output_bear.copy()
            tmp.insert(0, "signal_type", "bear")
            tmp.insert(0, "strategy", execution.name)
            combined_rows.append(tmp)

    date_label = as_of_date.strftime("%d_%m_%Y")
    summary_file = output_dir / f"all_strategy_summary_{date_label}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    combined_file = output_dir / f"all_strategy_matches_{date_label}.csv"
    if not args.skip_combined_file:
        if combined_rows:
            pd.concat(combined_rows, ignore_index=True).to_csv(combined_file, index=False)
        else:
            pd.DataFrame(columns=["strategy", "signal_type", "symbol", "tradingview_link"]).to_csv(
                combined_file,
                index=False,
            )

    print("\n=== Run Summary ===")
    for row in summary_rows:
        print(
            f"{row['strategy']}: bull={row['bull_count']}, "
            f"bear={row['bear_count']}, total={row['total_symbols']}"
        )

    print(f"Summary file: {summary_file}")
    if not args.skip_combined_file:
        print(f"Combined matches file: {combined_file}")


if __name__ == "__main__":
    main()
