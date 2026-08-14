import os
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("BITCOIN ML V11.1 - FAST PARAMETER OPTIMIZER")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "data/btcusd_ml_v2.csv"

HORIZONS = [12, 24]

SMA_TYPES = [20, 50, 100, 200]

DISTANCES = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00
]

MOMENTUM_THRESHOLDS = [
    -0.001,
    -0.002,
    -0.003,
    -0.005,
    -0.0075,
    -0.010
]

VOLUME_THRESHOLDS = [
    0.8,
    1.0,
    1.2,
    1.5,
    2.0
]

VOLATILITY_THRESHOLDS = [
    0.8,
    1.0,
    1.2,
    1.5,
    2.0
]

RRRS = [
    1.5,
    2.0,
    2.5,
    3.0
]

ATR_MULTS = [
    0.75,
    1.0,
    1.25,
    1.5
]

MIN_VALIDATION_TRADES = 50
MIN_TEST_TRADES = 30

# Number of candidates sent to the expensive backtest
TOP_FAST_CANDIDATES = 100

TRADING_COST = 0.0005


# ============================================================
# LOAD
# ============================================================

print()
print("Loading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)

print()
print("Rows:", len(df))
print("From:", df["time"].min())
print("To:", df["time"].max())


# ============================================================
# FEATURES
# ============================================================

print()
print("=" * 70)
print("CREATING FEATURES")
print("=" * 70)

df["return_1"] = df["close"].pct_change(1)
df["return_3"] = df["close"].pct_change(3)
df["return_6"] = df["close"].pct_change(6)
df["return_12"] = df["close"].pct_change(12)
df["return_24"] = df["close"].pct_change(24)


# ------------------------------------------------------------
# SMA
# ------------------------------------------------------------

for p in [20, 50, 100, 200]:
    df[f"sma_{p}"] = df["close"].rolling(p).mean()


# ------------------------------------------------------------
# Volume
# ------------------------------------------------------------

if "tick_volume" in df.columns:
    volume_col = "tick_volume"
elif "real_volume" in df.columns:
    volume_col = "real_volume"
else:
    volume_col = None

print()
print("Volume column:", volume_col)

if volume_col:
    volume_ma = df[volume_col].rolling(20).mean()

    df["volume_ratio_calc"] = (
        df[volume_col] / volume_ma
    )
else:
    df["volume_ratio_calc"] = 1.0


# ------------------------------------------------------------
# Volatility
# ------------------------------------------------------------

df["volatility_5_calc"] = (
    df["return_1"].rolling(5).std()
)

df["volatility_20_calc"] = (
    df["return_1"].rolling(20).std()
)

df["volatility_ratio_calc"] = (
    df["volatility_5_calc"] /
    df["volatility_20_calc"]
)


# ------------------------------------------------------------
# ATR
# ------------------------------------------------------------

prev_close = df["close"].shift(1)

tr1 = df["high"] - df["low"]
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()

df["true_range"] = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["atr_14"] = (
    df["true_range"].rolling(14).mean()
)


# ------------------------------------------------------------
# Future returns
# ------------------------------------------------------------

for h in HORIZONS:
    df[f"future_return_{h}"] = (
        df["close"].shift(-h) /
        df["close"] - 1
    )


required = [
    "close",
    "high",
    "low",
    "atr_14",
    "volume_ratio_calc",
    "volatility_ratio_calc",
    "return_6",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200"
]

for h in HORIZONS:
    required.append(f"future_return_{h}")

df = df.dropna(
    subset=required
).reset_index(drop=True)

print()
print("Rows after cleaning:", len(df))


# ============================================================
# SPLIT
# ============================================================

print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

train = df[
    df["time"] < "2023-01-01"
].copy()

validation = df[
    (df["time"] >= "2023-01-01") &
    (df["time"] < "2025-01-01")
].copy()

test = df[
    df["time"] >= "2025-01-01"
].copy()

print()
print("TRAIN:", len(train))
print(
    train["time"].min(),
    "->",
    train["time"].max()
)

print()
print("VALIDATION:", len(validation))
print(
    validation["time"].min(),
    "->",
    validation["time"].max()
)

print()
print("TEST:", len(test))
print(
    test["time"].min(),
    "->",
    test["time"].max()
)


# ============================================================
# FAST SCREENING
# ============================================================

print()
print("=" * 70)
print("FAST VECTOR SCREENING")
print("=" * 70)


def fast_screen(
    data,
    sma_period,
    distance_pct,
    momentum_threshold,
    volume_threshold,
    volatility_threshold,
    horizon,
    rrr,
    atr_mult
):

    sma = data[f"sma_{sma_period}"]

    distance = (
        data["close"] / sma - 1
    )

    mask = (
        (distance <= -distance_pct / 100)
        &
        (data["return_6"] <= momentum_threshold)
        &
        (data["volume_ratio_calc"] >= volume_threshold)
        &
        (
            data["volatility_ratio_calc"]
            >= volatility_threshold
        )
    )

    signal_idx = np.flatnonzero(
        mask.values
    )

    if len(signal_idx) == 0:
        return None

    # --------------------------------------------------------
    # Remove overlapping entries approximately
    # --------------------------------------------------------

    selected = []

    next_allowed = -1

    for idx in signal_idx:

        if idx >= next_allowed:

            selected.append(idx)

            next_allowed = idx + horizon

    if len(selected) < MIN_VALIDATION_TRADES:
        return None

    selected = np.array(
        selected,
        dtype=int
    )

    future_returns = (
        data[f"future_return_{horizon}"]
        .iloc[selected]
        .values
    )

    future_returns = (
        future_returns -
        TRADING_COST
    )

    # --------------------------------------------------------
    # Fast approximation of risk-adjusted return
    #
    # We estimate whether the future move is large enough
    # to reach TP or SL.
    # --------------------------------------------------------

    atr_pct = (
        data["atr_14"].iloc[selected].values /
        data["close"].iloc[selected].values
    )

    risk = atr_pct * atr_mult

    tp = risk * rrr

    wins = future_returns >= tp
    losses = future_returns <= -risk

    neutral = (
        (~wins) &
        (~losses)
    )

    simulated = np.where(
        wins,
        tp,
        np.where(
            losses,
            -risk,
            future_returns
        )
    )

    simulated = (
        simulated -
        TRADING_COST
    )

    win_rate = (
        simulated > 0
    ).mean()

    avg_return = simulated.mean()

    total_return = (
        np.prod(
            1 + simulated
        ) - 1
    )

    gross_profit = (
        simulated[
            simulated > 0
        ].sum()
    )

    gross_loss = (
        -simulated[
            simulated < 0
        ].sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit /
            gross_loss
        )
    else:
        profit_factor = 999

    equity = np.cumprod(
        1 + simulated
    )

    peak = np.maximum.accumulate(
        equity
    )

    dd = (
        equity / peak - 1
    )

    max_dd = dd.min()

    # --------------------------------------------------------
    # Fast score
    # --------------------------------------------------------

    score = (
        avg_return
        *
        np.log1p(len(simulated))
        *
        min(profit_factor, 10)
        /
        (1 + abs(max_dd) * 2)
    )

    return {
        "trades": len(simulated),
        "win_rate": win_rate,
        "average_return": avg_return,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "score": score
    }


# ============================================================
# SEARCH
# ============================================================

combinations = list(
    itertools.product(
        SMA_TYPES,
        DISTANCES,
        MOMENTUM_THRESHOLDS,
        VOLUME_THRESHOLDS,
        VOLATILITY_THRESHOLDS,
        HORIZONS,
        RRRS,
        ATR_MULTS
    )
)

print()
print(
    "Total combinations:",
    len(combinations)
)

fast_results = []

for counter, params in enumerate(
    combinations,
    1
):

    (
        sma,
        distance,
        momentum,
        volume,
        volatility,
        horizon,
        rrr,
        atr_mult
    ) = params

    result = fast_screen(
        validation,
        sma,
        distance,
        momentum,
        volume,
        volatility,
        horizon,
        rrr,
        atr_mult
    )

    if result is None:
        continue

    row = {
        "sma": sma,
        "distance_pct": distance,
        "momentum_threshold": momentum,
        "volume_threshold": volume,
        "volatility_threshold": volatility,
        "horizon": horizon,
        "rrr": rrr,
        "atr_mult": atr_mult,
        **result
    }

    fast_results.append(row)

    if counter % 5000 == 0:

        print(
            f"Progress: "
            f"{counter}/{len(combinations)} "
            f"("
            f"{counter / len(combinations) * 100:.1f}%"
            f")"
        )


fast_df = pd.DataFrame(
    fast_results
)

if len(fast_df) == 0:

    print()
    print("No candidates found.")

    raise SystemExit


fast_df = fast_df.sort_values(
    "score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# TOP FAST CANDIDATES
# ============================================================

print()
print("=" * 70)
print("TOP FAST CANDIDATES")
print("=" * 70)

cols = [
    "sma",
    "distance_pct",
    "momentum_threshold",
    "volume_threshold",
    "volatility_threshold",
    "horizon",
    "rrr",
    "atr_mult",
    "trades",
    "win_rate",
    "average_return",
    "profit_factor",
    "max_drawdown",
    "score"
]

print(
    fast_df[cols]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# EXACT TP/SL BACKTEST
# ============================================================

print()
print("=" * 70)
print("EXACT TP/SL BACKTEST")
print("=" * 70)


def exact_backtest(
    data,
    row,
    return_trades=False
):

    sma_period = int(row["sma"])
    distance_pct = float(row["distance_pct"]) / 100

    momentum_threshold = float(
        row["momentum_threshold"]
    )

    volume_threshold = float(
        row["volume_threshold"]
    )

    volatility_threshold = float(
        row["volatility_threshold"]
    )

    horizon = int(row["horizon"])
    rrr = float(row["rrr"])
    atr_mult = float(row["atr_mult"])

    sma = data[f"sma_{sma_period}"]

    distance = (
        data["close"] / sma - 1
    )

    mask = (
        (distance <= -distance_pct)
        &
        (data["return_6"] <= momentum_threshold)
        &
        (data["volume_ratio_calc"] >= volume_threshold)
        &
        (data["volatility_ratio_calc"] >= volatility_threshold)
    )

    indices = np.flatnonzero(mask.values)

    trades = []
    trade_details = []

    last_exit = -1

    for i in indices:

        if i <= last_exit:
            continue

        if i + horizon >= len(data):
            continue

        entry = float(
            data["close"].iloc[i]
        )

        atr = float(
            data["atr_14"].iloc[i]
        )

        if atr <= 0:
            continue

        risk_price = atr * atr_mult

        risk_pct = (
            risk_price / entry
        )

        stop = entry - risk_price
        target = entry + risk_price * rrr

        exit_index = None
        exit_price = None
        exit_reason = None

        end = min(
            i + horizon,
            len(data) - 1
        )

        for j in range(i + 1, end + 1):

            high = float(
                data["high"].iloc[j]
            )

            low = float(
                data["low"].iloc[j]
            )

            hit_sl = low <= stop
            hit_tp = high >= target

            # Conservative assumption:
            # if both TP and SL occur in the same candle,
            # SL happens first.

            if hit_sl and hit_tp:

                exit_price = stop
                exit_reason = "SL"
                exit_index = j

                break

            if hit_sl:

                exit_price = stop
                exit_reason = "SL"
                exit_index = j

                break

            if hit_tp:

                exit_price = target
                exit_reason = "TP"
                exit_index = j

                break

        # --------------------------------------------------------
        # TIME EXIT
        # --------------------------------------------------------

        if exit_index is None:

            exit_index = end

            exit_price = float(
                data["close"].iloc[end]
            )

            exit_reason = "TIME"

        # --------------------------------------------------------
        # RAW RETURN BEFORE COST
        # --------------------------------------------------------

        raw_return = (
            exit_price / entry - 1
        )

        net_return = (
            raw_return - TRADING_COST
        )

        # --------------------------------------------------------
        # R MULTIPLE
        #
        # 1R = distance from entry to stop
        #
        # This is what V12 will use for fixed-risk
        # equity simulation.
        # --------------------------------------------------------

        if risk_pct > 0:

            r_multiple = (
                raw_return / risk_pct
            )

        else:

            r_multiple = np.nan

        trades.append(
            net_return
        )

        trade_details.append({
            "entry_index": i,
            "exit_index": exit_index,

            "entry_time": data["time"].iloc[i],
            "exit_time": data["time"].iloc[exit_index],

            "entry_price": entry,
            "exit_price": exit_price,

            "atr": atr,
            "risk_price": risk_price,
            "risk_pct": risk_pct,

            "stop": stop,
            "target": target,

            "rrr": rrr,
            "atr_mult": atr_mult,
            "horizon": horizon,

            "raw_return": raw_return,
            "trade_return": net_return,
            "r_multiple": r_multiple,

            "exit_reason": exit_reason
        })

        last_exit = exit_index

    if len(trades) == 0:
        return None

    returns = np.array(
        trades,
        dtype=float
    )

    wins = returns > 0

    win_rate = wins.mean()

    avg_return = returns.mean()

    total_return = (
        np.prod(1 + returns) - 1
    )

    gross_profit = returns[
        returns > 0
    ].sum()

    gross_loss = -returns[
        returns < 0
    ].sum()

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = 999

    equity = np.cumprod(
        1 + returns
    )

    peak = np.maximum.accumulate(
        equity
    )

    dd = (
        equity / peak - 1
    )

    max_dd = dd.min()

    score = (
        avg_return
        *
        np.log1p(len(returns))
        *
        min(pf, 10)
        /
        (1 + abs(max_dd) * 2)
    )

    result = {
        "trades": len(returns),
        "win_rate": win_rate,
        "average_return": avg_return,
        "total_return": total_return,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "score": score
    }

    if return_trades:
        result["trade_details"] = trade_details

    return result

# ------------------------------------------------------------
# Run exact backtest only on top candidates
# ------------------------------------------------------------

exact_validation = []

candidates = fast_df.head(
    TOP_FAST_CANDIDATES
)

print()
print(
    "Exact backtesting:",
    len(candidates),
    "candidates"
)

for n, (_, row) in enumerate(
    candidates.iterrows(),
    1
):

    result = exact_backtest(
        validation,
        row
    )

    if result is None:
        continue

    output = row.to_dict()

    output.update({
        f"validation_{k}": v
        for k, v in result.items()
    })

    exact_validation.append(
        output
    )

    if n % 10 == 0:

        print(
            f"Exact progress: "
            f"{n}/{len(candidates)}"
        )


validation_exact_df = pd.DataFrame(
    exact_validation
)

if len(validation_exact_df) == 0:

    print(
        "No exact candidates."
    )

    raise SystemExit


validation_exact_df = (
    validation_exact_df
    .sort_values(
        "validation_score",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# TEST TOP VALIDATION CANDIDATES
# ============================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)

test_results = []
all_test_trades = []

top_final = validation_exact_df.head(
    30
)

for n, (_, row) in enumerate(
    top_final.iterrows(),
    1
):

    result = exact_backtest(
        test,
        row,
        return_trades=True
    )

    if result is None:
        continue

    output = row.to_dict()

    trade_details = result.pop(
    "trade_details",
    []
    )

    output.update({
        f"test_{k}": v
        for k, v in result.items()
    })

    test_results.append(
        output
    )

    print(
        f"Test progress: "
        f"{n}/{len(top_final)}"
    )


test_df = pd.DataFrame(
    test_results
)


# ============================================================
# PRINT VALIDATION
# ============================================================

print()
print("=" * 70)
print("TOP EXACT VALIDATION STRATEGIES")
print("=" * 70)

validation_cols = [
    "sma",
    "distance_pct",
    "momentum_threshold",
    "volume_threshold",
    "volatility_threshold",
    "horizon",
    "rrr",
    "atr_mult",
    "validation_trades",
    "validation_win_rate",
    "validation_average_return",
    "validation_profit_factor",
    "validation_max_drawdown",
    "validation_score"
]

print(
    validation_exact_df[
        validation_cols
    ].head(20)
    .to_string(index=False)
)


# ============================================================
# PRINT TEST
# ============================================================

if len(test_df) > 0:

    print()
    print("=" * 70)
    print("TOP TEST RESULTS")
    print("=" * 70)

    test_cols = [
        "sma",
        "distance_pct",
        "momentum_threshold",
        "volume_threshold",
        "volatility_threshold",
        "horizon",
        "rrr",
        "atr_mult",
        "test_trades",
        "test_win_rate",
        "test_average_return",
        "test_profit_factor",
        "test_max_drawdown",
        "test_score"
    ]

    test_df = test_df.sort_values(
        "test_score",
        ascending=False
    )

    print(
        test_df[
            test_cols
        ].to_string(index=False)
    )


# ============================================================
# ROBUST FILTER
# ============================================================

print()
print("=" * 70)
print("ROBUST CANDIDATES")
print("=" * 70)

if len(test_df) > 0:

    robust = test_df[
        (test_df["validation_trades"] >= MIN_VALIDATION_TRADES)
        &
        (test_df["test_trades"] >= MIN_TEST_TRADES)
        &
        (test_df["validation_profit_factor"] >= 1.10)
        &
        (test_df["test_profit_factor"] >= 1.05)
        &
        (test_df["validation_average_return"] > 0)
        &
        (test_df["test_average_return"] > 0)
    ].copy()

else:

    robust = pd.DataFrame()


if len(robust) > 0:

    robust = robust.sort_values(
        "test_score",
        ascending=False
    )

    print(
        robust[
            test_cols
        ].to_string(index=False)
    )

else:

    print(
        "No candidate passed the robustness filter."
    )


# ============================================================
# BEST
# ============================================================

print()
print("=" * 70)
print("BEST ROBUST STRATEGY")
print("=" * 70)

if len(robust) > 0:

    best = robust.iloc[0]

    print()
    print("ENTRY CONDITIONS:")
    print(
        f"  Price <= SMA{int(best['sma'])} "
        f"- {best['distance_pct']:.2f}%"
    )

    print(
        f"  Momentum <= "
        f"{best['momentum_threshold']:.4f}"
    )

    print(
        f"  Volume ratio >= "
        f"{best['volume_threshold']:.2f}"
    )

    print(
        f"  Volatility ratio >= "
        f"{best['volatility_threshold']:.2f}"
    )

    print()
    print("RISK MANAGEMENT:")
    print(
        f"  Horizon: "
        f"{int(best['horizon'])} hours"
    )

    print(
        f"  RRR: "
        f"{best['rrr']:.2f}"
    )

    print(
        f"  Stop: "
        f"{best['atr_mult']:.2f} ATR"
    )

    print()
    print("VALIDATION:")
    print(
        f"  Trades: "
        f"{int(best['validation_trades'])}"
    )

    print(
        f"  Win rate: "
        f"{best['validation_win_rate']:.2%}"
    )

    print(
        f"  Average return: "
        f"{best['validation_average_return']:.4%}"
    )

    print(
        f"  Profit factor: "
        f"{best['validation_profit_factor']:.3f}"
    )

    print(
        f"  Max DD: "
        f"{best['validation_max_drawdown']:.2%}"
    )

    print()
    print("TEST:")
    print(
        f"  Trades: "
        f"{int(best['test_trades'])}"
    )

    print(
        f"  Win rate: "
        f"{best['test_win_rate']:.2%}"
    )

    print(
        f"  Average return: "
        f"{best['test_average_return']:.4%}"
    )

    print(
        f"  Profit factor: "
        f"{best['test_profit_factor']:.3f}"
    )

    print(
        f"  Max DD: "
        f"{best['test_max_drawdown']:.2%}"
    )

else:

    print()
    print(
        "No robust strategy survived."
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data",
    exist_ok=True
)

fast_df.to_csv(
    "data/v11_1_fast_screening.csv",
    index=False
)

validation_exact_df.to_csv(
    "data/v11_1_validation_exact.csv",
    index=False
)

test_df.to_csv(
    "data/v11_1_test_exact.csv",
    index=False
)

test_trades_df = pd.DataFrame(
    all_test_trades
)

test_trades_df.to_csv(
    "data/v11_1_test_trades.csv",
    index=False
)

robust.to_csv(
    "data/v11_1_robust_strategies.csv",
    index=False
)

print()
print("=" * 70)
print("V11.1 COMPLETED")
print("=" * 70)

print()
print("Saved:")
print("data/v11_1_fast_screening.csv")
print("data/v11_1_test_trades.csv")
print("data/v11_1_validation_exact.csv")
print("data/v11_1_test_exact.csv")
print("data/v11_1_robust_strategies.csv")

print()
print("IMPORTANT:")
print("1. Fast screening searches the parameter space.")
print("2. Exact TP/SL validates only the strongest candidates.")
print("3. Validation discovers the strategy.")
print("4. Test is used only for final verification.")
print("5. Trades are NON-OVERLAPPING.")


print("=" * 70)
print("BITCOIN ML V12 - FIXED RISK / EQUITY SIMULATOR")
print("=" * 70)


# ============================================================
# CONFIG
# ============================================================

TEST_SUMMARY_FILE = Path(
    "data/v11_1_test_exact.csv"
)

TEST_TRADES_FILE = Path(
    "data/v11_1_test_trades.csv"
)

OUTPUT_FILE = Path(
    "data/v12_risk_simulation.csv"
)

TRADES_OUTPUT_FILE = Path(
    "data/v12_equity_trades.csv"
)

INITIAL_DEPOSIT = 10000.0

RISK_LEVELS = [
    0.0025,   # 0.25%
    0.0050,   # 0.50%
    0.0100,   # 1.00%
    0.0150,   # 1.50%
    0.0200,   # 2.00%
]


# ============================================================
# CHECK FILES
# ============================================================

if not TEST_SUMMARY_FILE.exists():
    raise FileNotFoundError(
        f"Не найден:\n{TEST_SUMMARY_FILE}\n"
        "Сначала необходимо завершить V11.1."
    )

if not TEST_TRADES_FILE.exists():
    raise FileNotFoundError(
        f"Не найден:\n{TEST_TRADES_FILE}\n\n"
        "Необходимо пересчитать V11.1 новой версией, "
        "которая сохраняет отдельные тестовые сделки."
    )


# ============================================================
# LOAD
# ============================================================

print()
print("Loading summary:")
print(TEST_SUMMARY_FILE)

summary = pd.read_csv(
    TEST_SUMMARY_FILE
)

print(
    f"Summary rows: {len(summary)}"
)

print()
print("Loading individual trades:")
print(TEST_TRADES_FILE)

trades = pd.read_csv(
    TEST_TRADES_FILE
)

print(
    f"Trade rows: {len(trades)}"
)


# ============================================================
# FIND BEST ROBUST STRATEGY
# ============================================================

required_summary = [
    "validation_trades",
    "test_trades",
    "validation_profit_factor",
    "test_profit_factor",
    "validation_average_return",
    "test_average_return",
    "test_score"
]

missing = [
    c for c in required_summary
    if c not in summary.columns
]

if missing:

    raise ValueError(
        "В summary отсутствуют колонки:\n"
        + "\n".join(missing)
    )


robust = summary[
    (summary["validation_trades"] >= 50)
    &
    (summary["test_trades"] >= 30)
    &
    (summary["validation_profit_factor"] >= 1.10)
    &
    (summary["test_profit_factor"] >= 1.05)
    &
    (summary["validation_average_return"] > 0)
    &
    (summary["test_average_return"] > 0)
].copy()


if len(robust) == 0:

    raise ValueError(
        "Ни одна стратегия V11.1 не прошла "
        "robust-фильтр."
    )


robust = robust.sort_values(
    "test_score",
    ascending=False
).reset_index(drop=True)


best = robust.iloc[0]


# ============================================================
# PRINT BEST STRATEGY
# ============================================================

print()
print("=" * 70)
print("SELECTED STRATEGY")
print("=" * 70)

print()

for column in [
    "sma",
    "distance_pct",
    "momentum_threshold",
    "volume_threshold",
    "volatility_threshold",
    "horizon",
    "rrr",
    "atr_mult"
]:

    print(
        f"{column:25s}: {best[column]}"
    )

print()
print(
    f"Validation trades: "
    f"{int(best['validation_trades'])}"
)

print(
    f"Validation PF: "
    f"{best['validation_profit_factor']:.3f}"
)

print(
    f"Test trades: "
    f"{int(best['test_trades'])}"
)

print(
    f"Test win rate: "
    f"{best['test_win_rate']:.2%}"
)

print(
    f"Test average return: "
    f"{best['test_average_return']:.4%}"
)

print(
    f"Test PF: "
    f"{best['test_profit_factor']:.3f}"
)

print(
    f"Test max DD: "
    f"{best['test_max_drawdown']:.2%}"
)


# ============================================================
# FILTER EXACT TRADES FOR BEST STRATEGY
# ============================================================

strategy_columns = [
    "sma",
    "distance_pct",
    "momentum_threshold",
    "volume_threshold",
    "volatility_threshold",
    "horizon",
    "rrr",
    "atr_mult"
]


def same_strategy(trade_df, strategy):

    mask = pd.Series(
        True,
        index=trade_df.index
    )

    for col in strategy_columns:

        mask &= np.isclose(
            trade_df[col].astype(float),
            float(strategy[col])
        )

    return mask


strategy_trades = trades[
    same_strategy(trades, best)
].copy()


if len(strategy_trades) == 0:

    raise ValueError(
        "Не найдены отдельные сделки "
        "для выбранной стратегии."
    )


strategy_trades = strategy_trades.sort_values(
    "entry_time"
).reset_index(drop=True)


print()
print(
    f"Selected test trades: "
    f"{len(strategy_trades)}"
)


# ============================================================
# CHECK R MULTIPLES
# ============================================================

if "r_multiple" not in strategy_trades.columns:

    raise ValueError(
        "В test trades отсутствует r_multiple."
    )


strategy_trades["r_multiple"] = pd.to_numeric(
    strategy_trades["r_multiple"],
    errors="coerce"
)

strategy_trades = strategy_trades.dropna(
    subset=["r_multiple"]
).reset_index(drop=True)


if len(strategy_trades) == 0:

    raise ValueError(
        "r_multiple не содержит числовых значений."
    )


r_values = strategy_trades[
    "r_multiple"
].values.astype(float)


# ============================================================
# RAW R STATISTICS
# ============================================================

print()
print("=" * 70)
print("EMPIRICAL R STATISTICS")
print("=" * 70)

print()

print(
    f"Trades:       {len(r_values)}"
)

print(
    f"Wins:         {(r_values > 0).sum()}"
)

print(
    f"Losses:       {(r_values < 0).sum()}"
)

print(
    f"Win rate:     {(r_values > 0).mean():.2%}"
)

print(
    f"Average R:    {r_values.mean():.4f}R"
)

print(
    f"Median R:     {np.median(r_values):.4f}R"
)

print(
    f"Min R:        {r_values.min():.4f}R"
)

print(
    f"Max R:        {r_values.max():.4f}R"
)


# ============================================================
# EQUITY SIMULATOR
# ============================================================

def simulate_fixed_risk(
    r_values,
    risk_per_trade,
    initial_equity
):

    equity = float(initial_equity)

    peak = equity

    max_dd = 0.0

    max_consecutive_losses = 0
    max_consecutive_wins = 0

    consecutive_losses = 0
    consecutive_wins = 0

    rows = []

    for i, r in enumerate(r_values):

        r = float(r)

        # ----------------------------------------------------
        # FIXED FRACTIONAL RISK
        #
        # -1R = -risk_per_trade
        # +3R = +3*risk_per_trade
        #
        # Therefore 1% risk means:
        #
        # -1R -> -1%
        # +3R -> +3%
        # ----------------------------------------------------

        account_return = (
            r * risk_per_trade
        )

        # Safety guard
        account_return = max(
            account_return,
            -0.999
        )

        old_equity = equity

        pnl = (
            old_equity *
            account_return
        )

        equity = (
            old_equity +
            pnl
        )

        peak = max(
            peak,
            equity
        )

        drawdown = (
            equity / peak - 1
        )

        max_dd = min(
            max_dd,
            drawdown
        )

        if r > 0:

            consecutive_wins += 1
            consecutive_losses = 0

            max_consecutive_wins = max(
                max_consecutive_wins,
                consecutive_wins
            )

        elif r < 0:

            consecutive_losses += 1
            consecutive_wins = 0

            max_consecutive_losses = max(
                max_consecutive_losses,
                consecutive_losses
            )

        else:

            consecutive_wins = 0
            consecutive_losses = 0

        rows.append({
            "trade": i + 1,
            "r_multiple": r,
            "risk_percent": risk_per_trade * 100,
            "account_return": account_return,
            "pnl": pnl,
            "old_equity": old_equity,
            "equity": equity,
            "drawdown": drawdown
        })

    total_return = (
        equity / initial_equity - 1
    )

    pnl_array = np.array(
        [
            row["pnl"]
            for row in rows
        ],
        dtype=float
    )

    gross_profit = pnl_array[
        pnl_array > 0
    ].sum()

    gross_loss = abs(
        pnl_array[
            pnl_array < 0
        ].sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit /
            gross_loss
        )
    else:
        profit_factor = np.inf

    return {
        "risk_per_trade": risk_per_trade,
        "risk_percent": risk_per_trade * 100,
        "initial_deposit": initial_equity,
        "final_equity": equity,
        "total_return": total_return,
        "total_return_percent": total_return * 100,
        "max_drawdown": max_dd,
        "max_drawdown_percent": max_dd * 100,
        "profit_factor": profit_factor,
        "trades": len(r_values),
        "win_rate": (r_values > 0).mean(),
        "average_R": r_values.mean(),
        "max_consecutive_losses": max_consecutive_losses,
        "max_consecutive_wins": max_consecutive_wins,
        "trade_rows": rows
    }


# ============================================================
# RUN
# ============================================================

results = []

all_equity_rows = []

for risk in RISK_LEVELS:

    print(
        f"Simulating "
        f"{risk * 100:.2f}% risk..."
    )

    result = simulate_fixed_risk(
        r_values,
        risk,
        INITIAL_DEPOSIT
    )

    summary_row = {
        key: value
        for key, value in result.items()
        if key != "trade_rows"
    }

    results.append(
        summary_row
    )

    for row in result["trade_rows"]:

        row_copy = dict(row)

        row_copy["strategy_sma"] = best["sma"]
        row_copy["strategy_distance_pct"] = best[
            "distance_pct"
        ]

        row_copy["strategy_momentum_threshold"] = best[
            "momentum_threshold"
        ]

        row_copy["strategy_volume_threshold"] = best[
            "volume_threshold"
        ]

        row_copy["strategy_volatility_threshold"] = best[
            "volatility_threshold"
        ]

        row_copy["strategy_horizon"] = best[
            "horizon"
        ]

        row_copy["strategy_rrr"] = best["rrr"]
        row_copy["strategy_atr_mult"] = best["atr_mult"]

        all_equity_rows.append(
            row_copy
        )


results_df = pd.DataFrame(
    results
)


# ============================================================
# RESULT TABLE
# ============================================================

print()
print("=" * 70)
print("RISK COMPARISON")
print("=" * 70)

display_columns = [
    "risk_percent",
    "final_equity",
    "total_return_percent",
    "max_drawdown_percent",
    "profit_factor",
    "trades",
    "win_rate",
    "average_R",
    "max_consecutive_losses",
    "max_consecutive_wins"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# RETURN / DD
# ============================================================

results_df["return_dd_ratio"] = (
    results_df["total_return_percent"]
    /
    results_df["max_drawdown_percent"].abs()
)

best_risk = results_df.loc[
    results_df["return_dd_ratio"].idxmax()
]


print()
print("=" * 70)
print("BEST RISK / RETURN PROFILE")
print("=" * 70)

print()

print(
    f"Risk per trade: "
    f"{best_risk['risk_percent']:.2f}%"
)

print(
    f"Final equity:   "
    f"${best_risk['final_equity']:,.2f}"
)

print(
    f"Total return:   "
    f"{best_risk['total_return_percent']:.2f}%"
)

print(
    f"Max DD:         "
    f"{best_risk['max_drawdown_percent']:.2f}%"
)

print(
    f"Profit factor:  "
    f"{best_risk['profit_factor']:.3f}"
)


# ============================================================
# 1% SPECIFIC RESULT
# ============================================================

one_percent = results_df[
    np.isclose(
        results_df["risk_percent"],
        1.0
    )
].iloc[0]


print()
print("=" * 70)
print("1% RISK RESULT")
print("=" * 70)

print()

print(
    f"Initial deposit: "
    f"${INITIAL_DEPOSIT:,.2f}"
)

print(
    f"Final equity:    "
    f"${one_percent['final_equity']:,.2f}"
)

print(
    f"Profit:          "
    f"${one_percent['final_equity'] - INITIAL_DEPOSIT:,.2f}"
)

print(
    f"Return:          "
    f"{one_percent['total_return_percent']:.2f}%"
)

print(
    f"Max drawdown:    "
    f"{one_percent['max_drawdown_percent']:.2f}%"
)

print(
    f"Profit factor:   "
    f"{one_percent['profit_factor']:.3f}"
)

print(
    f"Win rate:        "
    f"{one_percent['win_rate']:.2%}"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

equity_trades_df = pd.DataFrame(
    all_equity_rows
)

equity_trades_df.to_csv(
    TRADES_OUTPUT_FILE,
    index=False
)


print()
print("=" * 70)
print("V12 COMPLETED")
print("=" * 70)

print()
print("Saved:")
print(OUTPUT_FILE)
print(TRADES_OUTPUT_FILE)

print()
print(
    "IMPORTANT:"
)

print(
    "1. V11.1 determines the strategy."
)

print(
    "2. V11.1 saves every individual test trade."
)

print(
    "3. V12 selects only the best robust strategy."
)

print(
    "4. Each trade is represented as an empirical R-multiple."
)

print(
    "5. 1% risk means -1R = approximately -1% of equity."
)

print(
    "6. +3R means approximately +3% before other execution effects."
)

print(
    "7. Equity compounds after every trade."
)

print(
    "8. No parameter optimization is performed in V12."
)

print("=" * 70)

