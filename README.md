Historical BTC/USD data
        ↓
Feature engineering
        ↓
V11.1 Fast Parameter Screening
        ↓
Exact TP/SL Validation
        ↓
Out-of-Sample Test (2025+)
        ↓
Robustness Filter
        ↓
V12 Fixed-Risk Equity Simulation

The repository contains the dataset required to reproduce the main experiment. Generated backtest results are intentionally excluded from version control and can be recreated by running the main script.

# Bitcoin ML Strategy Optimizer

A research project for developing and evaluating a Bitcoin trading strategy using historical BTC/USD data, parameter optimization, walk-forward-style validation, exact TP/SL backtesting, and fixed-risk equity simulation.

## Project Overview

The project searches for robust trading parameters using historical Bitcoin price data and evaluates whether the discovered strategy remains profitable on previously unseen data.

The workflow is divided into two main stages:

1. **V11.1 — Strategy Discovery and Validation**
2. **V12 — Fixed-Risk Equity Simulation**

The main implementation is contained in:

`strategy_v10_optimizer.py`

## Dataset

The strategy uses the following dataset:

`data/btcusd_ml_v2.csv`

The dataset contains historical BTC/USD market data including:

* Open price
* High price
* Low price
* Close price
* Tick volume / real volume
* Timestamp

The data is sorted chronologically before processing.

## Data Split

The historical data is divided chronologically to avoid using future information during strategy development.

| Period      | Purpose                            |
| ----------- | ---------------------------------- |
| Before 2023 | Training / strategy development    |
| 2023–2024   | Validation and parameter selection |
| 2025 onward | Final out-of-sample test           |

The test period is not used for parameter optimization.

## Feature Engineering

The strategy calculates several technical features directly from the historical data.

### Returns

* 1-period return
* 3-period return
* 6-period return
* 12-period return
* 24-period return

### Moving Averages

The following Simple Moving Averages are tested:

* SMA 20
* SMA 50
* SMA 100
* SMA 200

### Volume

A volume ratio is calculated using the current volume relative to its 20-period moving average.

### Volatility

The model calculates:

* 5-period return volatility
* 20-period return volatility
* volatility ratio

### ATR

A 14-period Average True Range is calculated and used for dynamic stop-loss placement.

## V11.1 — Strategy Discovery

The first stage searches a large parameter space for potentially profitable trading configurations.

The following parameters are optimized:

* SMA period
* Distance from SMA
* Momentum threshold
* Volume threshold
* Volatility threshold
* Trading horizon
* Risk/reward ratio
* ATR stop multiplier

The strategy looks for situations where Bitcoin price is significantly below a selected moving average while momentum, volume, and volatility satisfy the specified conditions.

## Fast Screening

Because the full parameter space contains a large number of combinations, the first stage uses a fast vectorized screening process.

For each candidate strategy the following metrics are estimated:

* Number of trades
* Win rate
* Average return
* Total return
* Profit factor
* Maximum drawdown
* Composite strategy score

Only the strongest candidates are passed to the more expensive exact backtest.

## Exact TP/SL Backtest

The strongest candidates from the fast screening stage are tested using candle-level TP/SL logic.

For every trade:

* Entry occurs when the strategy conditions are satisfied.
* Stop-loss is based on ATR.
* Take-profit is calculated using the selected risk/reward ratio.
* A maximum holding horizon is applied.
* Trades are non-overlapping.
* Trading costs are included.

If both take-profit and stop-loss are reached within the same candle, a conservative assumption is used: **the stop-loss is considered to have occurred first**.

## Validation

The exact backtest is first performed on the validation period.

The best candidates are ranked using the validation score.

Only the strongest validation candidates are then evaluated on the final test period.

## Final Test

The test period represents previously unseen market data.

The test is used only for final verification of the strategy's robustness.

The strategy is not optimized using test-period results.

## Robustness Filter

A strategy must satisfy minimum conditions to be considered robust.

The current filters require:

* At least 50 validation trades
* At least 30 test trades
* Validation profit factor ≥ 1.10
* Test profit factor ≥ 1.05
* Positive average return on validation
* Positive average return on test

Only strategies satisfying these conditions are considered robust candidates.

## V12 — Fixed-Risk Equity Simulation

After V11.1 identifies the best robust strategy, V12 performs a separate equity simulation.

The purpose of V12 is not to optimize the strategy.

Instead, it evaluates how the selected strategy would affect an account when each trade risks a fixed percentage of current equity.

The simulation starts with:

**Initial deposit: $10,000**

The following risk levels are tested:

* 0.25%
* 0.50%
* 1.00%
* 1.50%
* 2.00%

## R-Multiple Model

Each individual test trade is converted into an empirical R-multiple.

The concept is:

* `-1R` ≈ one unit of predefined risk
* `+1R` ≈ one unit of predefined reward
* `+3R` ≈ three units of predefined risk

For example, with 1% risk per trade:

* `-1R` corresponds approximately to -1% of equity
* `+3R` corresponds approximately to +3% of equity

Equity is compounded after every trade.

## Output Files

The script generates several CSV files inside the `data/` directory:

### V11.1

`v11_1_fast_screening.csv`

Results of the initial parameter search.

`v11_1_validation_exact.csv`

Exact backtest results for the strongest validation candidates.

`v11_1_test_exact.csv`

Final test results.

`v11_1_robust_strategies.csv`

Strategies that passed the robustness filter.

`v11_1_test_trades.csv`

Individual test trades used for the equity simulation.

### V12

`v12_risk_simulation.csv`

Comparison of different fixed-risk levels.

`v12_equity_trades.csv`

Trade-by-trade equity simulation.

## Running the Project

Install the required Python packages:

```bash
pip install numpy pandas
```

Then run:

```bash
python strategy_v10_optimizer.py
```

The script first performs the V11.1 strategy discovery and testing process and then performs the V12 fixed-risk equity simulation.

## Methodology

The project is designed around several principles:

* Chronological data splitting
* No random train/test shuffling
* Out-of-sample testing
* Parameter search on validation data
* Exact TP/SL verification
* Non-overlapping trades
* Conservative same-candle TP/SL handling
* Trading cost adjustment
* Robustness filtering
* Separate fixed-risk equity simulation

## Important Limitations

This project is a research and backtesting system.

Historical performance does not guarantee future results.

The backtest does not fully reproduce all real-world trading conditions, such as:

* Slippage
* Market impact
* Liquidity limitations
* Exchange-specific execution
* Funding costs
* Latency
* Spread changes
* Unexpected market events

The results should therefore be interpreted as a quantitative research experiment rather than a guarantee of future trading performance.

## Project Status

Current version:

**V11.1 Strategy Optimizer + V12 Fixed-Risk Equity Simulator**

The project is focused on testing whether a parameterized Bitcoin trading strategy can demonstrate robustness across different historical market periods and risk assumptions.
