# Industry Multi-Factor Rotation Model

**Stage: Problem Framing & Scoping (Stage 01)**

---

## Problem Statement

In the stock market, sector rotation is a significant phenomenon. Traditional models such as Mean-Variance Optimization (MVO) and Hidden Markov Models (HMM) show limitations in sector allocation:

MVO is highly sensitive to parameter estimation errors and often leads to excessive turnover.
HMM relies heavily on regime switching but struggles to fully capture sector-specific dynamics.

This project develops an Industry Multi-Factor Rotation Model that integrates valuation, growth, quality, technical, analyst expectation, and high-frequency factors. By dynamically scoring and allocating weights across sectors, the model aims to enhance risk-adjusted returns and mitigate vulnerabilities in extreme market conditions.

---

## Stakeholder & User

* **Who decides?**
  Research teams and portfolio managers decide on factor selection, weighting methodology, and strategy execution.
* **Who uses the output?**
  Portfolio managers and traders use the sector rotation signals to adjust sector allocations and execute rebalancing.
* **Timing:**
  Factor data and sector scores are updated daily, with computations completed before rebalancing dates.

---

## Useful Answer & Decision

* **Predictive Metric:**
  Composite factor score and recommended portfolio weight for each sector.

---

## Assumptions & Constraints

* Factor data is sourced via APIs (e.g., Ricequant), covering financials, valuations, momentum, analyst expectations, and high-frequency trading data.
* Data undergoes cleaning, winsorization, standardization, and market-cap neutrality.
* Constraints: fully invested portfolio (weights sum to 1), no short-selling, turnover control to reduce transaction costs.

---

## Known Unknowns / Risks

* **Factor decay:** Factor performance may weaken across different market regimes. 
* **Overfitting:** Excessive reliance on historical data may reduce out-of-sample performance.
* **Extreme events:** Black swan events may overwhelm model predictions. 

---

## Lifecycle Mapping

1. **Define problem & scope** → Problem Framing & Scoping (Stage 01) → README.md
2. **Data collection & preprocessing** → Data Collection & Preprocessing (Stage 02) → Sector factor library + data dictionary
3. **Factor testing & synthesis** → Feature Engineering & Factor Testing (Stage 03) → Single-factor IC tests + composite factor construction
4. **Model development** → Modeling (Stage 04) → Sector scoring & weighting model
5. **Backtesting & validation** → Model Validation (Stage 05) → Performance metrics (returns, Sharpe, drawdown)
6. **Deployment** → Deployment (Stage 06) → Monthly sector allocation weights (CSV/trading orders)
7. **Monitoring & optimization** → Monitoring & Maintenance (Stage 07) → Factor validity tracking + dynamic reweighting

---

## Repo Plan

```
/data/            # Raw and processed factor and sector data  
/src/             # Scripts for factor computation, scoring, backtesting  
/notebooks/       # Jupyter notebooks for factor analysis & backtest exploration  
/docs/            # Research reports, README, stakeholder memos  
```

---