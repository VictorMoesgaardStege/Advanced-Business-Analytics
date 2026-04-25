# Hourly Shift Model Blueprint

## Purpose

This note translates the current project setup into a concrete next-step model for simulating household electricity shifting with a `120` hour price forecast.

This blueprint is intentionally anchored on the **XGBoost full-resolution price forecast only**. It does **not** build on the current demand-shift simulation modules. The forecasting input should come from `XG_Boost_full_Res.py` and the outputs it creates.

The project already has:

- hourly DK1 price and consumption data
- an XGBoost forecasting model for hourly spot prices over the next `120` hours
- historical load and supply data that can be used for impact evaluation

The main modelling gap is that the household shift logic should use the forecast at **hourly resolution** over the next `120` hours instead of comparing daily averages. This matters because most grid stress, cheap-price opportunities, renewable surplus hours, and rebound effects happen within the day.

## Why Move From Daily To Hourly

The daily model is useful as a first proof of concept, but it hides the most important operational patterns:

- evening stress happens over a few peak hours, not the whole day
- low-price and negative-price windows often occur for a few hours
- wind and solar surplus should attract demand in specific hours
- shifting can create a new secondary peak if too much load moves into the same hour
- thermal loads and EVs have deadlines and rebound, which daily averages do not capture well

For this project, the recommended next model is therefore a **rolling hourly shift model** with a `120` hour look-ahead.

## Recommended Modelling Philosophy

Use a hybrid approach:

- behavioral enough to remain realistic for households
- optimization-based enough to avoid obviously inefficient shifting
- uncertainty-aware enough to respect forecast error
- aggregated enough to fit the project scope and available data

The best fit is a **rolling-horizon, segmented, scenario-based model**.

The important restriction for the next prototype is:

- use `XG_Boost_full_Res.py` as the electricity price forecast engine
- do not reuse the current daily or hourly shift simulation logic as a modelling base
- build the shift model as a new standalone module

## Core Structure

At each decision time `t`:

1. observe current hour, current load, and the newest `120` hour forecast
2. split the household load into behavioral segments
3. optimize or simulate how flexible load is moved within the next `120` hours
4. implement only the decision for the current hour
5. move one hour forward and re-run with updated forecasts

This is equivalent to a simple model predictive control setup.

## Suggested Household Segments

Do not treat all households as equally flexible. A segmented model is much closer to the literature and to practical energy behavior.

### Segment A: Inflexible Base Load

Examples:

- lighting
- cooking
- electronics
- always-on appliances
- occupancy-driven demand

Characteristics:

- cannot be shifted in a meaningful way
- remains in the original hour
- acts as the baseline load floor

### Segment B: Deferrable Wet Loads

Examples:

- washing machine
- tumble dryer
- dishwasher

Characteristics:

- can typically be delayed by `2` to `24` hours
- total energy must still be delivered
- user inconvenience rises with waiting time
- good candidate for price-based scheduling

### Segment C: Thermal Loads

Examples:

- heat pumps
- electric space heating
- domestic hot water

Characteristics:

- can be shifted by pre-heating or pre-heating water
- depends on thermal storage and comfort constraints
- creates rebound if deferred too long
- should be controlled with comfort bounds, not pure price response

### Segment D: EV Charging

Examples:

- home charging overnight
- next-day commuting charge need

Characteristics:

- often the largest household flexibility source
- energy must be delivered before departure deadline
- highly shiftable within deadline windows
- should be rewarded for moving to low-price and high-surplus hours

## Starting Assumptions For The Project

These are not universal truths. They are good starting values for scenario analysis.

### System Share

- household share of system electricity: `0.32` to `0.35`
- responsive household share: `0.25` to `0.40`

These are literature-based starting ranges for scenario analysis and should be treated as explicit assumptions.

### Segment Shares Of Household Load

Use scenario ranges instead of one fixed value:

| Segment | Conservative | Central | Flexible Future |
|---|---:|---:|---:|
| Inflexible base | 0.70 | 0.60 | 0.45 |
| Wet loads | 0.08 | 0.10 | 0.10 |
| Thermal loads | 0.12 | 0.18 | 0.25 |
| EV charging | 0.02 | 0.07 | 0.15 |
| Other flexible | 0.08 | 0.05 | 0.05 |

These shares should sum to `1.00` within the household portion.

### Maximum Shift Windows

| Segment | Typical Max Wait |
|---|---:|
| Wet loads | `6` to `24` hours |
| Thermal loads | `1` to `12` hours |
| EV charging | `4` to `24` hours, sometimes longer |

### Maximum Shiftable Fraction

| Segment | Share That Can Move |
|---|---:|
| Wet loads | `0.40` to `0.80` |
| Thermal loads | `0.10` to `0.40` |
| EV charging | `0.60` to `0.95` |

The model should apply these as hard upper bounds, not guarantees.

## Data Inputs

The next model should be built directly on top of the repository data, but without depending on the existing shift simulators.

### Already Available In The Repo

- `XG_Boost_full_Res.py`
- `data/day_ahead_prices_dk1_raw.csv`
- `data/consumption_dk1_raw.csv`
- `data/supply_forecasts_dk1_raw.csv`
- `outputs/model/predictions.parquet`

### Price Forecast Source

The price forecast should come from `XG_Boost_full_Res.py`, which already defines:

- `5` XGBoost day models
- day groups `1-24`, `25-48`, `49-72`, `73-96`, `97-120`
- hourly target times over a full `120` hour horizon
- walk-forward validation output
- saved predictions in `outputs/model/predictions.parquet`

This means the shift model does not need to forecast prices itself. It should consume the XGBoost forecast as an external input layer.

### Forecast Inputs To Use

For each future hour `h` in `1..120`, store:

- XGBoost point forecast for price
- issue time
- target time
- forecast horizon `h`
- supply forecast, preferably wind and solar related supply if available
- optional probability of stress or extreme price

### Important Forecast Limitation

`XG_Boost_full_Res.py` is a **point-forecast** model, not a quantile model. Therefore the blueprint should not assume forecast quantiles are directly available. If uncertainty is needed, it should be added from the model's own walk-forward residuals or horizon-specific forecast errors, not from a separate simulation framework.

## Decision Variables

Model at the system-aggregated household level rather than individual household level.

For each hour `t` and segment `s`:

- `baseline_load[s,t]`: baseline household segment demand
- `served_load[s,t]`: final demand served after shifting
- `shift_out[s,t]`: energy moved away from hour `t`
- `shift_in[s,t,u]`: energy from origin hour `t` served in future hour `u`

For deadline-based loads such as EVs, it is often easier to define:

- `x[s,t,u]`: energy originally associated with hour `t` that is delivered at hour `u`

with `u >= t`.

## Core Constraints

### Energy Conservation

Flexible demand is shifted, not destroyed:

```text
served_load[s,t] = baseline_load[s,t] - shift_out[s,t] + sum_of_shifted_in_to_t
```

For purely deferrable loads:

```text
sum_u x[s,t,u] = flexible_energy_available[s,t]
```

### Shiftability Limits

```text
shift_out[s,t] <= flexible_share[s,t] * baseline_load[s,t]
```

### Deadline / Waiting Constraints

```text
x[s,t,u] = 0  for  u > t + max_wait[s]
```

For EV charging:

```text
sum_{u in charging_window} x[EV,t,u] >= required_charge_by_deadline
```

### Comfort Bounds For Thermal Loads

Use a simplified thermal state:

```text
thermal_state[t+1] = a * thermal_state[t] + b * heating_input[t] + c * outdoor_temp[t]
```

with:

```text
thermal_min <= thermal_state[t] <= thermal_max
```

If you want a lighter version for the hand-in, replace thermal state with a soft penalty on delaying thermal energy too long.

### Secondary Peak Protection

Prevent the simulator from creating a new artificial peak:

```text
system_shifted_load[t] <= peak_cap[t]
```

or penalize load above a stress threshold.

## Objective Function

The objective should combine household value and system value.

### Recommended Objective

```text
min
Expected_Energy_Cost
+ lambda_wait * Waiting_Disutility
+ lambda_comfort * Comfort_Penalty
+ lambda_rebound * Rebound_Penalty
+ lambda_peak * Grid_Stress_Penalty
- lambda_surplus * Renewable_Surplus_Alignment
+ rho * Tail_Risk_Penalty
```

### Interpretation Of Each Term

- `Expected_Energy_Cost`: bill paid under forecasted or scenario prices
- `Waiting_Disutility`: inconvenience from delaying activities
- `Comfort_Penalty`: thermal discomfort or missed energy service
- `Rebound_Penalty`: discourages moving too much load into one later hour
- `Grid_Stress_Penalty`: discourages demand during already stressed hours
- `Renewable_Surplus_Alignment`: rewards demand in high-supply or very low-price hours
- `Tail_Risk_Penalty`: protects against bad outcomes under forecast error

## Useful Functional Forms

### Waiting Utility

An exponential form is simple and intuitive:

```text
wait_penalty = alpha_s * (exp(beta_s * wait_hours) - 1)
```

Use low `beta` for EVs and higher `beta` for wet loads.

### Price Attractiveness

For behavioral realism, convert price spread to a bounded response:

```text
response = 1 / (1 + exp(-k_s * (spread - m_s)))
```

This should be implemented directly in the new shift model and calibrated against the XGBoost price forecast outputs.

### Grid Stress Penalty

Use a convex penalty once shifted load exceeds the baseline stress threshold:

```text
stress_penalty[t] = gamma * max(0, shifted_system_load[t] - stress_threshold)^2
```

The threshold can be estimated from historical DK1 load distributions or from a separate baseline stress analysis, but the shift logic itself should remain independent from the current simulation modules.

### Surplus Alignment Reward

Define surplus hours using supply forecast minus baseline demand:

```text
surplus[t] = max(0, supply_forecast[t] - baseline_system_load[t])
```

Then reward shifted consumption that lands in those hours:

```text
surplus_reward[t] = eta * min(shifted_in[t], surplus[t])
```

This is the cleanest way to capture the supplier perspective without building a full production dispatch model.

## Handling Forecast Uncertainty

The `120` hour forecast should not be treated as certain.

### Minimum Version

Use the XGBoost point forecast as the optimization input and evaluate robustness using horizon-specific residual bands derived from walk-forward validation.

For example:

- central path: XGBoost point forecast
- optimistic path: point forecast minus expected positive error band
- pessimistic path: point forecast plus expected negative error band

Optimize on the central path, then test performance under all paths.

### Better Version

Sample hourly scenarios from the XGBoost walk-forward residual distribution by horizon:

- scenario `1..N` for price paths
- optional linked supply scenarios

Then optimize expected utility plus a downside-risk term.

### Recommended Risk Term

Use a simple CVaR-style penalty:

```text
tail_risk = average_cost_of_worst_10_percent_scenarios
```

This is useful because a shift plan can look good on the point forecast but fail badly if the cheap future hours do not materialize.

## Recommended Grid Metrics

### Core Grid KPIs

- peak load reduction in `MW`
- reduction in `p95` and `p99` load
- overloaded hours avoided
- critical hours avoided
- stress-hour energy reduced
- peak-to-average ratio
- load factor improvement
- average and maximum ramp reduction
- maintenance-pressure change

### Hourly Stress Metrics

For each hour:

- load ratio relative to the historical p99 threshold
- overload indicator
- ramp ratio
- maintenance pressure contribution

These can be calculated with the same logic already used in the repo.
These can be calculated directly from historical DK1 hourly load data and the shifted load profile generated by the new model.

## Recommended Supplier Metrics

These metrics connect the shift model to production efficiency and curtailment risk.

### Supplier / System Balance KPIs

- renewable surplus absorbed in `MWh`
- curtailed energy avoided in `MWh`
- energy moved into low-price hours
- energy moved out of stress-price hours
- peaker-energy avoided
- balancing-cost proxy avoided
- generator-ramp proxy reduced

### Simple Curtailment Proxy

If only aggregate supply forecast is available:

```text
curtailment_proxy[t] = max(0, supply_forecast[t] - shifted_system_load[t])
```

Then compare before and after shifting:

```text
curtailment_avoided = baseline_curtailment_proxy - shifted_curtailment_proxy
```

This is not full production modelling, but it is transparent and appropriate for the project scope.

## Recommended Calibration Strategy

Do not calibrate one single "average household". Calibrate ranges and test scenarios.

### Behavioral Calibration

Use:

- literature-informed flexible shares
- scenario analysis for responsiveness and wait tolerance
- transparent household participation assumptions

### Technical Calibration

Use:

- historical hourly consumption profile from DK1
- stress thresholds estimated from observed hourly load percentiles
- supply forecast to define surplus windows
- XGBoost walk-forward forecast errors to characterize price uncertainty

### Validation Questions

The simulator should pass these sanity checks:

- does it reduce load during forecasted expensive hours
- does it avoid creating a larger new peak later
- does savings increase when forecast spreads increase
- does thermal shifting produce rebound if comfort is binding
- does more EV share create more flexibility
- does forecast error reduce realized benefit

## Recommended Implementation Path In This Repo

### Step 1: Use The Forecasting Asset Only

Use:

- `XG_Boost_full_Res.py`
- `outputs/model/predictions.parquet`
- historical DK1 consumption and supply files

### Step 2: Create A New Module

Suggested path:

`src/data/hourly_shift_model.py`

This module should be independent from the existing simulation scripts.

### Step 3: Core Functions

Recommended functions:

```python
load_hourly_inputs(...)
load_xgboost_price_forecast(...)
build_household_segments(...)
generate_forecast_scenarios(...)
optimize_hourly_shift_plan(...)
apply_first_step(...)
simulate_rolling_horizon(...)
evaluate_grid_impact(...)
evaluate_supplier_impact(...)
```

### Step 4: Output Tables

Recommended outputs:

- hourly baseline vs shifted load
- shifted energy by segment
- consumer savings by segment
- grid stress metrics before vs after
- supplier surplus absorption before vs after
- sensitivity table for scenario assumptions

### Step 5: Visuals

Recommended figures:

- hourly baseline vs shifted load for one week
- heatmap of shifted energy by hour and day
- segment-level shifted energy
- stress-hour reduction
- renewable-surplus absorption
- sensitivity to forecast error

## Suggested Minimal Mathematical Formulation

For a tractable hand-in version, use the following simplified optimization for each rolling window.

### Sets

- `t`: current hour
- `u`: future hours in `t..t+119`
- `s`: segment
- `w`: forecast scenario

### Parameters

- `P[w,u]`: forecast electricity price
- `B[s,u]`: baseline load by segment
- `F[s,u]`: flexible share of segment load
- `W[s]`: maximum wait
- `C[u]`: grid stress penalty coefficient
- `R[u]`: renewable surplus reward coefficient

### Decision Variable

- `x[s,t,u]`: amount of flexible segment energy from origin hour `t` served in hour `u`

### Objective

```text
min sum_w pi_w * sum_s sum_u x[s,t,u] * P[w,u]
  + sum_s sum_u x[s,t,u] * wait_penalty[s,u-t]
  + sum_u stress_penalty_coefficient[u] * shifted_load[u]^2
  - sum_u surplus_reward_coefficient[u] * shifted_load[u]
```

subject to:

```text
sum_u x[s,t,u] = flexible_energy_from_origin[s,t]
x[s,t,u] = 0 for u > t + W[s]
x[s,t,u] >= 0
```

This formulation is strong enough for the report and simple enough to implement.

## How This Connects To Your Existing Results

Conceptually, one important risk remains: more shifting is not always better shifting. That is exactly why the next model should include:

- explicit peak protection
- stress penalties
- rebound penalties
- surplus rewards

Without these, the simulator may move energy away from expensive hours but still create a worse system profile.

## Recommended Baseline Research Claims To Use In The Hand-In

You can support the model narrative with the following points:

- households are a meaningful share of total electricity demand and an even more important share during peak hours
- most classic household demand is not highly flexible
- wet appliances offer some flexibility, but EVs and heat pumps are the largest future sources
- only a subset of households responds strongly to prices
- forecast uncertainty matters, so point-forecast-based shift estimates can overstate real impact
- grid value comes not only from bill savings but from peak relief, ramp smoothing, and better use of surplus renewable generation

## Final Recommendation

The best next model for this project is:

- hourly
- rolling over `120` hours
- segmented by load type
- driven by `XG_Boost_full_Res.py` price forecasts
- uncertainty-aware through XGBoost residual-based scenarios
- optimized with both consumer and system objectives
- evaluated with both grid and supplier metrics

This will let the project move from "prices can influence shifting" to a much stronger claim:

**a forecast-guided household flexibility model can reduce expensive/stressed hours while also absorbing more renewable surplus, but only if the model penalizes rebound and secondary peaks.**
