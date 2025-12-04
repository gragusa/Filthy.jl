# Getting Started

This tutorial introduces the basic workflow for working with state space models in Filthy.jl. We'll cover:

1. Creating a model specification using templates
2. Computing the log-likelihood
3. Running the Kalman filter
4. Running the Kalman smoother
5. Estimating parameters via maximum likelihood

## The Local Level Model

We'll use the classic **local level model** (also known as the random walk plus noise model) as our running example. This model decomposes a time series into a slowly-evolving level plus observation noise:

```math
\begin{aligned}
y_t &= \mu_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2_\varepsilon) \\
\mu_{t+1} &= \mu_t + \eta_t, \quad \eta_t \sim N(0, \sigma^2_\eta)
\end{aligned}
```

In state space form:
- ``Z = [1]`` (the observation loads directly on the state)
- ``H = [\sigma^2_\varepsilon]`` (observation variance)
- ``T = [1]`` (random walk dynamics)
- ``R = [1]`` (state receives the shock directly)
- ``Q = [\sigma^2_\eta]`` (state innovation variance)

## Creating a Model Specification

```julia
using Filthy

# Create a local level model with initial parameter values
spec = local_level(σ_obs_init=15.0, σ_level_init=10.0)

# Inspect the specification
println("Model name: ", spec.name)
println("Number of states: ", spec.n_states)
println("Number of observations: ", spec.n_obs)
println("Parameters: ", param_names(spec))
println("Initial values: ", initial_values(spec))
```

Output:
```
Model name: LocalLevel
Number of states: 1
Number of observations: 1
Parameters: [:σ_obs, :σ_level]
Initial values: [15.0, 10.0]
```

## Simulating Data

Let's create some synthetic data to work with:

```julia
using Random
Random.seed!(42)

# True parameters
σ_obs_true = 15.0
σ_level_true = 10.0

# Simulate data
T = 100
μ = zeros(T)
y = zeros(1, T)  # Note: observations are p × T matrices

μ[1] = 0.0
for t in 2:T
    μ[t] = μ[t-1] + σ_level_true * randn()
end
for t in 1:T
    y[1, t] = μ[t] + σ_obs_true * randn()
end
```

## Computing the Log-Likelihood

Given a model specification and parameters, we can compute the log-likelihood:

```julia
# Build the state space components
θ = initial_values(spec)  # Use initial parameter values
ss = build_linear_state_space(spec, θ, y)

# Compute log-likelihood
ll = kalman_loglik(ss.p, y, ss.a1, ss.P1)
println("Log-likelihood: ", ll)
```

The `build_linear_state_space` function returns a named tuple with:
- `p`: A `KFParms` struct containing the system matrices (Z, H, T, R, Q)
- `a1`: Initial state mean vector
- `P1`: Initial state covariance matrix

### Using NamedTuple Parameters

You can also pass parameters as a NamedTuple for clarity:

```julia
using TransformVariables

# Build transformation to get constrained parameters
t = build_transformation(spec)

# Transform unconstrained values to NamedTuple
θ_nt = TransformVariables.transform(t, [2.7, 2.3])  # log-scale values
println(θ_nt)  # (σ_obs = 14.88..., σ_level = 9.97...)

# Compute log-likelihood with NamedTuple
ll = ssm_loglik(spec, θ_nt, y)
```

## Kalman Filter

The Kalman filter computes the sequence of filtered state estimates ``a_{t|t}`` and their covariances ``P_{t|t}``:

```julia
# Run the full Kalman filter
filter_result = kalman_filter_full(ss.p, y, ss.a1, ss.P1)

# Access results
a = filter_result.a      # Filtered state means (m × T+1)
P = filter_result.P      # Filtered state covariances (m × m × T+1)
v = filter_result.v      # Prediction errors (p × T)
F = filter_result.F      # Prediction error variances (p × p × T)
K = filter_result.K      # Kalman gains (m × p × T)
ll = filter_result.loglik # Log-likelihood

println("Final filtered state: ", a[:, end])
println("Log-likelihood: ", ll)
```

### Filter Output Structure

- `a[:, t]`: State estimate at time `t` given observations up to `t-1` (i.e., ``a_{t|t-1}``)
- `a[:, T+1]`: State estimate at time `T` given all observations (i.e., ``a_{T|T}``)
- `P[:, :, t]`: Corresponding covariance matrices
- `v[:, t]`: Innovation (prediction error) at time `t`
- `F[:, :, t]`: Innovation covariance at time `t`
- `K[:, :, t]`: Kalman gain at time `t`

## Kalman Smoother

The Kalman smoother computes smoothed state estimates using all available observations:

```julia
# Run the Kalman smoother
smooth_result = kalman_smoother(ss.p, y, ss.a1, ss.P1)

# Access results
α = smooth_result.α      # Smoothed state means (m × T)
V = smooth_result.V      # Smoothed state covariances (m × m × T)
r = smooth_result.r      # Smoothing cumulant (m × T)
N = smooth_result.N      # Smoothing variance cumulant (m × m × T)

println("Smoothed state at t=50: ", α[:, 50])
```

### Comparing Filtered and Smoothed Estimates

```julia
using Plots  # Optional, for visualization

# Compare filtered vs smoothed estimates
plot(1:T, μ, label="True level", linewidth=2)
plot!(1:T, a[1, 1:T], label="Filtered", linestyle=:dash)
plot!(1:T, α[1, :], label="Smoothed", linestyle=:dot)
xlabel!("Time")
ylabel!("Level")
```

The smoothed estimates are generally more accurate than filtered estimates because they use information from the entire sample.

## One-Step Filter and Smooth

For convenience, you can run both operations together:

```julia
result = kalman_filter_and_smooth(ss.p, y, ss.a1, ss.P1)

# Access both filter and smoother outputs
a = result.a          # Filtered states
α = result.α          # Smoothed states
loglik = result.loglik
```

## Parameter Estimation

Filthy.jl integrates with Optimization.jl for maximum likelihood estimation:

```julia
# Estimate parameters via MLE
result = optimize_ssm(spec, y)

# Access results
θ_mle = result.θ           # Estimated parameters (NamedTuple)
loglik = result.loglik     # Maximized log-likelihood
minimizer = result.minimizer  # Raw optimizer output

println("Estimated σ_obs: ", θ_mle.σ_obs)
println("Estimated σ_level: ", θ_mle.σ_level)
println("Log-likelihood: ", loglik)
```

### With Standard Errors

```julia
# Estimate with standard errors (uses Hessian)
result = optimize_ssm_with_stderr(spec, y)

println("σ_obs: ", θ_mle.σ_obs, " ± ", result.stderr.σ_obs)
println("σ_level: ", θ_mle.σ_level, " ± ", result.stderr.σ_level)
```

### Using Custom Initial Values

```julia
# Provide custom starting values
θ0 = (σ_obs = 20.0, σ_level = 5.0)
result = optimize_ssm(spec, y; θ0=θ0)
```

## Other Pre-Built Templates

Filthy.jl provides several other templates:

### Local Linear Trend

```julia
# Level + slope model
spec = local_linear_trend(
    σ_obs_init = 1.0,
    σ_level_init = 0.1,
    σ_slope_init = 0.01
)
# Parameters: [:σ_obs, :σ_level, :σ_slope]
```

### AR(1) Process

```julia
# AR(1) with measurement noise
spec = ar1(
    ρ_init = 0.9,         # AR coefficient
    σ_state_init = 1.0,   # State innovation std dev
    σ_obs_init = 0.5      # Observation noise std dev
)
# Parameters: [:ρ, :σ_state, :σ_obs]
```

### ARMA(p, q)

```julia
# ARMA(2, 1) model
spec = arma(2, 1;
    ar_init = [0.5, 0.3],
    ma_init = [0.2],
    σ_init = 1.0
)
# Parameters: [:ar_1, :ar_2, :ma_1, :σ]
```

## Working with Missing Data

Filthy.jl handles missing observations automatically:

```julia
# Create data with missing values
y_missing = copy(y)
y_missing[1, 20:25] .= NaN  # Mark as missing

# Filter and smooth work normally
result = kalman_filter_and_smooth(ss.p, y_missing, ss.a1, ss.P1)

# The smoother will interpolate through missing periods
```

## Next Steps

- Learn how to specify **[Custom Models](custom_models.md)** using the DSL
- Explore **[Dynamic Factor Models](dynamic_factor.md)** for multivariate analysis
- See the **[Core Functions](../api/core.md)** and **[DSL & Templates](../api/dsl.md)** for complete API documentation
