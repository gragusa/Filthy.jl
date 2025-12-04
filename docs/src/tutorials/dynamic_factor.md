# Dynamic Factor Models

This tutorial covers the `dynamic_factor` template for specifying dynamic factor models (DFMs). Dynamic factor models are widely used in macroeconomics, finance, and other fields to extract common latent factors from a large panel of observed time series.

## Model Overview

The dynamic factor model in Filthy.jl supports:

- **Multiple factors** with identification restrictions
- **VAR factor dynamics** with configurable lag order
- **Lagged factor loadings** (factors affecting observations with delay)
- **Exact and inexact variants** (diagonal vs. full observation error covariance)

### Mathematical Formulation

The general dynamic factor model is:

```math
\begin{aligned}
y_t &= \Lambda_0 f_t + \Lambda_1 f_{t-1} + \cdots + \Lambda_s f_{t-s} + \varepsilon_t, \quad \varepsilon_t \sim N(0, H) \\
f_t &= \Phi_1 f_{t-1} + \Phi_2 f_{t-2} + \cdots + \Phi_p f_{t-p} + \eta_t, \quad \eta_t \sim N(0, Q)
\end{aligned}
```

Where:
- ``y_t`` is the ``n \times 1`` vector of observables
- ``f_t`` is the ``k \times 1`` vector of latent factors
- ``\Lambda_l`` is the ``n \times k`` loadings matrix for lag ``l``
- ``\Phi_l`` is the ``k \times k`` AR coefficient matrix for lag ``l``
- ``H`` is the observation error covariance (diagonal or full)
- ``Q`` is the factor innovation covariance (diagonal)

## Basic Usage

### Single-Factor Model

```julia
using Filthy

# 5 observables, 1 factor, AR(1) dynamics
spec = dynamic_factor(5, 1)

println("States: ", spec.n_states)      # 1
println("Observations: ", spec.n_obs)   # 5
println("Parameters: ", param_names(spec))
```

Output:
```
States: 1
Observations: 5
Parameters: [:λ0_2_1, :λ0_3_1, :λ0_4_1, :λ0_5_1, :σ_obs_1, :σ_obs_2,
             :σ_obs_3, :σ_obs_4, :σ_obs_5, :φ_1_1, :σ_factor_1]
```

### Identification

For identification, the first ``k`` rows of ``\Lambda_0`` (contemporaneous loadings) form an identity matrix:
- ``\lambda_{i,i} = 1`` for ``i \leq k``
- ``\lambda_{i,j} = 0`` for ``i < j \leq k``

This means the first factor loads with unit weight on the first observable, the second factor on the second observable, etc.

## Multiple Factors

```julia
# 6 observables, 2 factors
spec = dynamic_factor(6, 2)

println("Parameters: ", param_names(spec))
```

With 2 factors, the loading matrix ``\Lambda_0`` has structure:

```math
\Lambda_0 = \begin{bmatrix}
1 & 0 \\
\lambda_{2,1} & 1 \\
\lambda_{3,1} & \lambda_{3,2} \\
\lambda_{4,1} & \lambda_{4,2} \\
\lambda_{5,1} & \lambda_{5,2} \\
\lambda_{6,1} & \lambda_{6,2}
\end{bmatrix}
```

## Factor Dynamics with Multiple Lags

By default, factors follow AR(1) dynamics. To specify higher-order VAR dynamics:

```julia
# 5 observables, 2 factors, VAR(3) factor dynamics
spec = dynamic_factor(5, 2; factor_lags=3)

println("States: ", spec.n_states)  # 6 = 2 factors × 3 lags
```

The state vector is organized in companion form:
```math
\alpha_t = \begin{bmatrix} f_t \\ f_{t-1} \\ f_{t-2} \end{bmatrix}
```

This creates AR parameters ``\phi_{i,l}`` for factor ``i`` at lag ``l``:
- `φ_1_1`, `φ_1_2`, `φ_1_3` for factor 1
- `φ_2_1`, `φ_2_2`, `φ_2_3` for factor 2

## Lagged Factor Loadings

Factors can affect observations with a delay using the `obs_lags` parameter:

```julia
# Factors load on observations with up to 2 lags
spec = dynamic_factor(4, 1; factor_lags=2, obs_lags=2)

println("States: ", spec.n_states)  # 3 = max(2, 2+1) × 1
```

The observation equation becomes:
```math
y_t = \Lambda_0 f_t + \Lambda_1 f_{t-1} + \Lambda_2 f_{t-2} + \varepsilon_t
```

### Parameters for Lagged Loadings

With `obs_lags=2`:
- Contemporaneous loadings: `λ0_i_j` (with identification restrictions)
- Lag 1 loadings: `λ1_i_j` (all free)
- Lag 2 loadings: `λ2_i_j` (all free)

```julia
spec = dynamic_factor(4, 1; obs_lags=1)
println("Loading parameters: ",
    filter(p -> startswith(string(p), "λ"), param_names(spec)))
# [:λ0_2_1, :λ0_3_1, :λ0_4_1, :λ1_1_1, :λ1_2_1, :λ1_3_1, :λ1_4_1]
```

Note: All lagged loadings are unrestricted (no identification constraints).

## State Dimension

The state dimension is determined by:

```math
\text{n\_states} = k \times \max(p, s+1)
```

where ``k`` = number of factors, ``p`` = `factor_lags`, ``s`` = `obs_lags`.

This ensures the state vector contains enough lags to support both factor dynamics and lagged loadings.

| Example | k | p (`factor_lags`) | s (`obs_lags`) | n_states |
|---------|---|-------------------|----------------|----------|
| `dynamic_factor(5, 1)` | 1 | 1 | 0 | 1 |
| `dynamic_factor(5, 2; factor_lags=3)` | 2 | 3 | 0 | 6 |
| `dynamic_factor(5, 1; factor_lags=2, obs_lags=1)` | 1 | 2 | 1 | 2 |
| `dynamic_factor(5, 2; factor_lags=1, obs_lags=2)` | 2 | 1 | 2 | 6 |

## Exact vs. Inexact DFM

### Exact DFM (Default)

With `correlated_errors=false` (default), the observation error covariance is diagonal:

```julia
spec = dynamic_factor(5, 2)  # Exact DFM
# H = diag(σ²_obs_1, ..., σ²_obs_5)
```

### Inexact DFM

With `correlated_errors=true`, the observation errors can be correlated. This uses the `cov_free` parameterization for a full positive-definite covariance matrix:

```julia
spec = dynamic_factor(5, 2; correlated_errors=true)  # Inexact DFM

println("H parameters: ",
    filter(p -> startswith(string(p), "H_"), param_names(spec)))
# [:H_σ_1, :H_σ_2, :H_σ_3, :H_σ_4, :H_σ_5,
#  :H_corr_1, :H_corr_2, ..., :H_corr_10]
```

The inexact DFM is useful when:
- The factor model doesn't fully capture common variation
- Cross-sectional correlations remain after factoring

## Complete Example: Macroeconomic Factor Model

```julia
using Filthy
using Random
using LinearAlgebra

Random.seed!(42)

# Simulate a panel of 8 macro variables, T=200 observations
n_obs = 8
T = 200

# True DGP: 2 factors with AR(1) dynamics
true_loadings = [1.0 0.0;
                 0.5 1.0;
                 0.8 0.3;
                 0.3 0.7;
                 0.6 0.4;
                 0.4 0.6;
                 0.7 0.2;
                 0.2 0.8]
true_φ = [0.9, 0.7]
true_σ_factor = [0.5, 0.4]
true_σ_obs = fill(0.3, n_obs)

# Simulate factors
f = zeros(2, T)
for t in 2:T
    f[:, t] = Diagonal(true_φ) * f[:, t-1] + Diagonal(true_σ_factor) * randn(2)
end

# Generate observations
y = zeros(n_obs, T)
for t in 1:T
    y[:, t] = true_loadings * f[:, t] + Diagonal(true_σ_obs) * randn(n_obs)
end

# Specify the model
spec = dynamic_factor(n_obs, 2;
    loadings_init = 0.5,
    ar_init = 0.7,
    σ_obs_init = 0.5,
    σ_factor_init = 0.5
)

println("Number of parameters: ", n_params(spec))
println("Parameters: ", param_names(spec))

# Estimate via MLE
result = optimize_ssm(spec, y)

println("\nEstimated AR coefficients:")
println("  φ_1_1 = ", result.θ.φ_1_1, " (true: 0.9)")
println("  φ_2_1 = ", result.θ.φ_2_1, " (true: 0.7)")

# Extract smoothed factors
ss = build_linear_state_space(spec, result.θ, y)
smooth = kalman_smoother(ss.p, y, ss.a1, ss.P1)

println("\nCorrelation of estimated vs. true factors:")
println("  Factor 1: ", cor(smooth.α[1, :], f[1, :]))
println("  Factor 2: ", cor(smooth.α[2, :], f[2, :]))
```

## Function Parameters

```julia
dynamic_factor(n_obs, n_factors;
    factor_lags = 1,           # AR lags in factor dynamics (p)
    obs_lags = 0,              # Lagged factor loadings (s)
    correlated_errors = false, # Inexact DFM if true
    loadings_init = 0.5,       # Initial value for free loadings
    ar_init = 0.5,             # Initial AR coefficient (÷ lag number)
    σ_obs_init = 1.0,          # Initial obs error std dev
    σ_factor_init = 1.0,       # Initial factor innovation std dev
    diffuse = true             # Diffuse initialization for factors
)
```

## Parameters Created

| Parameter | Description | Constraints |
|-----------|-------------|-------------|
| `λ0_i_j` | Contemporaneous loading of obs `i` on factor `j` | Identified block |
| `λl_i_j` | Loading at lag `l` | All free |
| `φ_i_l` | AR coefficient for factor `i` at lag `l` | (-0.9999, 0.9999) |
| `σ_obs_i` | Obs error std dev (exact DFM) | > 0 |
| `H_σ_i` | Obs error std dev (inexact DFM) | > 0 |
| `H_corr_i` | Correlation parameters (inexact DFM) | Unconstrained |
| `σ_factor_i` | Factor innovation std dev | > 0 |

## Tips and Best Practices

1. **Number of factors**: Start with fewer factors and add more if needed. More factors = more parameters = harder estimation.

2. **Initial values**: Factor model estimation can be sensitive to initial values. Consider:
   - Running PCA first to initialize loadings
   - Using grid search over initial AR coefficients

3. **Identification**: The identity block in ``\Lambda_0`` is one common normalization. Others are possible (e.g., orthogonal factors).

4. **Lagged loadings**: Use `obs_lags > 0` when factors have delayed effects on observables (common in macro models).

5. **Correlated errors**: The inexact DFM adds many parameters. Only use when truly needed.

6. **Convergence**: Check that optimization converged (`result.retcode`). Try different initial values if not.

## Advanced: Stock-Watson Type DFM

For a Stock-Watson (2002) style DFM where factors follow independent AR(p) processes:

```julia
# 12 monthly macro series, 3 factors, each AR(2)
spec = dynamic_factor(12, 3;
    factor_lags = 2,
    correlated_errors = false  # Exact factor model
)

# This creates:
# - 12 × 3 loading parameters (minus identification restrictions)
# - 3 × 2 = 6 AR parameters
# - 12 observation error variances
# - 3 factor innovation variances
```

## Next Steps

- Learn about **[Custom Models](custom_models.md)** for more flexible specifications
- See the **[Core Functions](../api/core.md)** and **[DSL & Templates](../api/dsl.md)** for complete API documentation
