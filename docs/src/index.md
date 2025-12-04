# Filthy.jl

*A Julia package for Linear State Space Models with Kalman filtering and smoothing.*

## Overview

Filthy.jl provides a comprehensive toolkit for working with linear Gaussian state space models. It combines high-performance Kalman filtering algorithms with an ergonomic domain-specific language (DSL) for model specification.

### Key Features

- **High-Performance Kalman Filter**: AD-compatible implementations supporting both scalar and matrix observations
- **Kalman Smoother**: Full backward smoothing with disturbance smoothing
- **Ergonomic DSL**: Specify models using intuitive matrix notation with automatic parameter handling
- **Pre-built Templates**: Common models like local level, local linear trend, AR(1), ARMA, and dynamic factor models
- **Optimization Integration**: Seamless integration with Optimization.jl for maximum likelihood estimation
- **Bayesian Support**: Prior specification and LogDensityProblems.jl interface for MCMC sampling
- **StaticArrays Support**: Automatic conversion to StaticArrays for small models
- **Missing Data Handling**: Native support for missing observations

## State Space Model Formulation

Filthy.jl implements the standard linear Gaussian state space model:

```math
\begin{aligned}
y_t &= Z \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, H) \\
\alpha_{t+1} &= T \alpha_t + R \eta_t, \quad \eta_t \sim N(0, Q)
\end{aligned}
```

where:
- ``y_t`` is the ``p \times 1`` observation vector
- ``\alpha_t`` is the ``m \times 1`` state vector
- ``Z`` is the ``p \times m`` observation matrix
- ``H`` is the ``p \times p`` observation error covariance
- ``T`` is the ``m \times m`` transition matrix
- ``R`` is the ``m \times r`` selection matrix
- ``Q`` is the ``r \times r`` state innovation covariance

## Installation

```julia
using Pkg
Pkg.add("Filthy")
```

## Quick Start

```julia
using Filthy

# Create a local level model specification
spec = local_level(σ_obs_init=15.0, σ_level_init=10.0)

# Your data (p × T matrix, where p = number of series, T = time periods)
y = randn(1, 100)

# Estimate parameters via maximum likelihood
result = optimize_ssm(spec, y)

# Access estimated parameters
println("Estimated parameters: ", result.θ)
println("Log-likelihood: ", result.loglik)

# Get filtered and smoothed states
ss = build_linear_state_space(spec, result.θ, y)
filter_result = kalman_filter_full(ss.p, y, ss.a1, ss.P1)
smooth_result = kalman_smoother(ss.p, y, ss.a1, ss.P1)
```

## Documentation Structure

- **[Getting Started](tutorials/getting_started.md)**: Basic tutorial covering Kalman filtering, smoothing, and parameter estimation
- **[Custom Models](tutorials/custom_models.md)**: Advanced tutorial on specifying custom state space models
- **[Dynamic Factor Models](tutorials/dynamic_factor.md)**: Tutorial on dynamic factor model specification
- **[Core Functions](api/core.md)**: Kalman filter and smoother API
- **[DSL & Templates](api/dsl.md)**: Model specification API

## Related Packages

- [StateSpaceModels.jl](https://github.com/LAMPSPUC/StateSpaceModels.jl): Alternative state space modeling package
- [Optimization.jl](https://github.com/SciML/Optimization.jl): Unified optimization interface used by Filthy.jl
- [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl): Parameter transformations used internally
- [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl): Interface for Bayesian inference

## License

Filthy.jl is released under the MIT License.
