module Filthy

using StaticArrays
using ForwardDiff
using LinearAlgebra

include("types.jl")
include("filter_ad.jl")
include("smoother_ad.jl")

# Core Kalman filter/smoother exports
export KFParms, KFParms_static
export STATIC_THRESHOLD, to_static_if_small
export kalman_loglik, kalman_loglik_scalar
export kalman_filter_full, kalman_filter_full_scalar
export kalman_smoother, kalman_smoother_scalar, kalman_filter_and_smooth

# Include DSL submodule (before predict.jl which depends on SSMSpec)
include("dsl/dsl.jl")
using .DSL

# Prediction and forecasting (depends on DSL types)
include("predict.jl")

# Prediction and forecasting exports
export predict, forecast, forecast_paths

# Missing data utilities
export missing_to_nan, nan_to_missing, count_missing, ismissing_obs

# Re-export DSL components
export SSMParameter, SSMSpec, FixedValue, ParameterRef
export param_names, n_params, initial_values, param_bounds
export build_linear_state_space, objective_function, ssm_loglik
export local_level, local_linear_trend, ar1, arma
export custom_ssm, FreeParam, @P

# Matrix helpers
export diag_free, scalar_free, diag_fixed
export identity_mat, zeros_mat, ones_mat
export lower_triangular_free, symmetric_free
export selection_mat, companion_mat
export cov_free, CovFree

# Functional expressions (for DNS, Svensson, etc.)
export ParamExpr, MatrixExpr
export build_dns_loadings, build_svensson_loadings
export dns_loading1, dns_loading2

# Transformation (TransformVariables.jl integration)
export build_transformation
export transform_to_constrained, transform_to_unconstrained

# Log-density for optimization/sampling (LogDensityProblems.jl integration)
export SSMLogDensity, logdensity
export FlatPrior, NormalPrior, NormalPriorVec, InverseGammaPrior, CompositePrior

# Optimization (Optimization.jl integration)
export optimize_ssm, optimize_ssm_with_stderr

end # module
