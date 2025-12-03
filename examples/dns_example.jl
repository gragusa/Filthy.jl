#=
Dynamic Nelson-Siegel (DNS) Model Example
=========================================

This example demonstrates how to estimate a Dynamic Nelson-Siegel model
for the term structure of interest rates using Filthy.jl.

The DNS model represents yields as:
    y(τ) = β₁ + β₂ * L₁(λ,τ) + β₃ * L₂(λ,τ) + ε

Where:
    - β₁ = level factor (long-term rate)
    - β₂ = slope factor
    - β₃ = curvature factor
    - L₁(λ,τ) = (1 - exp(-λτ))/(λτ)           [slope loading]
    - L₂(λ,τ) = (1 - exp(-λτ))/(λτ) - exp(-λτ) [curvature loading]

In state-space form:
    Observation: yₜ = Z(λ) * αₜ + εₜ,  εₜ ~ N(0, H)
    State:       αₜ₊₁ = T * αₜ + ηₜ,   ηₜ ~ N(0, Q)

Where αₜ = [β₁ₜ, β₂ₜ, β₃ₜ]' are the latent factors.

NOTE: DNS models can be numerically challenging for gradient-based
optimization because the Kalman filter may encounter near-singular
covariance matrices during the search. Good starting values are essential.

================================================================================
MatrixExpr: Building Parameter-Dependent Matrices
================================================================================

When a state-space matrix depends on parameters in a non-trivial way (e.g., the
DNS loading matrix Z(λ)), use `MatrixExpr` to define it.

## Constructor

    MatrixExpr(params, data, builder; dims=nothing)

## Arguments

1. `params::Vector{SSMParameter{Float64}}`
   A vector of SSMParameter objects specifying the parameters used in this matrix.
   Each SSMParameter has:
   - `name::Symbol` - parameter name (e.g., :λ)
   - `lower::Float64` - lower bound (use -Inf for unbounded)
   - `upper::Float64` - upper bound (use Inf for unbounded)
   - `init::Float64` - initial value for optimization
   - `transform` - transformation function (default: exp for positive params)

   Example:
       [SSMParameter(:λ, 0.001, 0.5, 0.0609)]  # positional: name, lower, upper, init
       [SSMParameter(:λ; lower=0.001, upper=0.5, init=0.0609)]  # keyword form

2. `data::NamedTuple`
   External (non-parameter) data needed to build the matrix. This data is fixed
   and not optimized. Pass `NamedTuple()` or `(;)` if no external data is needed.

   Examples:
       (maturities = [3, 6, 12, 24, 60, 120],)  # vector of maturities
       (n_yields = 31,)                          # scalar
       (maturities = τ, n_obs = n)               # multiple fields
       NamedTuple()                              # no external data

3. `builder::Function`
   A function with signature `(θ::Dict{Symbol,<:Real}, data::NamedTuple) -> Matrix`.

   - `θ` is a Dict mapping parameter names (Symbols) to their current values
   - `data` is the NamedTuple passed to MatrixExpr
   - Must return a matrix of the appropriate dimensions
   - For AD compatibility: use `typeof(param_value)` when creating arrays,
     e.g., `zeros(typeof(λ), n, m)` not `zeros(n, m)`

   Example:
       function my_builder(θ::Dict, data)
           λ = θ[:λ]              # extract parameter
           τ = data.maturities    # extract external data
           Z = zeros(typeof(λ), n, 3)  # AD-compatible allocation
           # ... fill Z ...
           return Z
       end

4. `dims::Tuple{Int,Int}` (keyword, optional)
   Expected output dimensions (rows, cols). If not provided, MatrixExpr will
   call `builder` once with initial parameter values to infer dimensions.

## Complete Example

    # Define the builder function
    function dns_loadings_builder(θ::Dict, data)
        λ = θ[:λ]
        τ = data.maturities
        n = length(τ)

        # Use typeof(λ) for AD compatibility
        Z = zeros(typeof(λ), n, 3)
        for i in 1:n
            λτ = λ * τ[i]
            L1 = (1 - exp(-λτ)) / λτ
            L2 = L1 - exp(-λτ)
            Z[i, 1] = one(λ)   # Level loading
            Z[i, 2] = L1       # Slope loading
            Z[i, 3] = L2       # Curvature loading
        end
        return Z
    end

    # Create the MatrixExpr
    Z_expr = MatrixExpr(
        [SSMParameter(:λ, 0.001, 0.5, 0.0609)],  # params: λ ∈ [0.001, 0.5], init=0.0609
        (maturities = [3, 6, 12, 24, 60, 120],), # data: fixed maturities
        dns_loadings_builder                      # builder function
    )

    # Use in custom_ssm
    spec = custom_ssm(Z = Z_expr, H = ..., T = ..., R = ..., Q = ..., ...)

## Convenience Functions

For common models, Filthy provides pre-built MatrixExpr constructors:

    build_dns_loadings(maturities; λ_init=0.0609, λ_lower=0.001, λ_upper=1.0)
    build_svensson_loadings(maturities; λ1_init=0.0609, λ2_init=0.03, ...)

================================================================================
=#

using Filthy
using DelimitedFiles
using LinearAlgebra
using Statistics

# ============================================
# Load Data
# ============================================

# Read the yield data
data_path = joinpath(@__DIR__, "monthlyyields.csv")
raw_data = readdlm(data_path, ',', Any, '\n')

# Extract headers and data
headers = String.(raw_data[1, 2:end])  # Skip date column
yields = Float64.(raw_data[2:end, 2:end])  # Skip date column and header row
dates = raw_data[2:end, 1]

# Extract maturities from column names (Y3 -> 3 months, Y360 -> 360 months)
maturities = [parse(Int, replace(h, "Y" => "")) for h in headers]

println("Loaded yield data:")
println("  Observations: $(size(yields, 1))")
println("  Maturities: $(length(maturities)) ($(minimum(maturities)) to $(maximum(maturities)) months)")
println("  Date range: $(dates[1]) to $(dates[end])")

# Transpose: Filthy expects (n_yields × T) matrix
y = Matrix(yields')

# ============================================
# Define the DNS Model
# ============================================

n_yields = length(maturities)
n_factors = 3

# Nelson-Siegel factor loadings: Z(λ) matrix
# Each row corresponds to a maturity τᵢ:
#   Z[i,:] = [1, (1-exp(-λτᵢ))/(λτᵢ), (1-exp(-λτᵢ))/(λτᵢ) - exp(-λτᵢ)]
function dns_loadings_builder(θ::Dict, data)
    λ = θ[:λ]
    τ = data.maturities
    n = length(τ)

    Z = zeros(typeof(λ), n, 3)
    for i in 1:n
        λτ = λ * τ[i]
        L1 = (1 - exp(-λτ)) / λτ      # Slope loading
        L2 = L1 - exp(-λτ)             # Curvature loading
        Z[i, 1] = one(λ)               # Level loading (always 1)
        Z[i, 2] = L1
        Z[i, 3] = L2
    end
    return Z
end

# Measurement noise covariance: H = σ²_ε * I
function H_builder(θ::Dict, data)
    σ_ε = θ[:σ_ε]
    n = data.n_yields
    σ_ε^2 * Matrix{typeof(σ_ε)}(I, n, n)
end

# State noise covariance: Q = diag(σ²_η₁, σ²_η₂, σ²_η₃)
function Q_builder(θ::Dict, data)
    σ_level = θ[:σ_level]
    σ_slope = θ[:σ_slope]
    σ_curve = θ[:σ_curve]
    T = promote_type(typeof(σ_level), typeof(σ_slope), typeof(σ_curve))
    diagm(0 => T[σ_level^2, σ_slope^2, σ_curve^2])
end

# Create MatrixExpr objects
Z_expr = MatrixExpr(
    [SSMParameter(:λ, 0.001, 0.5, 0.0609)],  # λ: decay parameter
    (maturities = maturities,),
    dns_loadings_builder
)

H_expr = MatrixExpr(
    [SSMParameter(:σ_ε, 0.0, Inf, 0.1)],     # σ_ε: measurement noise std
    (n_yields = n_yields,),
    H_builder
)

Q_expr = MatrixExpr(
    [SSMParameter(:σ_level, 0.0, Inf, 0.5),   # σ_level: level innovation std
     SSMParameter(:σ_slope, 0.0, Inf, 0.5),   # σ_slope: slope innovation std
     SSMParameter(:σ_curve, 0.0, Inf, 0.5)],  # σ_curve: curvature innovation std
    NamedTuple(),
    Q_builder
)

# Build the DNS state-space model
dns_spec = custom_ssm(
    Z = Z_expr,
    H = H_expr,
    T = diag_free(n_factors, :ϕ,              # AR(1) persistence for each factor
                  lower = -0.999,
                  upper = 0.999,
                  init = 0.99,
                  transform = identity),
    R = identity_mat(n_factors),
    Q = Q_expr,
    a1 = [mean(y[1,:]), 0.0, 0.0],            # Initial state: level ≈ short rate mean
    P1 = diag_fixed(100.0, n_factors),        # Diffuse initialization
    name = :DNS
)

println("\nDNS Model specification:")
println("  Parameters: ", param_names(dns_spec))
println("  Number of parameters: ", n_params(dns_spec))

# ============================================
# Compute Log-Likelihood
# ============================================

# Create log-density object
ld = SSMLogDensity(dns_spec, y)

# Get initial unconstrained parameters
θ_init = initial_values(dns_spec)
θ_u = transform_to_unconstrained(dns_spec, θ_init)

# Compute log-likelihood at initial values
ll_init = logdensity(ld, θ_u)
println("\nLog-likelihood at initial values: ", round(ll_init, digits=2))

# ============================================
# Estimate with Gradient-Based Optimization
# ============================================

println("\nEstimating DNS model...")
println("  This may take a minute...")

# For gradient-based optimization, wrap with LogDensityProblemsAD
using LogDensityProblems
using LogDensityProblemsAD
using ForwardDiff

using Optimization
using OptimizationOptimJL

f = Optimization.OptimizationFunction(
    (θ, p) -> -LogDensityProblems.logdensity(p, θ),  # Negative log-likelihood
    Optimization.AutoForwardDiff()
)

prob = Optimization.OptimizationProblem(f, θ_u, ld)

sol = solve(prob, OptimizationOptimJL.BFGS(); maxiters=500)

θ_opt, _ = transform_to_constrained(dns_spec, sol.u)


ss = build_linear_state_space(dns_spec, θ_opt, y)

filt = kalman_filter_full(ss.p, y, ss.a1, ss.P1)

# Extract the three DNS factors over time
level_factor = filt.a[1, :]      # β₁ₜ (level)
level_se = sqrt.(filt.P[1, 1, :])
slope_factor = filt.a[2, :]      # β₂ₜ (slope)
slope_se = sqrt.(filt.P[2, 2, :])
curvature_factor = filt.a[3, :]  # β₃ₜ (curvature)
curvature_se = sqrt.(filt.P[3, 3, :])

# ============================================
# Interpretation
# ============================================

println("\n" * "="^50)
println("Interpretation:")
println("="^50)
println("  λ = $(round(θ_opt.λ, digits=4))")
println("    → Peak of curvature loading at τ* = $(round(1/θ_opt.λ, digits=1)) months")
println()
println("  AR(1) persistence:")
println("    ϕ_level = $(round(θ_opt.ϕ_1, digits=3)) (level is $(θ_opt.ϕ_1 > 0.99 ? "very" : "") persistent)")
println("    ϕ_slope = $(round(θ_opt.ϕ_2, digits=3))")
println("    ϕ_curve = $(round(θ_opt.ϕ_3, digits=3))")
println()
println("  Measurement noise: σ_ε = $(round(θ_opt.σ_ε, digits=4))")
println("  State innovations:")
println("    σ_level = $(round(θ_opt.σ_level, digits=4))")
println("    σ_slope = $(round(θ_opt.σ_slope, digits=4))")
println("    σ_curve = $(round(θ_opt.σ_curve, digits=4))")
