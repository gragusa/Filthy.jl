"""
    templates.jl

Pre-built state-space model templates for common models.

# Specifying Fixed vs Free Parameters

All template functions accept keyword arguments to control which parameters are
estimated vs fixed:

- Pass a `Symbol` or `nothing` to estimate (optimize) that parameter
- Pass a `Real` value to fix the parameter at that value

## Examples

```julia
# Estimate both σ_obs and σ_level (default)
spec = local_level()

# Fix σ_obs at 15.0, estimate σ_level starting from 10.0
spec = local_level(σ_obs=15.0, σ_level=10.0)
# Wait, that's ambiguous! Use explicit :free marker:

# Fix σ_obs at 15.0, estimate σ_level
spec = local_level(σ_obs=15.0, σ_level=:free)

# Estimate σ_obs, fix σ_level at 10.0
spec = local_level(σ_obs=:free, σ_level=10.0)

# Set initial values for estimation
spec = local_level(σ_obs=(init=15.0,), σ_level=(init=10.0,))

# Full control: bounds + initial value
spec = local_level(σ_obs=(init=15.0, lower=1.0, upper=200.0), σ_level=:free)
```
"""

export local_level, local_linear_trend, ar1, arma, dynamic_factor

"""
    ParamSpec

Union type for parameter specification in templates:
- `Symbol` (:free) - estimate with default bounds/init
- `Real` - fix at this value
- `NamedTuple` - estimate with custom (init=, lower=, upper=)
"""
const ParamSpec = Union{Symbol, Real, NamedTuple}

"""
    parse_param_spec(name, spec, default_init, default_lower, default_upper)

Parse a parameter specification into either an SSMParameter (free) or FixedValue.
Returns (is_free::Bool, param_or_value)
"""
function parse_param_spec(name::Symbol, spec::Symbol, default_init, default_lower, default_upper)
    spec == :free || throw(ArgumentError("Unknown symbol :$spec for parameter $name. Use :free or a numeric value."))
    (true, SSMParameter(name; lower=default_lower, upper=default_upper, init=default_init))
end

function parse_param_spec(name::Symbol, spec::Real, default_init, default_lower, default_upper)
    (false, FixedValue(Float64(spec)))
end

function parse_param_spec(name::Symbol, spec::NamedTuple, default_init, default_lower, default_upper)
    init = get(spec, :init, default_init)
    lower = get(spec, :lower, default_lower)
    upper = get(spec, :upper, default_upper)
    (true, SSMParameter(name; lower=lower, upper=upper, init=init))
end

"""
    local_level(; σ_obs=:free, σ_level=:free, diffuse=true)

Create specification for local level (random walk + noise) model.

Model:
    yₜ = μₜ + εₜ,  εₜ ~ N(0, σ²_obs)
    μₜ₊₁ = μₜ + ηₜ,  ηₜ ~ N(0, σ²_level)

# Arguments
- `σ_obs`: Observation noise std dev. Use `:free` to estimate, a number to fix,
           or `(init=, lower=, upper=)` for custom estimation settings.
- `σ_level`: Level noise std dev. Same options as σ_obs.
- `diffuse`: Use diffuse initialization (default: true)

# Examples
```julia
# Estimate both (default)
spec = local_level()

# Fix observation noise, estimate level noise
spec = local_level(σ_obs=15.0, σ_level=:free)

# Custom initial value and bounds
spec = local_level(σ_obs=(init=15.0, lower=1.0, upper=100.0))
```
"""
function local_level(; σ_obs::ParamSpec=:free, σ_level::ParamSpec=:free, diffuse=true)
    # Parse parameter specifications
    obs_free, obs_spec = parse_param_spec(:σ_obs, σ_obs, 1.0, 1e-6, Inf)
    level_free, level_spec = parse_param_spec(:σ_level, σ_level, 1.0, 1e-6, Inf)

    # Build parameter list (only free parameters)
    params = SSMParameter{Float64}[]
    obs_free && push!(params, obs_spec)
    level_free && push!(params, level_spec)

    # Z = [1]
    Z = SSMMatrixSpec((1, 1))
    Z.elements[(1, 1)] = FixedValue(1.0)

    # H = σ²_obs (either ParameterRef or FixedValue)
    H = SSMMatrixSpec((1, 1))
    if obs_free
        H.elements[(1, 1)] = ParameterRef(:σ_obs, x -> x^2)
    else
        H.elements[(1, 1)] = FixedValue(obs_spec.value^2)
    end

    # T = [1]
    T = SSMMatrixSpec((1, 1))
    T.elements[(1, 1)] = FixedValue(1.0)

    # R = [1]
    R = SSMMatrixSpec((1, 1))
    R.elements[(1, 1)] = FixedValue(1.0)

    # Q = σ²_level (either ParameterRef or FixedValue)
    Q = SSMMatrixSpec((1, 1))
    if level_free
        Q.elements[(1, 1)] = ParameterRef(:σ_level, x -> x^2)
    else
        Q.elements[(1, 1)] = FixedValue(level_spec.value^2)
    end

    # Initial state
    a1 = [FixedValue(0.0)]
    P1_val = diffuse ? 1e7 : 1e4
    P1 = SSMMatrixSpec((1, 1))
    P1.elements[(1, 1)] = FixedValue(P1_val)

    SSMSpec(:LocalLevel, 1, 1, 1, params, Z, H, T, R, Q, a1, P1)
end

"""
    local_linear_trend(; σ_obs=:free, σ_level=:free, σ_slope=:free, diffuse=true)

Create specification for local linear trend model.

Model:
    yₜ = μₜ + εₜ,  εₜ ~ N(0, σ²_obs)
    μₜ₊₁ = μₜ + νₜ + ηₜ,  ηₜ ~ N(0, σ²_level)
    νₜ₊₁ = νₜ + ζₜ,  ζₜ ~ N(0, σ²_slope)

State: [μₜ, νₜ]

# Arguments
- `σ_obs`, `σ_level`, `σ_slope`: Use `:free` to estimate, a number to fix,
  or `(init=, lower=, upper=)` for custom settings.
"""
function local_linear_trend(; σ_obs::ParamSpec=:free, σ_level::ParamSpec=:free,
                              σ_slope::ParamSpec=:free, diffuse=true)
    # Parse parameter specifications
    obs_free, obs_spec = parse_param_spec(:σ_obs, σ_obs, 1.0, 1e-6, Inf)
    level_free, level_spec = parse_param_spec(:σ_level, σ_level, 0.1, 1e-6, Inf)
    slope_free, slope_spec = parse_param_spec(:σ_slope, σ_slope, 0.01, 1e-6, Inf)

    # Build parameter list (only free parameters)
    params = SSMParameter{Float64}[]
    obs_free && push!(params, obs_spec)
    level_free && push!(params, level_spec)
    slope_free && push!(params, slope_spec)

    # Z = [1 0]
    Z = SSMMatrixSpec((1, 2))
    Z.elements[(1, 1)] = FixedValue(1.0)

    # H = σ²_obs
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = obs_free ? ParameterRef(:σ_obs, x -> x^2) : FixedValue(obs_spec.value^2)

    # T = [1 1; 0 1]
    T = SSMMatrixSpec((2, 2))
    T.elements[(1, 1)] = FixedValue(1.0)
    T.elements[(1, 2)] = FixedValue(1.0)
    T.elements[(2, 2)] = FixedValue(1.0)

    # R = I
    R = SSMMatrixSpec((2, 2))
    R.elements[(1, 1)] = FixedValue(1.0)
    R.elements[(2, 2)] = FixedValue(1.0)

    # Q = diag(σ²_level, σ²_slope)
    Q = SSMMatrixSpec((2, 2))
    Q.elements[(1, 1)] = level_free ? ParameterRef(:σ_level, x -> x^2) : FixedValue(level_spec.value^2)
    Q.elements[(2, 2)] = slope_free ? ParameterRef(:σ_slope, x -> x^2) : FixedValue(slope_spec.value^2)

    # Initial state
    a1 = [FixedValue(0.0), FixedValue(0.0)]
    P1_val = diffuse ? 1e7 : 1e4
    P1 = SSMMatrixSpec((2, 2))
    P1.elements[(1, 1)] = FixedValue(P1_val)
    P1.elements[(2, 2)] = FixedValue(P1_val)

    SSMSpec(:LocalLinearTrend, 2, 1, 2, params, Z, H, T, R, Q, a1, P1)
end

"""
    ar1(; ρ=:free, σ_obs=:free, σ_state=:free)

Create specification for AR(1) plus noise model.

Model:
    yₜ = xₜ + εₜ,  εₜ ~ N(0, σ²_obs)
    xₜ₊₁ = ρ xₜ + ηₜ,  ηₜ ~ N(0, σ²_state)

# Arguments
- `ρ`: AR coefficient (|ρ| < 1). Use `:free`, a number, or `(init=, lower=, upper=)`.
- `σ_obs`, `σ_state`: Noise std devs. Same options.
"""
function ar1(; ρ::ParamSpec=:free, σ_obs::ParamSpec=:free, σ_state::ParamSpec=:free)
    # Parse parameter specifications
    rho_free, rho_spec = parse_param_spec(:ρ, ρ, 0.8, -0.9999, 0.9999)
    obs_free, obs_spec = parse_param_spec(:σ_obs, σ_obs, 1.0, 1e-6, Inf)
    state_free, state_spec = parse_param_spec(:σ_state, σ_state, 1.0, 1e-6, Inf)

    # Build parameter list
    params = SSMParameter{Float64}[]
    rho_free && push!(params, rho_spec)
    obs_free && push!(params, obs_spec)
    state_free && push!(params, state_spec)

    # Z = [1]
    Z = SSMMatrixSpec((1, 1))
    Z.elements[(1, 1)] = FixedValue(1.0)

    # H = σ²_obs
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = obs_free ? ParameterRef(:σ_obs, x -> x^2) : FixedValue(obs_spec.value^2)

    # T = [ρ]
    T = SSMMatrixSpec((1, 1))
    T.elements[(1, 1)] = rho_free ? ParameterRef(:ρ) : FixedValue(rho_spec.value)

    # R = [1]
    R = SSMMatrixSpec((1, 1))
    R.elements[(1, 1)] = FixedValue(1.0)

    # Q = σ²_state
    Q = SSMMatrixSpec((1, 1))
    Q.elements[(1, 1)] = state_free ? ParameterRef(:σ_state, x -> x^2) : FixedValue(state_spec.value^2)

    # Initial state (diffuse)
    a1 = [FixedValue(0.0)]
    P1 = SSMMatrixSpec((1, 1))
    P1.elements[(1, 1)] = FixedValue(1e4)

    SSMSpec(:AR1, 1, 1, 1, params, Z, H, T, R, Q, a1, P1)
end

"""
    arma(p::Int, q::Int; ar_init=nothing, ma_init=nothing, σ_init=1.0)

Create specification for ARMA(p,q) model in state-space form.

Uses the innovations state-space representation.

Parameters: ar coefficients φ₁...φₚ, ma coefficients θ₁...θᵧ, σ (innovation std dev)
"""
function arma(p::Int, q::Int; ar_init=nothing, ma_init=nothing, σ_init=1.0)
    # State dimension is max(p, q+1)
    r = max(p, q + 1)

    # Default initial values
    if ar_init === nothing
        ar_init = fill(0.5 / p, p)
    end
    if ma_init === nothing
        ma_init = fill(0.0, q)
    end

    # Parameters
    params = SSMParameter{Float64}[]

    # AR coefficients
    for i in 1:p
        push!(params, SSMParameter(Symbol("φ$i"); lower=-Inf, upper=Inf, init=ar_init[i]))
    end

    # MA coefficients
    for i in 1:q
        push!(params, SSMParameter(Symbol("θ$i"); lower=-Inf, upper=Inf, init=ma_init[i]))
    end

    # Innovation variance
    push!(params, SSMParameter(:σ; lower=1e-6, upper=Inf, init=σ_init))

    # Z = [1 0 ... 0]
    Z = SSMMatrixSpec((1, r))
    Z.elements[(1, 1)] = FixedValue(1.0)

    # H = 0 (no observation noise in pure ARMA)
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = FixedValue(0.0)

    # T = [φ₁ 1 0 ... 0]
    #     [φ₂ 0 1 ... 0]
    #     [⋮  ⋮ ⋮ ⋱  ⋮]
    #     [φᵣ 0 0 ... 0]
    T = SSMMatrixSpec((r, r))
    for i in 1:r
        if i <= p
            T.elements[(i, 1)] = ParameterRef(Symbol("φ$i"))
        end
        if i < r
            T.elements[(i, i + 1)] = FixedValue(1.0)
        end
    end

    # R = [1; θ₁; θ₂; ...; θᵧ; 0; ...]
    R = SSMMatrixSpec((r, 1))
    R.elements[(1, 1)] = FixedValue(1.0)
    for i in 1:q
        R.elements[(i + 1, 1)] = ParameterRef(Symbol("θ$i"))
    end

    # Q = σ²
    Q = SSMMatrixSpec((1, 1))
    Q.elements[(1, 1)] = ParameterRef(:σ, x -> x^2)

    # Initial state (diffuse)
    a1 = [FixedValue(0.0) for _ in 1:r]
    P1 = SSMMatrixSpec((r, r))
    for i in 1:r
        P1.elements[(i, i)] = FixedValue(1e7)
    end

    SSMSpec(Symbol("ARMA($p,$q)"), r, 1, 1, params, Z, H, T, R, Q, a1, P1)
end

"""
    dynamic_factor(n_obs, n_factors; factor_lags=1, obs_lags=0, correlated_errors=false,
                   loadings_init=0.5, ar_init=0.5, σ_obs_init=1.0,
                   σ_factor_init=1.0, diffuse=true)

Create specification for a dynamic factor model with VAR factor dynamics and
optional lagged factor loadings.

Model:
    yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + ... + Λₛ fₜ₋ₛ + εₜ,  εₜ ~ N(0, H)
    fₜ = Φ₁ fₜ₋₁ + Φ₂ fₜ₋₂ + ... + Φₚ fₜ₋ₚ + ηₜ,  ηₜ ~ N(0, Q)

where:
- yₜ is n_obs × 1 vector of observations
- fₜ is n_factors × 1 vector of latent factors
- Λₗ is n_obs × n_factors factor loadings matrix for lag l (l = 0, ..., s)
- Φₗ is n_factors × n_factors AR coefficient matrix for lag l (diagonal)
- H is observation error covariance (diagonal or full if correlated_errors=true)
- Q is factor innovation covariance (diagonal)
- s = obs_lags, p = factor_lags

The model is cast in companion form with state vector:
    αₜ = [fₜ', fₜ₋₁', ..., fₜ₋ₘ₊₁']'  where m = max(p, s+1)

# Identification

For identification, the first n_factors rows of Λ₀ (contemporaneous loadings)
are set to an identity matrix:
- λ₀_{i,i} = 1 for i ≤ n_factors
- λ₀_{i,j} = 0 for i < j ≤ n_factors

All other loadings (including lagged) are free parameters.

# Arguments
- `n_obs::Int`: Number of observable variables
- `n_factors::Int`: Number of latent factors (must be < n_obs)
- `factor_lags::Int=1`: Number of lags in factor VAR dynamics (p)
- `obs_lags::Int=0`: Number of lagged factors that load onto observations (s).
                     If s > 0, observations depend on fₜ, fₜ₋₁, ..., fₜ₋ₛ.
- `correlated_errors::Bool=false`: If true, H is a full covariance matrix (inexact DFM).
                                   If false, H is diagonal (exact DFM).
- `loadings_init::Real=0.5`: Initial value for free factor loadings
- `ar_init::Real=0.5`: Initial value for AR coefficients (divided by lag number)
- `σ_obs_init::Real=1.0`: Initial value for observation error std devs
- `σ_factor_init::Real=1.0`: Initial value for factor innovation std devs
- `diffuse::Bool=true`: Use diffuse initialization for factors

# State Space Representation

State dimension is `n_factors * max(factor_lags, obs_lags + 1)`.

State vector: αₜ = [fₜ', fₜ₋₁', ..., fₜ₋ₘ₊₁']'

Transition matrix (companion form):
    T = [Φ₁  Φ₂  ... Φₚ  0  ... 0]   (padded with zeros if m > p)
        [I   0   ... 0   0  ... 0]
        [0   I   ... 0   0  ... 0]
        [⋮   ⋮   ⋱   ⋮   ⋮  ⋱   ⋮]
        [0   0   ... I   0  ... 0]

Observation matrix: Z = [Λ₀  Λ₁  ...  Λₛ  0  ...  0]

# Example
```julia
# Standard 1-factor model (no lagged loadings)
spec = dynamic_factor(4, 1)

# 1-factor model with 2 factor lags, no lagged loadings
spec = dynamic_factor(4, 1; factor_lags=2)
# yₜ = Λ₀ fₜ + εₜ
# fₜ = φ₁ fₜ₋₁ + φ₂ fₜ₋₂ + ηₜ

# 1-factor model with lagged loadings (s=1)
spec = dynamic_factor(4, 1; factor_lags=2, obs_lags=1)
# yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + εₜ
# fₜ = φ₁ fₜ₋₁ + φ₂ fₜ₋₂ + ηₜ
# Parameters: λ0_i_j (contemporaneous), λ1_i_j (lag 1 loadings)

# 2-factor model with obs_lags=2, factor_lags=1
spec = dynamic_factor(5, 2; factor_lags=1, obs_lags=2)
# yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + Λ₂ fₜ₋₂ + εₜ
# State: [f₁ₜ, f₂ₜ, f₁ₜ₋₁, f₂ₜ₋₁, f₁ₜ₋₂, f₂ₜ₋₂]
```

# Parameters Created
- `λ0_i_j`: Contemporaneous loadings (i > j for identification in first k rows)
- `λl_i_j`: Loadings for lag l (l = 1, ..., obs_lags), all free
- `φ_i_l`: AR coefficient for factor i at lag l (l = 1, ..., factor_lags)
- `σ_obs_i`: Observation error std devs (if correlated_errors=false)
- `H_σ_i`, `H_corr_i`: Covariance parameters (if correlated_errors=true)
- `σ_factor_i`: Factor innovation std devs
"""
function dynamic_factor(n_obs::Int, n_factors::Int;
                        factor_lags::Int=1,
                        obs_lags::Int=0,
                        correlated_errors::Bool=false,
                        loadings_init::Real=0.5,
                        ar_init::Real=0.5,
                        σ_obs_init::Real=1.0,
                        σ_factor_init::Real=1.0,
                        diffuse::Bool=true)
    # Validation
    n_factors < n_obs || throw(ArgumentError("n_factors must be < n_obs for identification"))
    n_factors >= 1 || throw(ArgumentError("n_factors must be >= 1"))
    factor_lags >= 1 || throw(ArgumentError("factor_lags must be >= 1"))
    obs_lags >= 0 || throw(ArgumentError("obs_lags must be >= 0"))

    # State dimension = n_factors * max(factor_lags, obs_lags + 1)
    # We need enough lags in the state to support both factor dynamics and observation loadings
    n_state_lags = max(factor_lags, obs_lags + 1)
    n_states = n_factors * n_state_lags
    # Number of shocks = n_factors (only current factors have innovations)
    n_shocks = n_factors

    # Build using custom_ssm approach for flexibility
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol,Any}()

    # ==========================================
    # Z matrix (loadings): n_obs × n_states
    # ==========================================
    # Z = [Λ₀  Λ₁  ...  Λₛ  0  ...  0]
    # where each Λₗ is n_obs × n_factors
    #
    # For identification: upper-left block of Λ₀ is identity
    Z = SSMMatrixSpec((n_obs, n_states))

    # Contemporaneous loadings (Λ₀) - columns 1:n_factors
    for i in 1:n_obs
        for j in 1:n_factors
            if i == j && i <= n_factors
                # Diagonal of identity block: fixed at 1
                Z.elements[(i, j)] = FixedValue(1.0)
            elseif i < j && i <= n_factors
                # Upper triangle of identity block: fixed at 0
                Z.elements[(i, j)] = FixedValue(0.0)
            else
                # Free loading parameter (contemporaneous)
                pname = Symbol("λ0_$(i)_$(j)")
                push!(params, SSMParameter(pname; lower=-Inf, upper=Inf, init=loadings_init))
                push!(param_set, pname)
                Z.elements[(i, j)] = ParameterRef(pname)
            end
        end
    end

    # Lagged loadings (Λ₁, Λ₂, ..., Λₛ) - all free parameters
    for lag in 1:obs_lags
        col_offset = lag * n_factors
        for i in 1:n_obs
            for j in 1:n_factors
                col = col_offset + j
                # All lagged loadings are free (no identification restrictions)
                pname = Symbol("λ$(lag)_$(i)_$(j)")
                # Use smaller initial values for lagged loadings
                init_val = loadings_init / (lag + 1)
                push!(params, SSMParameter(pname; lower=-Inf, upper=Inf, init=init_val))
                push!(param_set, pname)
                Z.elements[(i, col)] = ParameterRef(pname)
            end
        end
    end
    # Remaining columns (beyond obs_lags) are zero - use default

    # ==========================================
    # H matrix (observation errors): n_obs × n_obs
    # ==========================================
    if correlated_errors
        # Full covariance matrix using CovMatrixExpr
        prefix = :H
        n = n_obs

        # Create σ parameters (positive)
        σ_param_names = Symbol[]
        for i in 1:n
            pname = Symbol("$(prefix)_σ_$i")
            if !(pname in param_set)
                push!(params, SSMParameter(pname; lower=0.0, upper=Inf, init=σ_obs_init))
                push!(param_set, pname)
            end
            push!(σ_param_names, pname)
        end

        # Create correlation parameters (unconstrained)
        n_corr = n * (n - 1) ÷ 2
        corr_param_names = Symbol[]
        for i in 1:n_corr
            pname = Symbol("$(prefix)_corr_$i")
            if !(pname in param_set)
                push!(params, SSMParameter(pname; lower=-Inf, upper=Inf, init=0.0))
                push!(param_set, pname)
            end
            push!(corr_param_names, pname)
        end

        # Store CovMatrixExpr
        matrix_exprs[:H] = CovMatrixExpr(n, σ_param_names, corr_param_names)
        H = SSMMatrixSpec((n_obs, n_obs))  # Placeholder
    else
        # Diagonal covariance
        H = SSMMatrixSpec((n_obs, n_obs))
        for i in 1:n_obs
            pname = Symbol("σ_obs_$i")
            push!(params, SSMParameter(pname; lower=1e-6, upper=Inf, init=σ_obs_init))
            push!(param_set, pname)
            H.elements[(i, i)] = ParameterRef(pname, x -> x^2)
        end
    end

    # ==========================================
    # T matrix (factor dynamics): n_states × n_states
    # ==========================================
    # Companion form:
    # T = [Φ₁  Φ₂  ... Φₚ  0  ... 0]   <- n_factors rows (AR coefficients, padded)
    #     [I   0   ... 0   0  ... 0]
    #     [0   I   ... 0   0  ... 0]
    #     [⋮   ⋮   ⋱   ⋮   ⋮  ⋱   ⋮]
    #     [0   0   ... I   0  ... 0]
    #
    # Each Φₗ is n_factors × n_factors diagonal matrix
    T = SSMMatrixSpec((n_states, n_states))

    # First n_factors rows: AR coefficient blocks (only up to factor_lags)
    for i in 1:n_factors
        for lag in 1:factor_lags
            # Column block for lag l starts at (lag-1)*n_factors + 1
            col = (lag - 1) * n_factors + i
            pname = Symbol("φ_$(i)_$(lag)")
            # Initialize with decreasing values for higher lags
            init_val = ar_init / lag
            push!(params, SSMParameter(pname; lower=-0.9999, upper=0.9999, init=init_val))
            push!(param_set, pname)
            T.elements[(i, col)] = ParameterRef(pname)
        end
        # Columns beyond factor_lags are zero (default)
    end

    # Identity blocks for shifting lags (rows n_factors+1 to n_states)
    if n_state_lags > 1
        for lag in 1:(n_state_lags - 1)
            for i in 1:n_factors
                row = lag * n_factors + i
                col = (lag - 1) * n_factors + i
                T.elements[(row, col)] = FixedValue(1.0)
            end
        end
    end

    # ==========================================
    # R matrix (selection): n_states × n_shocks
    # ==========================================
    # Only the first n_factors states (current factors) receive shocks
    # R = [I]   <- n_factors × n_factors identity
    #     [0]   <- (n_states - n_factors) × n_factors zeros
    R = SSMMatrixSpec((n_states, n_shocks))
    for i in 1:n_factors
        R.elements[(i, i)] = FixedValue(1.0)
    end

    # ==========================================
    # Q matrix (factor innovations): n_shocks × n_shocks
    # ==========================================
    # Diagonal covariance for factor innovations
    Q = SSMMatrixSpec((n_shocks, n_shocks))
    for i in 1:n_factors
        pname = Symbol("σ_factor_$i")
        push!(params, SSMParameter(pname; lower=1e-6, upper=Inf, init=σ_factor_init))
        push!(param_set, pname)
        Q.elements[(i, i)] = ParameterRef(pname, x -> x^2)
    end

    # ==========================================
    # Initial state
    # ==========================================
    a1 = [FixedValue(0.0) for _ in 1:n_states]
    P1_val = diffuse ? 1e7 : 1e4
    P1 = SSMMatrixSpec((n_states, n_states))
    for i in 1:n_states
        P1.elements[(i, i)] = FixedValue(P1_val)
    end

    SSMSpec(:DynamicFactor, n_states, n_obs, n_shocks, params,
            Z, H, T, R, Q, a1, P1, matrix_exprs)
end
