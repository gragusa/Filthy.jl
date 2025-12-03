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

export local_level, local_linear_trend, ar1, arma

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
