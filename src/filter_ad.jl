"""
    filter_ad.jl

AD-compatible Kalman filter implementation for use with automatic differentiation.

This module provides pure functional filter implementations that:
- Avoid mutable state (no GrowableArrays)
- Avoid try-catch blocks
- Use element type propagation for AD compatibility
- Handle missing observations (marked as NaN)

Missing Data Handling:
When an observation contains NaN, the filter skips the measurement update
and propagates the state using only the transition equation:
    a_{t+1} = T * a_t
    P_{t+1} = T * P_t * T' + R * Q * R'
This is equivalent to having infinite observation noise for that period.
"""

using LinearAlgebra

# ============================================
# Missing observation detection
# ============================================

"""
    _has_missing(y_t::AbstractVector) -> Bool

Check if observation vector contains any NaN (missing) values.
"""
@inline _has_missing(y_t::AbstractVector) = any(isnan, y_t)

"""
    _has_missing(y_t::Real) -> Bool

Check if scalar observation is NaN (missing).
"""
@inline _has_missing(y_t::Real) = isnan(y_t)

"""
    kalman_loglik(p::KFParms, y, a1, P1) -> loglik

Compute the log-likelihood of a linear Gaussian state space model using the Kalman filter.

This is an AD-compatible implementation that:
- Uses pure functional operations (no in-place mutations)
- Avoids try-catch blocks
- Propagates element types correctly for automatic differentiation
- Handles missing observations (NaN values)

# Arguments
- `p::KFParms`: State space parameters (Z, H, T, R, Q)
- `y::AbstractMatrix`: Observations (p × n matrix, where p is observation dim, n is time).
                       Missing values should be marked as `NaN`.
- `a1::AbstractVector`: Initial state mean
- `P1::AbstractMatrix`: Initial state covariance

# Returns
- `loglik::Real`: Log-likelihood of the observed (non-missing) data

# State Space Model
```
y_t = Z * α_t + ε_t,    ε_t ~ N(0, H)
α_{t+1} = T * α_t + R * η_t,    η_t ~ N(0, Q)
```

# Missing Data
When `y[:, t]` contains any NaN, the observation is treated as missing.
The filter skips the measurement update and propagates the state:
    a_{t+1} = T * a_t
    P_{t+1} = T * P_t * T' + R * Q * R'
"""
function kalman_loglik(p::KFParms, y::AbstractMatrix,
                       a1::AbstractVector, P1::AbstractMatrix)
    n = size(y, 2)
    obs_dim = size(y, 1)

    # Get element type from parameters for AD compatibility
    T = promote_type(eltype(p.Z), eltype(p.H), eltype(p.T),
                     eltype(p.R), eltype(p.Q), eltype(a1), eltype(P1))

    # Initialize state
    a = convert(Vector{T}, copy(a1))
    P = convert(Matrix{T}, copy(P1))

    # Initialize log-likelihood (without constant term)
    loglik = zero(T)

    # Count non-missing observations for constant term
    n_obs = 0

    for t in 1:n
        y_t = y[:, t]

        # Check for missing observation
        if _has_missing(y_t)
            # Missing: skip update, just propagate state
            a = p.T * a
            P = p.T * P * p.T' + p.R * p.Q * p.R'
            continue
        end

        # Observation is present
        n_obs += 1

        # Innovation
        v = y_t - p.Z * a

        # Innovation covariance
        F = p.Z * P * p.Z' + p.H

        # For numerical stability with scalars
        if obs_dim == 1
            F_val = F[1, 1]
            # Check for non-positive variance
            if F_val <= zero(T)
                return T(-Inf)
            end
            Finv = one(T) / F_val
            logdetF = log(F_val)
            quad_form = v[1]^2 * Finv
        else
            # For multivariate case, use Cholesky for stability
            F_sym = Symmetric((F + F') / 2)  # Ensure symmetry
            chol_result = cholesky(F_sym; check=false)
            if !issuccess(chol_result)
                # F is not positive definite - return -Inf
                return T(-Inf)
            end
            Finv = inv(chol_result)
            logdetF = 2 * sum(log.(diag(chol_result.U)))
            quad_form = dot(v, Finv * v)
        end

        # Log-likelihood contribution
        loglik += -T(0.5) * (logdetF + quad_form)

        # Check for NaN/Inf
        if !isfinite(loglik)
            return T(-Inf)
        end

        # Kalman gain
        K = p.T * P * p.Z' * Finv

        # State update (predicted state for next period)
        a = p.T * a + K * v

        # Covariance update
        # P = T * P * T' + R * Q * R' - K * Z * P * T'
        # More numerically stable form:
        L = p.T - K * p.Z
        P = L * P * p.T' + p.R * p.Q * p.R'
    end

    # Constant term based on actual observations
    const_term = -obs_dim * n_obs * log(T(2π)) / 2

    return loglik + const_term
end

"""
    kalman_loglik_scalar(Z, H, T, R, Q, a1, P1, y) -> loglik

Scalar (univariate) version of the Kalman filter log-likelihood.

Optimized for the common case of scalar state and observation.
Handles missing observations marked as NaN.
"""
function kalman_loglik_scalar(Z::Real, H::Real, Tmat::Real, R::Real, Q::Real,
                              a1::Real, P1::Real, y::AbstractVector)
    n = length(y)
    T = promote_type(typeof(Z), typeof(H), typeof(Tmat), typeof(R),
                     typeof(Q), typeof(a1), typeof(P1))

    a = convert(T, a1)
    P = convert(T, P1)
    loglik = zero(T)
    n_obs = 0

    for t in 1:n
        y_t = y[t]

        # Check for missing observation
        if _has_missing(y_t)
            # Missing: skip update, just propagate state
            a = Tmat * a
            P = Tmat * P * Tmat + R * Q * R
            continue
        end

        n_obs += 1

        v = y_t - Z * a
        F = Z * P * Z + H

        # Check for non-positive variance
        if F <= zero(T)
            return T(-Inf)
        end

        Finv = one(T) / F
        loglik += -T(0.5) * (log(F) + v^2 * Finv)

        # Check for NaN/Inf
        if !isfinite(loglik)
            return T(-Inf)
        end

        K = Tmat * P * Z * Finv
        a = Tmat * a + K * v
        L = Tmat - K * Z
        P = L * P * Tmat + R * Q * R
    end

    const_term = -n_obs * log(T(2π)) / 2
    return loglik + const_term
end

"""
    kalman_filter_full(p::KFParms, y, a1, P1) -> NamedTuple

Run full Kalman filter returning states and covariances.

For `n` observations, returns `n+1` states:
- `a[:, 1]` = a₁ = E[α₁ | ∅] = initial state (a1)
- `a[:, t]` = E[αₜ | y₁:ₜ₋₁] for t = 2, ..., n+1
- `a[:, n+1]` = E[αₙ₊₁ | y₁:ₙ] = one-step-ahead forecast

Similarly for `P` (state covariances).

Returns a named tuple with:
- `loglik`: Log-likelihood (of non-missing observations)
- `a`: State means (m × (n+1))
- `P`: State covariances (m × m × (n+1))
- `v`: Innovations (p × n), NaN for missing observations
- `F`: Innovation covariances (p × p × n)
- `K`: Kalman gains (m × p × n), zero for missing observations
- `missing_mask`: BitVector indicating missing observations (length n)

# Missing Data
When `y[:, t]` contains any NaN, the observation is treated as missing.
The filter skips the measurement update and propagates the state.
"""
function kalman_filter_full(p::KFParms, y::AbstractMatrix,
                            a1::AbstractVector, P1::AbstractMatrix)
    n = size(y, 2)
    obs_dim = size(y, 1)
    state_dim = length(a1)

    ET = promote_type(eltype(p.Z), eltype(p.H), eltype(p.T),
                      eltype(p.R), eltype(p.Q), eltype(a1), eltype(P1))

    # State storage: n+1 entries (initial + after each observation)
    a_store = Matrix{ET}(undef, state_dim, n + 1)
    P_store = Array{ET}(undef, state_dim, state_dim, n + 1)

    # Innovation storage: n entries
    v_store = Matrix{ET}(undef, obs_dim, n)
    F_store = Array{ET}(undef, obs_dim, obs_dim, n)
    K_store = Array{ET}(undef, state_dim, obs_dim, n)
    missing_mask = BitVector(undef, n)

    # Initialize: a[1] = a1, P[1] = P1
    a = convert(Vector{ET}, copy(a1))
    P = convert(Matrix{ET}, copy(P1))
    a_store[:, 1] = a
    P_store[:, :, 1] = P

    loglik = zero(ET)
    n_obs = 0

    for t in 1:n
        y_t = y[:, t]

        # Check for missing observation
        if _has_missing(y_t)
            missing_mask[t] = true
            v_store[:, t] .= ET(NaN)
            F_store[:, :, t] = p.Z * P * p.Z' + p.H
            K_store[:, :, t] .= zero(ET)

            # Propagate state without update
            a = p.T * a
            P = p.T * P * p.T' + p.R * p.Q * p.R'
        else
            missing_mask[t] = false
            n_obs += 1

            # Innovation
            v = y_t - p.Z * a
            F = p.Z * P * p.Z' + p.H

            if obs_dim == 1
                F_val = F[1, 1]
                Finv = reshape([one(ET) / F_val], 1, 1)
                logdetF = log(F_val)
                quad_form = v[1]^2 / F_val
            else
                Finv = inv(F)
                logdetF = logdet(F)
                quad_form = dot(v, Finv * v)
            end

            loglik += -ET(0.5) * (logdetF + quad_form)

            K = p.T * P * p.Z' * Finv

            v_store[:, t] = v
            F_store[:, :, t] = F
            K_store[:, :, t] = K

            # Update and predict
            a_filt = a + P * p.Z' * Finv * v
            P_filt = P - P * p.Z' * Finv * p.Z * P

            a = p.T * a_filt
            P = p.T * P_filt * p.T' + p.R * p.Q * p.R'
        end

        # Store state for next period
        a_store[:, t + 1] = a
        P_store[:, :, t + 1] = P
    end

    const_term = -obs_dim * n_obs * log(ET(2π)) / 2

    return (loglik = loglik + const_term,
            a = a_store,
            P = P_store,
            v = v_store,
            F = F_store,
            K = K_store,
            missing_mask = missing_mask)
end

"""
    kalman_filter_full_scalar(Z, H, T, R, Q, a1, P1, y) -> NamedTuple

Scalar version of full Kalman filter for univariate state-space models.
Handles missing observations marked as NaN.

For `n` observations, returns `n+1` states:
- `a[1]` = initial state
- `a[t+1]` = E[αₜ₊₁ | y₁:ₜ] for t = 1, ..., n
"""
function kalman_filter_full_scalar(Z::Real, H::Real, Tmat::Real, R::Real, Q::Real,
                                   a1::Real, P1::Real, y::AbstractVector)
    n = length(y)
    ET = promote_type(typeof(Z), typeof(H), typeof(Tmat), typeof(R),
                      typeof(Q), typeof(a1), typeof(P1))

    # State storage: n+1 entries
    a_store = Vector{ET}(undef, n + 1)
    P_store = Vector{ET}(undef, n + 1)

    # Innovation storage: n entries
    v_store = Vector{ET}(undef, n)
    F_store = Vector{ET}(undef, n)
    missing_mask = BitVector(undef, n)

    a = convert(ET, a1)
    P = convert(ET, P1)
    a_store[1] = a
    P_store[1] = P

    loglik = zero(ET)
    n_obs = 0

    for t in 1:n
        y_t = y[t]

        if _has_missing(y_t)
            missing_mask[t] = true
            v_store[t] = ET(NaN)
            F_store[t] = Z * P * Z + H

            # Propagate without update
            a = Tmat * a
            P = Tmat * P * Tmat + R * Q * R
        else
            missing_mask[t] = false
            n_obs += 1

            v = y_t - Z * a
            F = Z * P * Z + H
            Finv = one(ET) / F

            loglik += -ET(0.5) * (log(F) + v^2 * Finv)

            v_store[t] = v
            F_store[t] = F

            # Update and predict
            K = Tmat * P * Z * Finv
            a_filt = a + P * Z * Finv * v
            P_filt = P - P * Z * Finv * Z * P

            a = Tmat * a_filt
            P = Tmat * P_filt * Tmat + R * Q * R
        end

        a_store[t + 1] = a
        P_store[t + 1] = P
    end

    const_term = -n_obs * log(ET(2π)) / 2

    return (loglik = loglik + const_term,
            a = a_store,
            P = P_store,
            v = v_store,
            F = F_store,
            missing_mask = missing_mask)
end
