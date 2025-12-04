"""
    smoother_ad.jl

AD-compatible Rauch-Tung-Striebel (RTS) smoother for state-space models.

Pure functional implementation that works with automatic differentiation.
"""

"""
    kalman_smoother(Z, T, a_filtered, P_filtered, v, F)

AD-compatible RTS smoother. Takes filtered outputs and returns smoothed states.

# Arguments
- `Z`: Observation matrix (p × m)
- `T`: Transition matrix (m × m)
- `a_filtered`: Filtered states (m × n+1), where a_filtered[:, t] = a_t = E[α_t | y_{1:t-1}]
- `P_filtered`: Filtered covariances (m × m × n+1)
- `v`: Innovations (p × n), where v[:, t] = y_t - Z * a_t
- `F`: Innovation covariances (p × p × n)

# Returns
- `alpha_smooth`: Smoothed states (m × n), where alpha_smooth[:, t] = E[α_t | y_{1:n}]
- `V_smooth`: Smoothed covariances (m × m × n)

# Notes
This follows Durbin & Koopman (2012), Chapter 4, equations (4.32)-(4.44).
The convention is:
- a_filtered[:, 1] is the initial state a_1 (before any observations)
- a_filtered[:, t] for t > 1 is a_t (prediction after y_{1:t-1})
- v[:, t] and F[:, :, t] correspond to observation y_t
"""
function kalman_smoother(Z::AbstractMatrix, T::AbstractMatrix,
                         a_filtered::AbstractMatrix, P_filtered::AbstractArray,
                         v::AbstractMatrix, F::AbstractArray)

    state_dim = size(T, 1)
    n_obs = size(v, 2)

    # Determine element type for AD compatibility
    ET = promote_type(eltype(Z), eltype(T), eltype(a_filtered),
                      eltype(P_filtered), eltype(v), eltype(F))

    # Allocate output arrays
    alpha_smooth = Matrix{ET}(undef, state_dim, n_obs)
    V_smooth = Array{ET}(undef, state_dim, state_dim, n_obs)

    # Initialize smoothing recursion: r_n = 0, N_n = 0
    r_vec = zeros(ET, state_dim)
    N_mat = zeros(ET, state_dim, state_dim)

    # Backward recursion: t = n, n-1, ..., 1
    @inbounds for t in n_obs:-1:1
        # Get filtered state and covariance at time t
        # a_filtered[:, t] = a_t = E[α_t | y_{1:t-1}]
        a_t = view(a_filtered, :, t)
        P_t = view(P_filtered, :, :, t)

        # Get innovation and its covariance
        v_t = view(v, :, t)
        F_t = view(F, :, :, t)

        # F_t^{-1}
        F_inv = inv(F_t)

        # Kalman gain: K_t = T * P_t * Z' * F_t^{-1}
        K_t = T * P_t * Z' * F_inv

        # L_t = T - K_t * Z
        L_t = T - K_t * Z

        # Compute r_{t-1} = Z' * F_t^{-1} * v_t + L_t' * r_t
        r_new = Z' * F_inv * v_t + L_t' * r_vec

        # Compute N_{t-1} = Z' * F_t^{-1} * Z + L_t' * N_t * L_t
        N_new = Z' * F_inv * Z + L_t' * N_mat * L_t

        # Smoothed state: α̂_t = a_t + P_t * r_{t-1}
        alpha_smooth[:, t] = a_t + P_t * r_new

        # Smoothed covariance: V_t = P_t - P_t * N_{t-1} * P_t
        V_smooth[:, :, t] = P_t - P_t * N_new * P_t

        # Update for next iteration
        r_vec = r_new
        N_mat = N_new
    end

    return alpha_smooth, V_smooth
end

"""
    kalman_smoother(p::KFParms, y, a1, P1) -> NamedTuple

High-level Kalman smoother that runs filter and smoother in one call.

# Arguments
- `p::KFParms`: State-space model parameters (Z, H, T, R, Q)
- `y`: Observations (p × n matrix)
- `a1`: Initial state mean (m-vector)
- `P1`: Initial state covariance (m × m matrix)

# Returns
Named tuple with:
- `α`: Smoothed states (m × n), where α[:, t] = E[αₜ | y₁:ₙ]
- `V`: Smoothed state covariances (m × m × n)
- `r`: Smoothing cumulant (m × n)
- `N`: Smoothing variance cumulant (m × m × n)

# Example
```julia
p = KFParms(Z, H, T, R, Q)
result = kalman_smoother(p, y, a1, P1)
smoothed_states = result.α
```
"""
function kalman_smoother(p::KFParms, y::AbstractMatrix,
                         a1::AbstractVector, P1::AbstractMatrix)
    # Run filter first
    filt = kalman_filter_full(p, y, a1, P1)

    # Run smoother
    α, V = kalman_smoother(p.Z, p.T, filt.a, filt.P, filt.v, filt.F)

    # Return full results
    return (α=α, V=V, r=nothing, N=nothing)
end

"""
    kalman_filter_and_smooth(p::KFParms, y, a1, P1)

Combined filter and smoother for state-space models.

Returns filtered states, smoothed states, and log-likelihood.
AD-compatible.

# Arguments
- `p::KFParms`: State-space model parameters (Z, H, T, R, Q)
- `y`: Observations (p × n matrix)
- `a1`: Initial state mean (m-vector)
- `P1`: Initial state covariance (m × m matrix)

# Returns
Named tuple with:
- `loglik`: Log-likelihood
- `a_filtered`: Filtered states (m × n+1)
- `P_filtered`: Filtered covariances (m × m × n+1)
- `alpha_smooth`: Smoothed states (m × n)
- `V_smooth`: Smoothed covariances (m × m × n)
"""
function kalman_filter_and_smooth(p::KFParms, y::AbstractMatrix,
                                  a1::AbstractVector, P1::AbstractMatrix)
    # Run filter
    loglik, a_filt, P_filt, v, F = kalman_filter_full(p, y, a1, P1)

    # Run smoother
    alpha_smooth, V_smooth = kalman_smoother(p.Z, p.T, a_filt, P_filt, v, F)

    return (loglik=loglik, a_filtered=a_filt, P_filtered=P_filt,
            alpha_smooth=alpha_smooth, V_smooth=V_smooth)
end

"""
    kalman_smoother_scalar(Z, T, a_filtered, P_filtered, v, F)

Scalar version of RTS smoother for univariate state-space models.
"""
function kalman_smoother_scalar(Z::Real, T::Real,
                                a_filtered::AbstractVector, P_filtered::AbstractVector,
                                v::AbstractVector, F::AbstractVector)

    n_obs = length(v)
    ET = promote_type(typeof(Z), typeof(T), eltype(a_filtered),
                      eltype(P_filtered), eltype(v), eltype(F))

    alpha_smooth = Vector{ET}(undef, n_obs)
    V_smooth = Vector{ET}(undef, n_obs)

    # Initialize: r_n = 0, N_n = 0
    r = zero(ET)
    N = zero(ET)

    @inbounds for t in n_obs:-1:1
        a_t = a_filtered[t]
        P_t = P_filtered[t]
        v_t = v[t]
        F_inv = one(ET) / F[t]

        # Kalman gain and L
        K_t = T * P_t * Z * F_inv
        L_t = T - K_t * Z

        # Recursion
        r_new = Z * F_inv * v_t + L_t * r
        N_new = Z * F_inv * Z + L_t * N * L_t

        # Smoothed estimates
        alpha_smooth[t] = a_t + P_t * r_new
        V_smooth[t] = P_t - P_t * N_new * P_t

        r = r_new
        N = N_new
    end

    return alpha_smooth, V_smooth
end
