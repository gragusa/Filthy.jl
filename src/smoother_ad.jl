"""
    smoother_ad.jl

AD-compatible Rauch-Tung-Striebel (RTS) smoother for state-space models.

Pure functional implementation that works with automatic differentiation.
"""

"""
    kalman_smoother(Z, T, at, Pt, vt, Ft)

AD-compatible RTS smoother. Takes filter outputs and returns smoothed states.

# Arguments
- `Z`: Observation matrix (p × m)
- `T`: Transition matrix (m × m)
- `at`: Predicted states (m × n), where at[:, t] = E[αₜ | y₁:ₜ₋₁]
- `Pt`: Predicted covariances (m × m × n)
- `vt`: Innovations (p × n), where vt[:, t] = yₜ - Z * at[:, t]
- `Ft`: Innovation covariances (p × p × n)

# Returns
- `alpha_smooth`: Smoothed states (m × n), where alpha_smooth[:, t] = E[αₜ | y₁:ₙ]
- `V_smooth`: Smoothed covariances (m × m × n)

# Notes
This follows Durbin & Koopman (2012), Chapter 4, equations (4.32)-(4.44).
Uses predicted states (at = a_{t|t-1}) not filtered states for the recursion.
"""
function kalman_smoother(Z::AbstractMatrix, T::AbstractMatrix,
                         at::AbstractMatrix, Pt::AbstractArray,
                         vt::AbstractMatrix, Ft::AbstractArray)

    state_dim = size(T, 1)
    n_obs = size(vt, 2)

    # Determine element type for AD compatibility
    ET = promote_type(eltype(Z), eltype(T), eltype(at),
                      eltype(Pt), eltype(vt), eltype(Ft))

    # Allocate output arrays
    alpha_smooth = Matrix{ET}(undef, state_dim, n_obs)
    V_smooth = Array{ET}(undef, state_dim, state_dim, n_obs)

    # Initialize smoothing recursion: r_n = 0, N_n = 0
    r_vec = zeros(ET, state_dim)
    N_mat = zeros(ET, state_dim, state_dim)

    # Backward recursion: t = n, n-1, ..., 1
    @inbounds for t in n_obs:-1:1
        # Get predicted state and covariance at time t
        a_t = view(at, :, t)
        P_t = view(Pt, :, :, t)

        # Get innovation and its covariance
        v_t = view(vt, :, t)
        F_t = view(Ft, :, :, t)

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
    kalman_smoother(result::KalmanFilterResult)

Run RTS smoother using filter result. Convenience method.

# Returns
- `alpha_smooth`: Smoothed states (m × n)
- `V_smooth`: Smoothed covariances (m × m × n)
"""
function kalman_smoother(result::KalmanFilterResult, Z::AbstractMatrix, T::AbstractMatrix)
    return kalman_smoother(Z, T, result.at, result.Pt, result.vt, result.Ft)
end

"""
    kalman_smoother_full(Z, T, at, Pt, vt, Ft; compute_crosscov=false)

Extended RTS smoother that optionally returns cross-lag covariances for EM algorithm.

# Arguments
- `Z`: Observation matrix (p × m)
- `T`: Transition matrix (m × m)
- `at`: Predicted states (m × n), where at[:, t] = E[αₜ | y₁:ₜ₋₁]
- `Pt`: Predicted covariances (m × m × n)
- `vt`: Innovations (p × n), where vt[:, t] = yₜ - Z * at[:, t]
- `Ft`: Innovation covariances (p × p × n)
- `compute_crosscov`: If true, compute lag-one covariances P_{t+1,t|n} (default: false)

# Returns
Named tuple with:
- `alpha`: Smoothed states (m × n), where alpha[:, t] = E[αₜ | y₁:ₙ]
- `V`: Smoothed covariances (m × m × n)
- `P_crosslag`: Cross-lag covariances P_{t+1,t|n} (m × m × (n-1)), only if compute_crosscov=true

# Cross-lag covariance formula
Using Shumway & Stoffer (2017), the lag-one covariance smoother gives:
    P_{t+1,t|n} = (I - P_{t+1|t} * N_t) * T * P_{t|t-1}
where N_t is the smoothing cumulant computed during backward recursion.

# Notes
This is used by the EM algorithm which requires E[αₜ αₜ₋₁' | y₁:ₙ] for M-step updates.
"""
function kalman_smoother_full(Z::AbstractMatrix, T::AbstractMatrix,
                               at::AbstractMatrix, Pt::AbstractArray,
                               vt::AbstractMatrix, Ft::AbstractArray;
                               compute_crosscov::Bool=false)

    state_dim = size(T, 1)
    n_obs = size(vt, 2)

    # Determine element type for AD compatibility
    ET = promote_type(eltype(Z), eltype(T), eltype(at),
                      eltype(Pt), eltype(vt), eltype(Ft))

    # Allocate output arrays
    alpha_smooth = Matrix{ET}(undef, state_dim, n_obs)
    V_smooth = Array{ET}(undef, state_dim, state_dim, n_obs)

    # Initialize smoothing recursion: r_n = 0, N_n = 0
    r_vec = zeros(ET, state_dim)
    N_mat = zeros(ET, state_dim, state_dim)

    # Backward recursion: t = n, n-1, ..., 1
    @inbounds for t in n_obs:-1:1
        # Get predicted state and covariance at time t
        a_t = view(at, :, t)
        P_t = view(Pt, :, :, t)

        # Get innovation and its covariance
        v_t = view(vt, :, t)
        F_t = view(Ft, :, :, t)

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

    # Compute cross-lag covariances if requested
    # Using Shumway & Stoffer (2017) formula:
    #   J_t = P_{t|t} * T' * inv(P_{t+1|t})
    #   P_{t+1,t|n} = V_{t+1} * J_t'
    #
    # where P_{t|t} = P_{t|t-1} - K_t * Z * P_{t|t-1} (updated covariance after obs t)
    if compute_crosscov
        P_crosslag = Array{ET}(undef, state_dim, state_dim, n_obs - 1)

        @inbounds for t in 1:(n_obs - 1)
            # Pt[:,:,t] is P_{t|t-1}
            # Pt[:,:,t+1] is P_{t+1|t}
            P_pred_t = view(Pt, :, :, t)      # P_{t|t-1}
            P_pred_tp1 = view(Pt, :, :, t+1)  # P_{t+1|t}
            F_t = view(Ft, :, :, t)

            # P_{t|t} = P_{t|t-1} - P_{t|t-1} * Z' * inv(F_t) * Z * P_{t|t-1}
            F_inv = inv(F_t)
            P_upd_t = P_pred_t - P_pred_t * Z' * F_inv * Z * P_pred_t

            # J_t = P_{t|t} * T' * inv(P_{t+1|t})
            J_t = P_upd_t * T' * inv(P_pred_tp1)

            # V_{t+1} is the smoothed covariance at t+1
            V_tp1 = view(V_smooth, :, :, t+1)

            # P_{t+1,t|n} = V_{t+1} * J_t'
            P_crosslag[:, :, t] = V_tp1 * J_t'
        end

        return (alpha=alpha_smooth, V=V_smooth, P_crosslag=P_crosslag)
    else
        return (alpha=alpha_smooth, V=V_smooth)
    end
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

    # Run smoother using predicted states (at, Pt)
    α, V = kalman_smoother(p.Z, p.T, filt.at, filt.Pt, filt.vt, filt.Ft)

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
- `a_filtered`: Filtered states (m × n) - same as att from filter result
- `P_filtered`: Filtered covariances (m × m × n) - same as Ptt from filter result
- `alpha_smooth`: Smoothed states (m × n)
- `V_smooth`: Smoothed covariances (m × m × n)
"""
function kalman_filter_and_smooth(p::KFParms, y::AbstractMatrix,
                                  a1::AbstractVector, P1::AbstractMatrix)
    # Run filter
    filt = kalman_filter_full(p, y, a1, P1)

    # Run smoother using predicted states
    alpha_smooth, V_smooth = kalman_smoother(p.Z, p.T, filt.at, filt.Pt, filt.vt, filt.Ft)

    return (loglik=filt.loglik, a_filtered=filt.att, P_filtered=filt.Ptt,
            alpha_smooth=alpha_smooth, V_smooth=V_smooth)
end

"""
    kalman_smoother_scalar(Z, T, at, Pt, vt, Ft)

Scalar version of RTS smoother for univariate state-space models.
"""
function kalman_smoother_scalar(Z::Real, T::Real,
                                at::AbstractVector, Pt::AbstractVector,
                                vt::AbstractVector, Ft::AbstractVector)

    n_obs = length(vt)
    ET = promote_type(typeof(Z), typeof(T), eltype(at),
                      eltype(Pt), eltype(vt), eltype(Ft))

    alpha_smooth = Vector{ET}(undef, n_obs)
    V_smooth = Vector{ET}(undef, n_obs)

    # Initialize: r_n = 0, N_n = 0
    r = zero(ET)
    N = zero(ET)

    @inbounds for t in n_obs:-1:1
        a_t = at[t]
        P_t = Pt[t]
        v_t = vt[t]
        F_inv = one(ET) / Ft[t]

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
