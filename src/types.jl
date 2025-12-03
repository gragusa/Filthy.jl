"""
    KFParms{Zt, Ht, Tt, Rt, Qt}

State-space model parameters for the Kalman filter.

# State Space Model
```
y_t = Z * α_t + ε_t,    ε_t ~ N(0, H)
α_{t+1} = T * α_t + R * η_t,    η_t ~ N(0, Q)
```

# Fields
- `Z`: Observation matrix (p × m)
- `H`: Observation noise covariance (p × p)
- `T`: State transition matrix (m × m)
- `R`: State noise selection matrix (m × r)
- `Q`: State noise covariance (r × r)
"""
struct KFParms{Zt, Ht, Tt, Rt, Qt}
    Z::Zt
    H::Ht
    T::Tt
    R::Rt
    Q::Qt
end

# Size utilities
numstates(T::AbstractMatrix) = size(T, 1)
numstates(T::Real) = 1

nummeasur(Z::AbstractMatrix) = size(Z, 1)
nummeasur(Z::Real) = 1

Base.size(p::KFParms{P}) where P<:Real = (1, 1, 1)
Base.size(p::KFParms{P}) where P<:AbstractArray = (size(p.Z, 1), size(p.T, 1), size(p.Q, 1))

# ============================================
# StaticArrays automatic conversion
# ============================================

"""
Maximum dimension for automatic conversion to StaticArrays.
Matrices with max(rows, cols) ≤ STATIC_THRESHOLD are converted.
"""
const STATIC_THRESHOLD = 13

"""
    to_static_if_small(x)

Convert array to StaticArray if dimensions are small enough (≤ STATIC_THRESHOLD).
Returns the input unchanged if dimensions exceed threshold or if already a StaticArray.

This enables automatic performance optimization for small state-space models
by using stack-allocated arrays that the compiler can unroll.
"""
to_static_if_small(x::Real) = x

function to_static_if_small(x::AbstractVector)
    n = length(x)
    n ≤ STATIC_THRESHOLD ? SVector{n}(x) : x
end

function to_static_if_small(x::AbstractMatrix)
    m, n = size(x)
    (m ≤ STATIC_THRESHOLD && n ≤ STATIC_THRESHOLD) ? SMatrix{m,n}(x) : x
end

# Already static - pass through unchanged
to_static_if_small(x::StaticArray) = x

"""
    KFParms_static(Z, H, T, R, Q)

Create KFParms with automatic StaticArrays conversion for small matrices.

Equivalent to `KFParms(to_static_if_small(Z), ...)` for all parameters.
Use this for potential performance gains when state/observation dimensions are small (≤ 13).

# Example
```julia
# These become SMatrix automatically
p = KFParms_static([1.0;;], [1.0;;], [1.0;;], [1.0;;], [1.0;;])
typeof(p.Z)  # SMatrix{1, 1, Float64, 1}

# Large matrices stay as Matrix
p = KFParms_static(rand(20,20), rand(20,20), rand(20,20), rand(20,20), rand(20,20))
typeof(p.Z)  # Matrix{Float64}
```
"""
function KFParms_static(Z, H, T, R, Q)
    KFParms(
        to_static_if_small(Z),
        to_static_if_small(H),
        to_static_if_small(T),
        to_static_if_small(R),
        to_static_if_small(Q)
    )
end
