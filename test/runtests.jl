using Filthy
using Test
using LinearAlgebra
using DelimitedFiles
using ForwardDiff

# Load Nile data
nile = readdlm("Nile.csv", ',', Float64)
y = reshape(nile[:, 1], 1, :)  # 1 x 100 matrix

# MLE estimates from Durbin & Koopman (2012)
const Z_nile = [1.0;;]
const H_nile = [15099.0;;]
const T_nile = [1.0;;]
const R_nile = [1.0;;]
const Q_nile = [1469.1;;]
const a1_nile = [0.0]
const P1_nile = [1e7;;]

@testset "KFParms" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    @test p.Z == Z_nile
    @test p.H == H_nile
    @test p.T == T_nile
    @test p.R == R_nile
    @test p.Q == Q_nile
    @test size(p) == (1, 1, 1)
end

@testset "kalman_loglik" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    ll = kalman_loglik(p, y, a1_nile, P1_nile)

    # Log-likelihood should be finite and negative
    @test isfinite(ll)
    @test ll < 0

    # With approximate diffuse (P1=1e7), log-likelihood differs from exact diffuse
    # The value should be in a reasonable range for 100 observations
    @test ll > -700  # Not too small
    @test ll < -600  # Not too large
end

@testset "kalman_loglik AD" begin
    function ll_wrapper(theta)
        H = [theta[1];;]
        Q = [theta[2];;]
        p = KFParms(Z_nile, H, T_nile, R_nile, Q)
        return kalman_loglik(p, y, a1_nile, P1_nile)
    end

    theta = [15099.0, 1469.1]
    g = ForwardDiff.gradient(ll_wrapper, theta)

    @test length(g) == 2
    @test all(isfinite.(g))

    # Gradients should be near zero at MLE
    @test all(abs.(g) .< 1.0)
end

@testset "kalman_loglik_scalar" begin
    ll = kalman_loglik_scalar(1.0, 15099.0, 1.0, 1.0, 1469.1, 0.0, 1e7, vec(nile))

    @test isfinite(ll)
    @test ll < 0

    # Should match matrix version
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    ll_mat = kalman_loglik(p, y, a1_nile, P1_nile)
    @test isapprox(ll, ll_mat, rtol=1e-10)
end

@testset "kalman_filter_full" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter_full(p, y, a1_nile, P1_nile)

    # Check struct type
    @test result isa Filthy.KalmanFilterResult

    # Check fields exist via hasproperty
    @test hasproperty(result, :loglik)
    @test hasproperty(result, :at)
    @test hasproperty(result, :Pt)
    @test hasproperty(result, :att)
    @test hasproperty(result, :Ptt)
    @test hasproperty(result, :vt)
    @test hasproperty(result, :Ft)
    @test hasproperty(result, :Kt)

    n = size(y, 2)
    # States have n entries (one per time point)
    @test size(result.at) == (1, n)
    @test size(result.Pt) == (1, 1, n)
    @test size(result.att) == (1, n)
    @test size(result.Ptt) == (1, 1, n)
    # Innovations have n entries (one per observation)
    @test size(result.vt) == (1, n)
    @test size(result.Ft) == (1, 1, n)

    # First predicted state is initial state
    @test result.at[1, 1] == a1_nile[1]
    @test result.Pt[1, 1, 1] == P1_nile[1, 1]

    # Log-likelihood should match kalman_loglik
    ll_direct = kalman_loglik(p, y, a1_nile, P1_nile)
    @test isapprox(result.loglik, ll_direct, rtol=1e-10)

    # Test accessor methods
    @test Filthy.predicted_states(result) === result.at
    @test Filthy.filtered_states(result) === result.att
    @test Filthy.loglikelihood(result) == result.loglik
end

@testset "kalman_smoother" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter_full(p, y, a1_nile, P1_nile)

    alpha_smooth, V_smooth = kalman_smoother(Z_nile, T_nile,
                                              result.at, result.Pt,
                                              result.vt, result.Ft)

    n = size(y, 2)
    @test size(alpha_smooth) == (1, n)
    @test size(V_smooth) == (1, 1, n)

    # Smoothed states should be finite
    @test all(isfinite.(alpha_smooth))

    # Smoothed variances should be positive
    @test all(V_smooth .> 0)

    # Smoothed variance should be smaller than predicted variance
    for t in 2:n-1
        @test V_smooth[1, 1, t] <= result.Pt[1, 1, t]
    end

    # Reference value at t=100 from D&K Table 2.1
    @test isapprox(alpha_smooth[1, 100], 798.4, rtol=0.01)
end

@testset "kalman_smoother AD" begin
    function smooth_sum(theta)
        H = [theta[1];;]
        Q = [theta[2];;]
        p = KFParms(Z_nile, H, T_nile, R_nile, Q)
        result = kalman_filter_full(p, y, a1_nile, P1_nile)
        alpha, V = kalman_smoother(Z_nile, T_nile, result.at, result.Pt, result.vt, result.Ft)
        return sum(alpha)
    end

    theta = [15099.0, 1469.1]
    g = ForwardDiff.gradient(smooth_sum, theta)

    @test length(g) == 2
    @test all(isfinite.(g))
end

@testset "kalman_filter_and_smooth" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter_and_smooth(p, y, a1_nile, P1_nile)

    @test haskey(result, :loglik)
    @test haskey(result, :a_filtered)
    @test haskey(result, :P_filtered)
    @test haskey(result, :alpha_smooth)
    @test haskey(result, :V_smooth)

    # Compare with separate calls
    ll = kalman_loglik(p, y, a1_nile, P1_nile)
    @test isapprox(result.loglik, ll, rtol=1e-10)
end

@testset "kalman_smoother_scalar" begin
    result_scalar = kalman_filter_full_scalar(
        1.0, 15099.0, 1.0, 1.0, 1469.1, 0.0, 1e7, vec(nile))

    # Check struct type
    @test result_scalar isa Filthy.KalmanFilterResultScalar

    alpha, V = kalman_smoother_scalar(1.0, 1.0, result_scalar.at, result_scalar.Pt,
                                       result_scalar.vt, result_scalar.Ft)

    n = length(nile)
    @test length(alpha) == n
    @test length(V) == n

    # Should match matrix version
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter_full(p, y, a1_nile, P1_nile)
    alpha_mat, V_mat = kalman_smoother(Z_nile, T_nile, result.at, result.Pt, result.vt, result.Ft)

    @test isapprox(alpha, vec(alpha_mat), rtol=1e-10)
end

# DSL tests
include("dsl_tests.jl")

# EM algorithm tests (uses TestItems)
include("em_tests.jl")

# Generalized EM tests (MARSS validation - diagonal covariances)
include("em_generalized_tests.jl")

# General EM tests (MARSS validation - full Z and T estimation)
include("em_general_tests.jl")

# FKF validation tests (R FKF package comparison)
include("fkf_validation_tests.jl")
