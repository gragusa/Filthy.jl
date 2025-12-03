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

    @test haskey(result, :loglik)
    @test haskey(result, :a)
    @test haskey(result, :P)
    @test haskey(result, :v)
    @test haskey(result, :F)
    @test haskey(result, :K)

    n = size(y, 2)
    # States have n+1 entries: initial + after each observation
    @test size(result.a) == (1, n + 1)
    @test size(result.P) == (1, 1, n + 1)
    # Innovations have n entries (one per observation)
    @test size(result.v) == (1, n)
    @test size(result.F) == (1, 1, n)

    # First state is initial state
    @test result.a[1, 1] == a1_nile[1]
    @test result.P[1, 1, 1] == P1_nile[1, 1]

    # Log-likelihood should match kalman_loglik
    ll_direct = kalman_loglik(p, y, a1_nile, P1_nile)
    @test isapprox(result.loglik, ll_direct, rtol=1e-10)
end

@testset "kalman_smoother" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter_full(p, y, a1_nile, P1_nile)

    alpha_smooth, V_smooth = kalman_smoother(Z_nile, T_nile,
                                              result.a, result.P,
                                              result.v, result.F)

    n = size(y, 2)
    @test size(alpha_smooth) == (1, n)
    @test size(V_smooth) == (1, 1, n)

    # Smoothed states should be finite
    @test all(isfinite.(alpha_smooth))

    # Smoothed variances should be positive
    @test all(V_smooth .> 0)

    # Smoothed variance should be smaller than filtered variance
    for t in 2:n-1
        @test V_smooth[1, 1, t] <= result.P[1, 1, t]
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
        alpha, V = kalman_smoother(Z_nile, T_nile, result.a, result.P, result.v, result.F)
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
    ll, a_pred, P_pred, v, F = kalman_filter_full_scalar(
        1.0, 15099.0, 1.0, 1.0, 1469.1, 0.0, 1e7, vec(nile))

    alpha, V = kalman_smoother_scalar(1.0, 1.0, a_pred, P_pred, v, F)

    n = length(nile)
    @test length(alpha) == n
    @test length(V) == n

    # Should match matrix version
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter_full(p, y, a1_nile, P1_nile)
    alpha_mat, V_mat = kalman_smoother(Z_nile, T_nile, result.a, result.P, result.v, result.F)

    @test isapprox(alpha, vec(alpha_mat), rtol=1e-10)
end

# DSL tests
include("dsl_tests.jl")
