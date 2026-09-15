using ClassicalOrthogonalPolynomials, QuasiArrays, Random, StatsBase, Test

Random.seed!(5)

@testset "roots" begin
    T = Chebyshev()
    P = Legendre()
    f = x -> cos(10x)
    @test findall(iszero, expand(T, f)) ≈ findall(iszero, expand(P, f)) ≈ [k*π/20 for k=-5:2:5]

    g = x -> x + 0.001cos(x)
    @test searchsortedfirst(expand(T, g), 0.1) ≈ searchsortedfirst(expand(P, g), 0.1) ≈ findall(iszero, expand(T, x -> g(x)-0.1))[1]
end

@testset "roots of polynomial expansions" begin
    for P in (Chebyshev(), Legendre(), Jacobi(0.2, 0.3))
        @test findall(iszero, expand(P, x -> x - 1)) ≈ [1.0]
        @test findall(iszero, expand(P, x -> x + 1)) ≈ [-1.0]
        @test findall(iszero, expand(P, x -> x^2 - 1)) ≈ [-1.0, 1.0]
    end

    T01 = chebyshevt(0..1)
    @test findall(iszero, expand(T01, x -> x)) ≈ [0.0]
    @test findall(iszero, expand(T01, x -> x - 1)) ≈ [1.0]
    @test findall(iszero, expand(T01, x -> (x - 1) * (x - 0.25))) ≈ [0.25, 1.0]
end

@testset "high-degree roots" begin
    r = findall(iszero, expand(Chebyshev(), x -> cos(200x)))
    @test r ≈ [(2k + 1) * π / 400 for k in -64:63 if -1 ≤ (2k + 1) * π / 400 ≤ 1]
end

@testset "sample" begin
    f = expand(Chebyshev(), exp)
    @test sum(sample(f, 100_000))/100_000 ≈ 0.31 atol=1E-2
end

@testset "det point sampling" begin
    P = Normalized(Legendre()); x = axes(P,1)
    A =  cos.((0:5)' .* x)
    @test (Legendre() \ A)[1:5,1] ≈ [1; zeros(4)] # test transform bug
    @test (P \ A)[1:5,1] ≈ [sqrt(2); zeros(4)]
    Q,R = qr(P \ A)
    @test (P * (Q*R))[0.1,:] ≈ A[0.1,:]
    @test (A*inv(R))[0.1,:] ≈ QuasiArrays.ApplyQuasiArray(*, P, Q)[0.1,1:6] 

    Q,R = qr(A)
    @test Q[0.1,:]'R ≈ A[0.1,:]'

    @test abs(sum(sample(sum(expand(Q[:,k] .^2) for k=axes(Q,2)), 1000))) ≤ 100 # mean is (numerically) zero
end

@testset "minimum/maximum/extrema (#242)" begin
    f = expand(ChebyshevT(), x -> exp(x) * cos(100x.^2))
    @test minimum(f) ≈ -2.682833127491678
    @test maximum(f) ≈ 2.6401248792053362
    @test extrema(f) == (minimum(f), maximum(f))

    f = expand(ChebyshevT(), exp)
    @test minimum(f) ≈ 1/ℯ
    @test maximum(f) ≈ ℯ
    @test extrema(f) == (minimum(f), maximum(f))
end