using ClassicalOrthogonalPolynomials, Random, Test
using ClassicalOrthogonalPolynomials: sample

Random.seed!(5)

@testset "roots" begin
    T = Chebyshev()
    P = Legendre()
    f = x -> cos(10x)
    @test findall(iszero, expand(T, f)) ≈ findall(iszero, expand(P, f)) ≈ [k*π/20 for k=-5:2:5]

    g = x -> x + 0.001cos(x)
    @test searchsortedfirst(expand(T, g), 0.1) ≈ searchsortedfirst(expand(P, g), 0.1) ≈ findall(iszero, expand(T, x -> g(x)-0.1))[1]
end

@testset "sample" begin
    f = expand(Chebyshev(), exp)
    @test sum(sample(f, 100_000))/100_000 ≈ 0.31 atol=1E-2
end

@testset "det point sampling" begin
    P = Normalized(Legendre()); x = axes(P,1)
    A =  [(1 .+ x) (1 .+ x.^3)]
    Q,R = qr(P \ A)
end

@testset "minimum/maximum/extrema (#242)" begin
    f = expand(ChebyshevT(), x -> exp(x) * cos(100x.^2))
    @test minimum(f) ≈ -2.682833127491678
    @test maximum(f) ≈ 2.6401248792053362
    @test extrema(f) == (minimum(f), maximum(f))
end