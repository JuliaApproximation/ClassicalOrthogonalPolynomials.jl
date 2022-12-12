using ClassicalOrthogonalPolynomials, Test

@testset "roots" begin
    T = Chebyshev()
    P = Legendre()
    f = x -> cos(10x)
    @test findall(iszero, expand(T, f)) ≈ findall(iszero, expand(P, f)) ≈ [k*π/20 for k=-5:2:5]

    g = x -> x + 0.001cos(x)
    @test searchsortedfirst(expand(T, g), 0.1) ≈ searchsortedfirst(expand(P, g), 0.1) ≈ findall(iszero, expand(T, x -> g(x)-0.1))[1]
end