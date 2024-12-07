using DynamicPolynomials, ClassicalOrthogonalPolynomials, Test

@polyvar x

@testset "DynamicPolynomials" begin
    @test chebyshevt(0,x) == 1
    @test chebyshevt(1,x) == x
    @test chebyshevt(5,x) == 5x - 20x^3 + 16x^5
    @test chebyshevu(5,x) == ultrasphericalc(5,1,x) == 6x - 32x^3 + 32x^5
    @test legendrep(5,x) ≈ (15x - 70x^3 + 63x^5)/8
    @test hermiteh(5,x) == 120x - 160x^3 + 32x^5
    jacobip(5,1,0,x)
end