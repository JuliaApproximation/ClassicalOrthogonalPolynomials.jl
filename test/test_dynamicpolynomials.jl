using DynamicPolynomials, ClassicalOrthogonalPolynomials, Test

@polyvar x

@testset "DynamicPolynomials" begin
    @test chebyshevt(0,x) == 1
    @test chebyshevt(1,x) == x
    @test chebyshevt(5,x) == 5x - 20x^3 + 16x^5
    @test chebyshevu(5,x) == ultrasphericalc(5,1,x) == 6x - 32x^3 + 32x^5
    @test legendrep(5,x) ≈ (15x - 70x^3 + 63x^5)/8
    @test hermiteh(5,x) == 120x - 160x^3 + 32x^5
    @test jacobip(5,1,0,x) ≈ (5 + 35x - 70x^2 - 210x^3 + 105x^4 + 231x^5)/16
    @test jacobip(5,0.1,-0.2,x) ≈ 3*(3306827 + 75392855x - 37466770x^2 - 350553810x^3 + 47705335x^4 + 314855211x^5)/128000000
end