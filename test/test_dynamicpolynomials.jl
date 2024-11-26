using DynamicPolynomials, ClassicalOrthogonalPolynomials, Test

@polyvar x

@testset "DynamicPolynomials" begin
    @test chebyshevt(0,x) == 1
    @test chebyshevt(1,x) == x
    @test chebyshevt(5,x) == 5x - 20x^3 + 16x^5
    @test chebyshevu(5,x) == 6x - 32x^3 + 32x^5
    legendrep(5,x)
end