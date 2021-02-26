using ClassicalOrthogonalPolynomials, Test
import ClassicalOrthogonalPolynomials: OrthogonalPolynomialRatio, recurrencecoefficients

@testset "OrthogonalPolynomialRatio" begin
    P = Legendre()
    R = OrthogonalPolynomialRatio(P,0.1)
    @test P[0.1,1:10] ./ P[0.1,2:11] ≈ R[1:10]
    R = OrthogonalPolynomialRatio(P,-1)
    @test R[1:10] ≈ fill(-1,10)
end