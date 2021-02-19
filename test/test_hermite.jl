using ClassicalOrthogonalPolynomials, ContinuumArrays, FillArrays, Test
import ClassicalOrthogonalPolynomials: jacobimatrix, oneto
import DomainSets: ℝ

@testset "Hermite" begin
    H = Hermite()
    w = HermiteWeight()
    
    @test axes(H) == (Inclusion(ℝ), oneto(∞))
    x = axes(H,1)
    @test H[0.1,1:4] ≈ [1,2*0.1,4*0.1^2-2,8*0.1^3-12*0.1]
    @test w[0.1] ≈ exp(-0.1^2)

    X = jacobimatrix(H)
    @test 0.1 * H[0.1,1:10]' ≈ H[0.1,1:11]'*X[1:11,1:10]

    @test (H'*(w .* H))[1,1] ≈ sqrt(π)
    @test (H'*(w .* H))[2,2] ≈ 2sqrt(π)
    @test (H'*(w .* H))[3,3] ≈ 8sqrt(π)

    D = Derivative(x)
    @test (D*H)[0.1,1:4] ≈ [0,2,8*0.1,24*0.1^2-12]
end