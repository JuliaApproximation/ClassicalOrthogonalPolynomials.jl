using ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets, FillArrays, Test
import ClassicalOrthogonalPolynomials: jacobimatrix

@testset "Hermite" begin
    H = Hermite()
    @test axes(H) == (Inclusion(ℝ), Base.OneTo(∞))
    x = axes(H,1)
    X = jacobimatrix(H)

    w = HermiteWeight()
    wH = w.*H
    M = H'* ( w.*H)
    S = Diagonal(M.diag .^ (-1/2))
    Si = Diagonal(M.diag .^ (1/2))
    J = Si*X*S

    (J - im*Eye(∞)) \ [1;zeros(∞)]

    @test H[0.1,1] === 1.0 # equivalent to H_0(0.1) == 1.0
    D = Derivative(x)
    
    h = 0.000001
    @test (D*H)[0.1,1:5] ≈ (H[0.1+h,1:5] - H[0.1,1:5])/h atol=100h
end