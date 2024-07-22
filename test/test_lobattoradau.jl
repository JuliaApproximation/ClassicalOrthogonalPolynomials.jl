using ClassicalOrthogonalPolynomials, FastGaussQuadrature
const COP = ClassicalOrthogonalPolynomials
const FGQ = FastGaussQuadrature
using Test
using ClassicalOrthogonalPolynomials: symtridiagonalize
using LinearAlgebra

@testset "gaussradau" begin
    @testset "Compare with FastGaussQuadrature" begin
        x1, w1 = COP.gaussradau(Legendre(), 5, -1.0)
        x2, w2 = FGQ.gaussradau(6)
        @test x1 ≈ x2 && w1 ≈ w2
        @test x1[1] == -1

        x1, w1 = COP.gaussradau(Jacobi(1.0, 3.5), 25, -1.0)
        x2, w2 = FGQ.gaussradau(26, 1.0, 3.5)
        @test x1 ≈ x2 && w1 ≈ w2
        @test x1[1] == -1

        I0, I1 = COP.ChebyshevInterval(), COP.UnitInterval()
        P = Jacobi(2.0, 0.0)[COP.affine(I1, I0), :]
        x1, w1 = COP.gaussradau(P, 18, 0.0)
        x2, w2 = FGQ.gaussradau(19, 2.0, 0.0)
        @test 2x1 .- 1 ≈ x2 && 2w1 ≈ w2

        x1, w1 = COP.gaussradau(Jacobi(1 / 2, 1 / 2), 4, 1.0)
        x2, w2 = FGQ.gaussradau(5, 1 / 2, 1 / 2)
        @test sort(-x1) ≈ x2
        @test_broken w1 ≈ w2 # What happens to the weights when inverting the interval?
    end

    @testset "Example 3.5 in Gautschi (2004)'s book" begin
        P = Laguerre(3.0)
        n = 5
        J = symtridiagonalize(jacobimatrix(P))[1:(n-1), 1:(n-1)]
        _J = zeros(n, n)
        _J[1:n-1, 1:n-1] .= J
        _J[n-1, n] = sqrt((n - 1) * (n - 1 + P.α))
        _J[n, n-1] = _J[n-1, n]
        _J[n, n] = n - 1
        x, V = eigen(_J)
        w = 6V[1, :] .^ 2
        xx, ww = COP.gaussradau(P, n - 1, 0.0)
        @test xx ≈ x && ww ≈ w
    end

    @testset "Some numerical integration" begin
        f = x -> 2x + 7x^2 + 10x^3 + exp(-x)
        x, w = COP.gaussradau(Chebyshev(), 10, -1.0)
        @test dot(f.(x), w) ≈ 14.97303754807069897 # integral of (2x + 7x^2 + 10x^3 + exp(-x))/sqrt(1-x62)
        @test x[1] == -1
        
        f = x -> -1.0 + 5x^6
        x, w = COP.gaussradau(Jacobi(-1/2, -1/2), 2, 1.0)
        @test dot(f.(x), w) ≈ 9π/16
        @test x[end] == 1 
        @test length(x) == 3
    end
end

@testset "gausslobatto" begin
    @testset "Compare with FastGaussQuadrature" begin
        x1, w1 = COP.gausslobatto(Legendre(), 5)
        x2, w2 = FGQ.gausslobatto(7)
        @test x1 ≈ x2 && w1 ≈ w2
        @test x1[1] == -1
        @test x1[end] == 1

        I0, I1 = COP.ChebyshevInterval(), COP.UnitInterval()
        P = Legendre()[COP.affine(I1, I0), :]
        x1, w1 = COP.gausslobatto(P, 18)
        x2, w2 = FGQ.gausslobatto(20)
        @test 2x1 .- 1 ≈ x2 && 2w1 ≈ w2
    end

    @testset "Some numerical integration" begin
        f = x -> 2x + 7x^2 + 10x^3 + exp(-x)
        x, w = COP.gausslobatto(Chebyshev(), 10)
        @test dot(f.(x), w) ≈ 14.97303754807069897
        @test x[1] == -1
        @test x[end] == 1 
        @test length(x) == 12
        
        f = x -> -1.0 + 5x^6
        x, w = COP.gausslobatto(Jacobi(-1/2, -1/2), 4)
        @test dot(f.(x), w) ≈ 9π/16
        @test x[1]==-1
        @test x[end] == 1 
        @test length(x) == 6
    end
end
