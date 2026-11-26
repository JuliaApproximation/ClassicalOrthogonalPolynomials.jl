using ClassicalOrthogonalPolynomials, ContinuumArrays, BlockArrays, LinearAlgebra, Test
import ContinuumArrays: MulPlan, ApplyPlan
import ClassicalOrthogonalPolynomials: JacobiITransformPlan, ShuffledIR2HC, ShuffledIFFT, ChebyshevInterval

@testset "itransform" begin
    x = Inclusion(ChebyshevInterval())
    @testset "$P" for P in (ChebyshevT(), ChebyshevU(), Legendre(), Jacobi(0.1,0.2),
                            Ultraspherical(0.3), Ultraspherical(-0.3), Ultraspherical(1.5), Ultraspherical(2),
                            Normalized(Legendre()), Normalized(Jacobi(0.1,0.2)), Normalized(ChebyshevT()),
                            Fourier(), Laurent(), ChebyshevT()[affine(0..1, x), :], Legendre()[affine(0..1, x), :])
        n = 7 # odd so that every Fourier coefficient is determined by the grid
        g = grid(P, n)
        V = P[g, 1:n]
        @test !(plan_itransform(P, n) isa MulPlan) # not the default, which evaluates the basis on the grid

        c = randn(n); c₀ = copy(c)
        @test itransform(P, c) ≈ plan_itransform(P, c) * c ≈ V * c
        @test c == c₀ # the input is not modified
        @test transform(P, itransform(P, c)) ≈ c
        @test itransform(P, transform(P, cos.(g))) ≈ cos.(g)
        @test itransform(P, [1:n;]) ≈ V * (1:n)

        C = randn(n, n); C₀ = copy(C)
        @test itransform(P, C) ≈ V * C * transpose(V)
        @test C == C₀
        @test itransform(P, C, 1) ≈ V * C
        @test itransform(P, C, 2) ≈ C * transpose(V)
        @test transform(P, itransform(P, C)) ≈ C

        # FastTransforms only supports conversions between Jacobi polynomials for vectors and matrices
        if !(P isa Union{Jacobi, Ultraspherical, Normalized{<:Any,<:Jacobi}})
            X = randn(n, 3, n); X₀ = copy(X)
            @test itransform(P, X, 3) ≈ mapslices(c -> V * c, X; dims=3)
            @test itransform(P, X, (1,3)) ≈ mapslices(C -> V * C * transpose(V), X; dims=(1,3))
            @test X == X₀
            @test transform(P, itransform(P, X, (1,3)), (1,3)) ≈ X
        end
    end

    @testset "plans" begin
        @test plan_itransform(Legendre(), 5) isa JacobiITransformPlan
        @test plan_itransform(Ultraspherical(2), 5) isa JacobiITransformPlan
        @test plan_itransform(Ultraspherical(0.3), 5) isa ApplyPlan # converts via Jacobi
        @test plan_itransform(Fourier(), 5) isa ShuffledIR2HC
        @test plan_itransform(Laurent(), 5) isa ShuffledIFFT
        @test size(plan_itransform(Legendre(), (5,6))) == (5,6)
    end

    @testset "Fourier and Laurent blocks" begin
        for F in (Fourier(), Laurent())
            C = randn(5, 5)
            CB = BlockedArray(C, (axes(F,2)[Block.(1:3)], axes(F,2)[Block.(1:3)]))
            @test plan_itransform(F, Block(3,3)) * CB ≈ plan_itransform(F, (Block(3), Block(3))) * C ≈ itransform(F, C)
            @test plan_transform(F, Block(3,3)) * (plan_itransform(F, Block(3,3)) * CB) ≈ CB
        end
    end

    @testset "Fourier with an even number of points" begin
        # sin(2θ) vanishes on the grid so its coefficient is used for cos(2θ), as in the transform
        F = Fourier()
        g = grid(F, 4)
        c = randn(4)
        @test transform(F, itransform(F, c)) ≈ c
        @test itransform(F, [c[1:3]; 0]) ≈ F[g, 1:4] * [c[1:3]; 0]
        @test itransform(F, [0, 0, 0, 1]) ≈ -cos.(2g)
    end
end
