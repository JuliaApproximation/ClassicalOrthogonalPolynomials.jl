using Test, ClassicalOrthogonalPolynomials, BandedMatrices, LinearAlgebra, LazyArrays, ContinuumArrays, LazyBandedMatrices
import ClassicalOrthogonalPolynomials: symmjacobim, CholeskyJacobiBands
import LazyArrays: AbstractCachedMatrix
import LazyBandedMatrices: SymTridiagonal

@testset "Basic properties" begin
    @testset "Test the Q&D conversion to BandedMatrix format" begin
        # Legendre
        P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
        x = axes(P,1)
        J = jacobimatrix(P)
        Jx = symmjacobim(J)
        @test J[1:100,1:100] == Jx[1:100,1:100]
        # Jacobi
        P = Normalized(Jacobi(1,2)[affine(0..1,Inclusion(-1..1)),:])
        x = axes(P,1)
        J = jacobimatrix(P)
        Jx = symmjacobim(J)
        @test J[1:100,1:100] == Jx[1:100,1:100]
    end

    @testset "Basic types and getindex variants" begin
        # basis
        P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
        x = axes(P,1)
        J = jacobimatrix(P)
        Jx = symmjacobim(J)
        # example weight
        w = (I - Jx^2)
        # banded cholesky for symmetric-tagged W
        @test cholesky(w).U isa UpperTriangular
        # compute Jacobi matrix via cholesky
        Jchol = cholesky_jacobimatrix(w)
        @test Jchol isa LazyBandedMatrices.SymTridiagonal
        # CholeskyJacobiBands object
        Cbands = CholeskyJacobiBands(w)
        @test Cbands isa CholeskyJacobiBands
        @test Cbands isa AbstractCachedMatrix
        @test getindex(Cbands,1,100) == getindex(Cbands,1,1:100)[100]
    end
end

@testset "Comparison with Lanczos and Classical" begin
    @testset "Not using Clenshaw" begin
        @testset "w(x) = x^2*(1-x)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            Jx = symmjacobim(J)
            w = (Jx^2 - Jx^3)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(w)
            # compute Jacobi matrix via classical recurrence
            Q = Normalized(Jacobi(1,2)[affine(0..1,Inclusion(-1..1)),:])
            Jclass = jacobimatrix(Q)
            # compute Jacobi matrix via Lanczos
            wf = x.^2 .* (1 .- x)
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
            # Comparison with Classical
            @test Jchol[1:500,1:500] ≈ Jclass[1:500,1:500]
        end

        @testset "w(x) = (1-x^2)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            Jx = symmjacobim(J)
            w = (I - Jx^2)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(w)
            # compute Jacobi matrix via Lanczos
            wf = (1 .- x.^2)
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end

        @testset "w(x) = (1-x^4)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            Jx = symmjacobim(J)
            w = (I - Jx^4)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(w)
            # compute Jacobi matrix via Lanczos
            wf = (1 .- x.^4)
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end

        @testset "w(x) = (1.014-x^3)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            Jx = symmjacobim(J)
            t = 1.014
            w = (t*I - Jx^3)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(w)
            # compute Jacobi matrix via Lanczos
            wf = (t .- x.^3)
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
    end

    @testset "Using Clenshaw for polynomial weights" begin
        @testset "w(x) = x^2*(1-x)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = x^2*(1-x)
            # compute Jacobi matrix via cholesky
            W = P \ (wf.(x) .* P)
            Jchol = cholesky_jacobimatrix(Symmetric(W))
            # compute Jacobi matrix via classical recurrence
            Q = Normalized(Jacobi(1,2)[affine(0..1,Inclusion(-1..1)),:])
            Jclass = jacobimatrix(Q)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
            # Comparison with Classical
            @test Jchol[1:500,1:500] ≈ Jclass[1:500,1:500]
        end

        @testset "w(x) = (1-x^2)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = (1-x^2)
            # compute Jacobi matrix via cholesky
            W = P \ (wf.(x) .* P)
            Jchol = cholesky_jacobimatrix(Symmetric(W))
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end

        @testset "w(x) = (1-x^4)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = (1-x^4)
            # compute Jacobi matrix via cholesky
            W = P \ (wf.(x) .* P)
            Jchol = cholesky_jacobimatrix(Symmetric(W))
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end

        @testset "w(x) = (1.014-x^3)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = 1.014-x^4
            # compute Jacobi matrix via cholesky
            W = P \ (wf.(x) .* P)
            Jchol = cholesky_jacobimatrix(Symmetric(W))
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
    end

    @testset "Using Clenshaw with exponential weights" begin
        @testset "w(x) = exp(x)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = exp(x)
            # compute Jacobi matrix via cholesky
            W = P \ (wf.(x) .* P)
            Jchol = cholesky_jacobimatrix(Symmetric(W))
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
        
        @testset "w(x) = (1-x)*exp(x)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = (1-x)*exp(x)
            # compute Jacobi matrix via cholesky
            W = P \ (wf.(x) .* P)
            Jchol = cholesky_jacobimatrix(Symmetric(W))
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
        
        @testset "w(x) = (1-x^2)*exp(x^2)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = (1-x^2)*exp(x^2)
            # compute Jacobi matrix via cholesky
            W = P \ (wf.(x) .* P)
            Jchol = cholesky_jacobimatrix(Symmetric(W))
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
        
        @testset "w(x) = x*(1-x^2)*exp(-x^2)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = x*(1-x^2)*exp(-x^2)
            # compute Jacobi matrix via cholesky
            W = P \ (wf.(x) .* P)
            Jchol = cholesky_jacobimatrix(Symmetric(W))
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
    end
end
