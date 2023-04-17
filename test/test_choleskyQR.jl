using Test, ClassicalOrthogonalPolynomials, BandedMatrices, LinearAlgebra, LazyArrays, ContinuumArrays, LazyBandedMatrices, InfiniteLinearAlgebra
import ClassicalOrthogonalPolynomials: CholeskyJacobiBands, cholesky_jacobimatrix, qr_jacobimatrix, QRJacobiBands
import LazyArrays: AbstractCachedMatrix

@testset "Cholesky Jacobi Matrix - Basic properties" begin
    # basis
    P = Normalized(legendre(0..1))
    x = axes(P,1)
    J = jacobimatrix(P)
    # example weight
    w(x) = (1 - x^2)
    W = Symmetric(P \ (w.(x) .* P))
    # banded cholesky for symmetric-tagged W
    @test cholesky(W).U isa UpperTriangular
    # compute Jacobi matrix via cholesky
    Jchol = cholesky_jacobimatrix(w,P)
    Jcholalt = cholesky_jacobimatrix(W,P)
    @test Jchol[1:200,1:200] ≈ Jcholalt[1:200,1:200]
    # CholeskyJacobiBands object
    Cbands = CholeskyJacobiBands(W,P)
    @test Cbands isa CholeskyJacobiBands
    @test Cbands isa AbstractCachedMatrix
    @test getindex(Cbands,1,100) == getindex(Cbands,1,1:100)[100]
    @test getindex(Cbands,2,2) == getindex(Cbands,1:2,1:2)[2,2]
end

@testset "Comparison with Lanczos and Classical" begin
    @testset "Using Clenshaw for polynomial weights" begin
        @testset "w(x) = x^2*(1-x)" begin
            P = Normalized(legendre(0..1))
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = x^2*(1-x)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(wf, P)
            # compute Jacobi matrix via classical recurrence
            Q = Normalized(Jacobi(1,2)[affine(0..1,Inclusion(-1..1)),:])
            Jclass = jacobimatrix(Q)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
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
            Jchol = cholesky_jacobimatrix(wf, P)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end

        @testset "w(x) = (1-x^4)" begin
            P = Normalized(legendre(0..1))
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = (1-x^4)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(wf, P)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end

        @testset "w(x) = (1.014-x^3)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = 1.014-x^4
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(wf, P)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
    end

    @testset "Using Cholesky with exponential weights" begin
        @testset "w(x) = exp(x)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = exp(x)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(wf, P)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
        
        @testset "w(x) = (1-x)*exp(x)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = (1-x)*exp(x)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(wf, P)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
        
        @testset "w(x) = (1-x^2)*exp(x^2)" begin
            P = Normalized(legendre(0..1))
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = (1-x)^2*exp(x^2)
            sqrtwf(x) = (1-x)*exp(x)
            # compute Jacobi matrix via decomp
            Jchol = cholesky_jacobimatrix(wf, P)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
        
        @testset "w(x) = x*(1-x^2)*exp(-x^2)" begin
            P = Normalized(Legendre()[affine(0..1,Inclusion(-1..1)),:])
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = x*(1-x^2)*exp(-x^2)
            # compute Jacobi matrix via cholesky
            Jchol = cholesky_jacobimatrix(wf, P)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
    end
end

@testset "QR version" begin
    @testset "basic properties" begin
        # basis
        P = Normalized(legendre(0..1))
        x = axes(P,1)
        J = jacobimatrix(P)
        # example weight
        sqrtw(x) = (1 - x)
        sqrtW = Symmetric(P \ (sqrtw.(x) .* P))
        # bands test
        QRbands = QRJacobiBands(sqrtW,P)
        @test QRbands isa QRJacobiBands
        @test QRbands isa AbstractCachedMatrix
        @test getindex(QRbands,1,100) == getindex(QRbands,1,1:100)[100]
        @test getindex(QRbands,2,2) == getindex(QRbands,1:2,1:2)[2,2]
    end

    @testset "QR case, w(x) = (1-x)^2" begin
        P = Normalized(legendre(0..1))
        x = axes(P,1)
        J = jacobimatrix(P)
        wf(x) = (1-x)^2
        sqrtwf(x) = (1-x)
        # compute Jacobi matrix via decomp
        Jchol = cholesky_jacobimatrix(wf, P)
        Jqr = qr_jacobimatrix(sqrtwf, P)
        # use alternative inputs
        sqrtW = (P \ (sqrtwf.(x) .* P))
        Jqralt = qr_jacobimatrix(sqrtW, P, false)
        # compute Jacobi matrix via Lanczos
        Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
        # Comparison with Lanczos
        @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
        @test Jqr[1:500,1:500] ≈ Jlanc[1:500,1:500]
        @test Jqralt[1:500,1:500] ≈ Jlanc[1:500,1:500]
    end
end
