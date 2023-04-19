using Test, ClassicalOrthogonalPolynomials, BandedMatrices, LinearAlgebra, LazyArrays, ContinuumArrays, LazyBandedMatrices, InfiniteLinearAlgebra
import ClassicalOrthogonalPolynomials: cholesky_jacobimatrix, qr_jacobimatrix
import LazyArrays: AbstractCachedMatrix


@testset "CholeskyQR" begin
    @testset "Comparison of Cholesky with Lanczos and Classical" begin
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

    @testset "Comparison of QR with Lanczos" begin
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
            Jqralt = qr_jacobimatrix(sqrtW, P)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
            @test Jqr[1:500,1:500] ≈ Jlanc[1:500,1:500]
            @test Jqralt[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
    end

    @testset "ConvertedOP" begin
        P = Legendre()
        x = axes(P,1)
        Q = OrthogonalPolynomial(1 .- x)
        Q̃ = Normalized(Jacobi(1,0))

        @test Q == Q
        @test P ≠ Q
        @test Q ≠ P
        @test Q == Q̃
        @test Q̃ == Q
        
        @test Q[0.1,1] ≈ 1/sqrt(2)
        @test Q[0.1,1:10] ≈ Q̃[0.1,1:10]
        # AWESOME, thanks TSGUT!!
        @test Q[0.1,10_000] ≈ Q̃[0.1,10_000]

        R = P \ Q
        @test inv(R[1:10,1:10]) ≈ (Q̃ \ P)[1:10,1:10]

        R = Q \ P
        @test bandwidths(R) == (0,1)
        @test R[1:10,1:10] ≈ (Q̃ \ P)[1:10,1:10]

        # need to fix InfiniteLinearAlgebra to add AdaptiveBandedLayout
        @test_broken R[1:10,1:10] isa BandedMatrix
    end
end