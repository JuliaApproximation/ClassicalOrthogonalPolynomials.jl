using Test, ClassicalOrthogonalPolynomials, BandedMatrices, LinearAlgebra, LazyArrays, ContinuumArrays, LazyBandedMatrices, InfiniteLinearAlgebra
import ClassicalOrthogonalPolynomials: cholesky_jacobimatrix, qr_jacobimatrix
import LazyArrays: AbstractCachedMatrix, resizedata!


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

            @testset "Jacobi matrix multiplication" begin
                P = Normalized(legendre(0..1))
                x = axes(P,1)
                J = jacobimatrix(P)
                wf(x) = (1-x)^2
                sqrtwf(x) = (1-x)
                # compute Jacobi matrix via decomp
                Jchol = cholesky_jacobimatrix(wf, P)
                JqrQ = qr_jacobimatrix(sqrtwf, P)
                JqrR = qr_jacobimatrix(sqrtwf, P, :R)
                @test (Jchol*Jchol)[1:10,1:10] ≈ ApplyArray(*,Jchol,Jchol)[1:10,1:10]
                @test (Jchol*Jchol)[1:10,1:10] ≈ (JqrQ*JqrQ)[1:10,1:10]
                @test (Jchol*Jchol)[1:10,1:10] ≈ (JqrR*JqrR)[1:10,1:10]
            end
        end
    end

    @testset "Comparison of QR with Classical and Lanczos" begin
        @testset "QR case, w(x) = (1-x)^2" begin
            P = Normalized(legendre(0..1))
            x = axes(P,1)
            J = jacobimatrix(P)
            wf(x) = (1-x)^2
            sqrtwf(x) = (1-x)
            # compute Jacobi matrix via decomp
            Jchol = cholesky_jacobimatrix(wf, P)
            JqrQ = qr_jacobimatrix(sqrtwf, P)
            JqrR = qr_jacobimatrix(sqrtwf, P, :R)
            # use alternative inputs
            sqrtW = (P \ (sqrtwf.(x) .* P))
            JqrQalt = qr_jacobimatrix(sqrtW, P)
            JqrRalt = qr_jacobimatrix(sqrtW, P, :R)
            # compute Jacobi matrix via Lanczos
            Jlanc = jacobimatrix(LanczosPolynomial(@.(wf.(x)),Normalized(legendre(0..1))))
            # Comparison with Lanczos
            @test Jchol[1:500,1:500] ≈ Jlanc[1:500,1:500]
            @test JqrQ[1:500,1:500] ≈ Jlanc[1:500,1:500]
            @test JqrR[1:500,1:500] ≈ Jlanc[1:500,1:500]
            @test JqrQalt[1:500,1:500] ≈ Jlanc[1:500,1:500]
            @test JqrRalt[1:500,1:500] ≈ Jlanc[1:500,1:500]
        end
        @testset "QR case, w(x) = (1-x)^4" begin
            P = Normalized(legendre(0..1))
            x = axes(P,1)
            J = jacobimatrix(P)
            sqrtwf(x) = (1-x)^2
            # compute Jacobi matrix via decomp
            JqrQ = qr_jacobimatrix(sqrtwf, P)
            JqrR = qr_jacobimatrix(sqrtwf, P, :R)
            # use alternative inputs
            sqrtW = (P \ (sqrtwf.(x) .* P))
            JqrQalt = qr_jacobimatrix(sqrtW, P)
            JqrRalt = qr_jacobimatrix(sqrtW, P, :R)
            # compute Jacobi matrix via Lanczos
            Jclass = jacobimatrix(Normalized(jacobi(4,0,0..1)))
            # Comparison with Lanczos
            @test JqrQ[1:10,1:10] ≈ Jclass[1:10,1:10]
            @test JqrR[1:10,1:10] ≈ Jclass[1:10,1:10]
            @test JqrQalt[1:10,1:10] ≈ Jclass[1:10,1:10]
            @test JqrRalt[1:10,1:10] ≈ Jclass[1:10,1:10]
        end
        @testset "QR case, w(x) = (x)^2*(1-x)^4" begin
            P = Normalized(legendre(0..1))
            x = axes(P,1)
            J = jacobimatrix(P)
            sqrtwf(x) = (x)*(1-x)^2
            # compute Jacobi matrix via decomp
            JqrQ = qr_jacobimatrix(sqrtwf, P)
            JqrR = qr_jacobimatrix(sqrtwf, P, :R)
            # use alternative inputs
            sqrtW = (P \ (sqrtwf.(x) .* P))
            JqrQalt = qr_jacobimatrix(sqrtW, P)
            JqrRalt = qr_jacobimatrix(sqrtW, P, :R)
            # compute Jacobi matrix via Lanczos
            Jclass = jacobimatrix(Normalized(jacobi(4,2,0..1)))
            # Comparison with Lanczos
            @test JqrQ[1:10,1:10] ≈ Jclass[1:10,1:10]
            @test JqrR[1:10,1:10] ≈ Jclass[1:10,1:10]
            @test JqrQalt[1:10,1:10] ≈ Jclass[1:10,1:10]
            @test JqrRalt[1:10,1:10] ≈ Jclass[1:10,1:10]
            # test consistency of resizing in succession
            F = qr_jacobimatrix(sqrtwf, P);
            resizedata!(JqrQ.dv,70)
            resizedata!(JqrQ.ev,70)
            @test JqrQ[1:5,1:5] ≈ F[1:5,1:5]
            @test JqrQ[1:20,1:20] ≈ F[1:20,1:20]
            @test JqrQ[50:70,50:70] ≈ F[50:70,50:70]
        end
        @testset "BigFloat returns correct values"
            t = BigFloat("1.1")
            P = Normalized(legendre(big(0)..big(1)))
            Xq = qr_jacobimatrix(t*I-X, P, :Q)
            Xr = qr_jacobimatrix(t*I-X, P, :R)
            @test Xq[1:20,1:20] ≈ Xr[1:20,1:20]
            @test_broken Xq[1:20,1:20] ≈ cholesky_jacobimatrix(Symmetric((t*I-X)^2), P)[1:20,1:20]
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