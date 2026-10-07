using ClassicalOrthogonalPolynomials, FillArrays, BandedMatrices, ContinuumArrays, ArrayLayouts, LazyArrays, Base64, LinearAlgebra, QuasiArrays, Test
import ClassicalOrthogonalPolynomials: NormalizedOPLayout, recurrencecoefficients, Normalized, Clenshaw, weighted, grid, plotgrid
import LazyArrays: CachedVector, PaddedColumns
using BlockArrays: blockedrange, Block
import ContinuumArrays: MappedWeightedBasisLayout

@testset "Normalized" begin
    @testset "Legendre" begin
        P = Legendre()
        Q = Normalized(P)

        @testset "Basic" begin
            @test MemoryLayout(Q) isa NormalizedOPLayout
            @test (Q\Q) ≡ Eye(∞)
            @test Q == Q
            @test P ≠ Q
            @test Q ≠ P
            @test Q ≠ P[:,1:end]
            @test P[:,1:end] ≠ Q
        end

        @testset "recurrencecoefficients" begin
            A,B,C = recurrencecoefficients(Q)
            @test B isa Zeros
            @test A[3:∞][1:10] == A[3:12]
            @test B[3:∞] ≡ Zeros(∞)
        end

        @testset "Evaluation" begin
            M = P'P
            @test Q[0.1,1] ≈ 1/sqrt(2)
            @test Q[0.1,2] ≈ sqrt(1/M[2,2]) * P[0.1,2]
            @test Q[0.1,Base.OneTo(10)] ≈ Q[0.1,1:10] ≈ sqrt.(inv(M)[1:10,1:10]) * P[0.1,Base.OneTo(10)]
            @test (Q'Q)[1:10,1:10] ≈ I
        end

        @testset "Expansion" begin
            f = Q*[1:5; zeros(∞)]
            @test f[0.1] ≈ Q[0.1,1:5]'*(1:5) ≈ f[[0.1]][1]
            x = axes(f,1)
            @test MemoryLayout(Q \ (1 .- x.^2)) isa PaddedColumns
            w = Q * (Q \ (1 .- x.^2));
            @test w[0.1] ≈ (1-0.1^2) ≈ w[[0.1]][1]
        end

        @testset "Conversion" begin
            @test ((P \ Q) * (Q \ P))[1:10,1:10] ≈ (Q \Q)[1:10,1:10] ≈ I
            @test (Jacobi(1,1) \ Q)[1:10,1:10] ≈ ((Jacobi(1,1) \ P) * (P \ Q))[1:10,1:10]
        end

        @testset "Derivatives" begin
            D = Derivative(axes(Q,1))
            f = Q*[1:5; zeros(∞)]
            h = 0.000001
            @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=1E-4
        end

        @testset "Jacobi" begin
            X = jacobimatrix(Q)
            M = P'P
            @test X[1:10,1:10] ≈ sqrt(M)[1:10,1:10] * jacobimatrix(P)[1:10,1:10] * inv(sqrt(M))[1:10,1:10]
            @test 0.1*Q[0.1,1:10] ≈ (Q*X)[0.1,1:10]
        end

        @testset "Multiplication" begin
            x = axes(Q,1)
            @test Q \ (x .* Q) isa ClassicalOrthogonalPolynomials.SymTridiagonal

            w = P * (P \ (1 .- x.^2));
            W = Q \ (w .* Q)
            @test W isa Clenshaw
            W̃ = Q' * (w .* Q)
            @test bandwidths(W) == bandwidths(W̃) == (2,2)
            @test W[1:10,1:10] ≈ W[1:10,1:10]' ≈ W̃[1:10,1:10]

            w = @. x + x^2 + 1 # w[x] == x + x^2 + 1
            W = Q \ (w .* Q)
            @test W isa Clenshaw

            # Clenshaw is symmetric for normalized polynomials
            S = Symmetric(W)
            @test S[2,1:7] ≈ W[2,1:7]
            @test S[1:10,2] ≈ W[1:10,2]
            @test S[1:10,1:10] ≈ W[1:10,1:10]
            @test S[13,15] ≈ W[13,15]
        end

        @testset "show" begin
            @test stringmime("text/plain", Normalized(Legendre())) == "Normalized(Legendre())"
        end

        @testset "qr" begin
            P = Legendre()
            Q,R = qr(P)
            @test_throws BoundsError (Q,R,p) = qr(P)
            @test Q == Normalized(P)
            @test Q ≠ P
            @test P ≠ Q
            @test LinearSpline(-1:1) ≠ Q
            @test Q ≠ LinearSpline(-1:1)
            @test R[1:10,1:10] == (P\Q)[1:10,1:10]
        end
    end

    @testset "Chebyshev" begin
        T = ChebyshevT()
        w = ChebyshevWeight()
        wT = Weighted(ChebyshevT())
        Q = Normalized(T)

        @testset "Basic" begin
            @test MemoryLayout(Q) isa NormalizedOPLayout
            @test (Q\Q) ≡ Eye(∞)
        end

        @testset "recurrencecoefficients" begin
            A,B,C = recurrencecoefficients(Q)
            @test A[1] ≈ sqrt(2)
            @test A[2:5] ≈ fill(2,4)
            @test C[1:3] ≈ [0,sqrt(2),1]
            @test A[3:∞][1:10] == A[3:12]
            @test B[3:∞] ≡ Zeros(∞)
        end

        @testset "Evaluation" begin
            M = T'wT
            @test Q[0.1,1] == 1/sqrt(π)
            @test Q[0.1,2] ≈ sqrt(1/M[2,2]) * T[0.1,2]
            @test Q[0.1,Base.OneTo(10)] ≈ Q[0.1,1:10] ≈ sqrt.(inv(M)[1:10,1:10]) * T[0.1,Base.OneTo(10)]
            @test (Q'*(w .* Q))[1:10,1:10] ≈ I
        end

        @testset "Expansion" begin
            f = Q*[1:5; zeros(∞)]
            @test f[0.1] ≈ Q[0.1,1:5]'*(1:5) ≈ f[[0.1]][1]
            x = axes(f,1)
            w = Q * (Q \ (1 .- x.^2));
            @test w[0.1] ≈ (1-0.1^2) ≈ w[[0.1]][1]
            @test (Q \ [(1 .- x.^2) x])[1:4,:] ≈ [(Q\w)[1:4] (Q\x)[1:4]]
        end

        @testset "Conversion" begin
            @test ((T \ Q) * (Q \ T))[1:10,1:10] ≈ (Q \Q)[1:10,1:10] ≈ I
            @test (ChebyshevU() \ Q)[1:10,1:10] ≈ ((ChebyshevU() \ T) * (T \ Q))[1:10,1:10]
        end

        @testset "Derivatives" begin
            D = Derivative(axes(Q,1))
            f = Q*[1:5; zeros(∞)]
            h = 0.000001
            @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=1E-4
        end

        @testset "Multiplication" begin
            x = axes(Q,1)
            @test Q \ (x .* Q) isa ClassicalOrthogonalPolynomials.SymTridiagonal

            w = T * (T \ (1 .- x.^2));
            W = Q \ (w .* Q)
            @test W isa Clenshaw
            @test bandwidths(W) == (2,2)
            W̃ = Q\  (w .* Q)
            @test W[1:10,1:10] ≈ W[1:10,1:10]' ≈ W̃[1:10,1:10]
        end
    end

    @testset "Jacobi" begin
        Q = Normalized(Jacobi(1/2,0))
        # Emperical from Mathematica
        @test Q[0.1,1:4] ≈ [0.728237657560985,0.41715052371131806,-0.6523500049588019,-0.5607891513201705]
        w = JacobiWeight(1/2,0)
        @test (Q'*(w .* Q))[1:10,1:10] ≈ I

        P⁰⁰ = Normalized(Jacobi(0, 0))
        P¹⁰ = Normalized(Jacobi(1, 0))
        L = Jacobi(0, 0) \ Weighted(Jacobi(1, 0))
        L̄ = (Diagonal(P⁰⁰.scaling) \ L) * Diagonal(P¹⁰.scaling)
        L2 = P⁰⁰ \ Weighted(P¹⁰)
        @test L2[1:100, 1:100] ≈ L̄[1:100, 1:100]
    end

    @testset "Mapped" begin
        P = legendre(0..1)
        x = axes(P,1)
        Q = Normalized(P)

        # Emperical from Mathematica
        @test Q[0.1,1:4] ≈ [1,-1.3856406460551018,1.028591269649903,-0.21166010488516684]

        u = Q[:,1:20] * (Q[:,1:20] \ exp.(x))
        @test u[0.1] ≈ exp(0.1)
        u = Q * (Q \ exp.(x))
        @test u[0.1] ≈ exp(0.1)

        @test P \ Q isa Diagonal
        @test Q \ P isa Diagonal

        Q = Normalized(jacobi(1/2,0,0..1))
        @testset "Recurrences" begin
            A,B,C = recurrencecoefficients(Q)
            Ã,B̃,C̃ = recurrencecoefficients(Normalized(Jacobi(1/2,0)))
            @test A[1:10] ≈ 2Ã[1:10]
            @test B[1:10] ≈ B̃[1:10] .- Ã[1:10]
            @test C[1:10] ≈ C̃[1:10]
        end
        wQ = weighted(Q)
        x = axes(Q,1)
        @test wQ[0.1,1:10] ≈ Q[0.1,1:10] * sqrt(1-(2*0.1-1))

        u = wQ[:,1:20] * (wQ[:,1:20] \  @.(sqrt(1-x^2)))
        @test u[0.1] ≈ sqrt(1-0.1^2)
        u = wQ * (wQ \ @.(sqrt(1-x^2)))
        @test u[0.1] ≈ sqrt(1-0.1^2)
    end

    @testset "Christoffel–Darboux" begin
        Q = Normalized(Legendre())
        X = Q\ (axes(Q,1) .* Q)
        x,y = 0.1,0.2
        n = 10
        Pn = Diagonal([Ones(n); Zeros(∞)])
        @test (X*Pn - Pn*X)[1:n,1:n] ≈ zeros(n,n)
        @test MemoryLayout(Pn * Q[y,:]) isa PaddedColumns

        # @test (x-y) * Q[x,1:n]'*Q[y,1:n] ≈ (x-y) * Q[x,:]'*Pn*Q[y,:] ≈ (x-y) * Q[x,:]'*Pn*Q[y,:]
        # Q[x,:]' * ((X*Pn - Pn*X)* Q[y,:])
        @test (x-y) * Q[x,1:n]'*Q[y,1:n] ≈ Q[x,n:n+1]' * (X*Pn - Pn*X)[n:n+1,n:n+1] * Q[y,n:n+1]
    end

    @testset "plotting" begin
        P = Legendre()
        Q = Normalized(P)
        @test grid(Q[:,1:5]) == grid(Q[:,collect(1:5)]) # == grid(P[:,1:5])
        @test plotgrid(Q[:,1:5]) == plotgrid(Q[:,collect(1:5)]) # == plotgrid(P[:,1:5])
    end

    @testset "Transform" begin
        Q = Normalized(Hermite())
        n = 20
        Qₙ = Q[:,Base.OneTo(n)]
        x = axes(Q,1)
        g = grid(Qₙ)
        v = exp.(g)
        P = plan_transform(Q, v)
        @test P * v ≈ Qₙ[g,:] \ exp.(g) ≈ transform(Qₙ, exp)

        V = cos.(g .* (1:3)')
        P = plan_transform(Q, V, 1)
        @test P * V ≈ Qₙ \ cos.(x .* (1:3)')

        X = randn(n, n)
        P₂ = plan_transform(Q, X, 2)

        P = plan_transform(Q, X)

        PX = P * X
        for k = 1:n
            X[:, k] = Qₙ[g,:] \ X[:, k]
        end
        for k = 1:n
            X[k, :] = Qₙ[g,:] \ X[k, :]
        end
        @test PX ≈ X

        X = randn(n, n, n)
        P = plan_transform(Q, X)
        PX = P * X
        for k = 1:n, j = 1:n
            X[:, k, j] = Qₙ[g,:] \ X[:, k, j]
        end
        for k = 1:n, j = 1:n
            X[k, :, j] = Qₙ[g,:] \ X[k, :, j]
        end
        for k = 1:n, j = 1:n
            X[k, j, :] = Qₙ[g,:] \ X[k, j, :]
        end
        @test PX ≈ X
    end

    @testset "simplifable" begin
        P = Legendre()
        Q = Normalized(P)
        f = expand(Q, exp)
        @test (Q*(Q\f))[0.1] ≈ exp(0.1)

        W = JacobiWeight(1,1) .* Normalized(Jacobi(1,1))
        g = ApplyQuasiArray(*, W, [1:3; zeros(∞)])
        @test P \ g ≈ transform(P, x -> g[x])
    end

    @testset "laplacian" begin
        x = 0.3
        c = [1:6; zeros(∞)]
        for P in (Legendre(), Chebyshev(), Jacobi(1.0,1.0), Jacobi(0.1,0.2))
            Q = Normalized(P)
            D = Diagonal(Q.scaling)
            Δ = laplacian(Q)
            @test eltype(Δ) == Float64
            @test Δ[x,1:6] ≈ diff(Q,2)[x,1:6] ≈ laplacian(P)[x,1:6] .* Q.scaling[1:6]
            @test (Δ * c)[x] ≈ (laplacian(P) * (D * c))[x]
            @test (Laplacian(axes(Q,1)) * Q)[x,1:6] ≈ Δ[x,1:6]
            @test abslaplacian(Q)[x,1:6] ≈ abslaplacian(Q,1)[x,1:6] ≈ -Δ[x,1:6]
            @test laplacian(Q,2)[x,1:6] ≈ diff(Q,4)[x,1:6]
        end
        P = Legendre()
        Q = Normalized(P)
        @test (Ultraspherical(5/2) \ laplacian(Q))[1:5,1:7] ≈ (Ultraspherical(5/2) \ laplacian(P))[1:5,1:7] * Diagonal(Q.scaling[1:7])

        for P in (Jacobi(1,1), Jacobi(2,2), Jacobi(2.0,3.0))
            Q = Normalized(P)
            W = Weighted(Q)
            D = Diagonal(Q.scaling)
            Δ = laplacian(W)
            @test Δ[x,1:6] ≈ laplacian(Weighted(P))[x,1:6] .* Q.scaling[1:6]
            @test (Δ * c)[x] ≈ (laplacian(Weighted(P)) * (D * c))[x]
            @test (Laplacian(axes(W,1)) * W)[x,1:6] ≈ Δ[x,1:6]
            @test abslaplacian(W)[x,1:6] ≈ abslaplacian(W,1)[x,1:6] ≈ abslaplacian(Weighted(P),1)[x,1:6] .* Q.scaling[1:6] ≈ -Δ[x,1:6]
            @test (Legendre() \ Δ)[1:5,1:5] ≈ (Legendre() \ laplacian(Weighted(P)))[1:5,1:5] * D[1:5,1:5]
            # compare with finite differences
            f = W * c
            h = 1E-3
            @test (Δ * c)[x] ≈ (f[x+h] - 2f[x] + f[x-h])/h^2 rtol=1E-4
        end
        @test laplacian(Weighted(Normalized(Jacobi(1,1))))[x,1] ≈ -2Normalized(Jacobi(1,1)).scaling[1]
    end

    @testset "inv bug (#182)" begin
        P = Jacobi(2.0, 0.5)
        Q = Jacobi(3.0, 0.5)
        @test (P \ Normalized(Q))[1:10,1:10] ≈ inv((Normalized(Q) \ P)[1:10,1:10])
    end

    @testset "normalized_layout and evaluation" begin
        P = Legendre()
        Q = Normalized(P)
        @test MemoryLayout(Q) == ClassicalOrthogonalPolynomials.normalized_layout(MemoryLayout(P)) == NormalizedOPLayout{typeof(MemoryLayout(P))}()
        x = 0.1
        @test Q[x, 5] ≈ P[x, 5] * Q.scaling[5]
        @test Q[x, 2:5] ≈ P[x, 2:5] .* Q.scaling[2:5]
        @test Q[[0.1,0.2], 1:5] ≈ P[[0.1,0.2], 1:5] .* Q.scaling[1:5]'
        @test Q[[0.1,0.2], 2:5] ≈ P[[0.1,0.2], 2:5] .* Q.scaling[2:5]'
        @test Q[[0.1,0.2], 1:∞][:, 1:5] ≈ Q[[0.1,0.2], 1:5]
        @test Q \ Q isa Eye
        @test (Q'P)[1:3,1:3] ≈ Diagonal(Q.scaling[1:3]) * (P'P)[1:3,1:3]
    end

    @testset "transform keeps axes" begin
        Q = Normalized(Legendre())
        x = axes(Q,1)
        c = Q \ exp.(x)
        @test c[1:5] ≈ Q.scaling[1:5] .\ (Legendre() \ exp.(x))[1:5]
        @test axes(ClassicalOrthogonalPolynomials.padrows([1.0, 2], axes(Q,2))) ≡ (axes(Q,2),)
        # e.g. for normalized multivariate OPs the coefficients are blocked
        ax = blockedrange(Base.oneto(∞))
        d = ClassicalOrthogonalPolynomials.padrows([1.0, 2, 3], ax)
        @test axes(d,1) == ax
        @test d[Block(2)] == [2, 3]
        @test d[Block(3)] == zeros(3)
    end
end
