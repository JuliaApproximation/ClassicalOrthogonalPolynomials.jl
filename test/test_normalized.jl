using ClassicalOrthogonalPolynomials, FillArrays, BandedMatrices, ContinuumArrays, ArrayLayouts, LazyArrays, Base64, Test
import ClassicalOrthogonalPolynomials: NormalizationConstant, NormalizedBasisLayout, recurrencecoefficients, Normalized, Clenshaw, weighted
import LazyArrays: CachedVector, PaddedLayout
import ContinuumArrays: MappedWeightedBasisLayout

@testset "Normalized" begin
    @testset "Legendre" begin
        P = Legendre()
        Q = Normalized(P)

        @testset "Basic" begin
            @test MemoryLayout(Q) isa NormalizedBasisLayout
            @test (Q\Q) ≡ Eye(∞)
            @test Q == Q
            @test P ≠ Q
            @test Q ≠ P
            @test Q ≠ P[:,1:end]
            @test P[:,1:end] ≠ Q
        end

        @testset "recurrencecoefficients" begin
            A,B,C = recurrencecoefficients(Q)
            @test B isa Zeros
            @test A[3:∞][1:10] == A[3:12]
            @test B[3:∞] ≡ Zeros(∞)
        end

        @testset "Evaluation" begin
            M = P'P
            @test Q[0.1,1] == 1/sqrt(2)
            @test Q[0.1,2] ≈ sqrt(1/M[2,2]) * P[0.1,2]
            @test Q[0.1,Base.OneTo(10)] ≈ Q[0.1,1:10] ≈ sqrt.(inv(M)[1:10,1:10]) * P[0.1,Base.OneTo(10)]
            @test (Q'Q)[1:10,1:10] ≈ I
        end

        @testset "Expansion" begin
            f = Q*[1:5; zeros(∞)]
            @test f[0.1] ≈ Q[0.1,1:5]'*(1:5) ≈ f[[0.1]][1]
            x = axes(f,1)
            @test MemoryLayout(Q \ (1 .- x.^2)) isa PaddedLayout
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
        end

        @testset "show" begin
            @test stringmime("text/plain", Normalized(Legendre())) == "Normalized(Legendre{Float64})"
            @test summary(Normalized(Legendre()).scaling) == "NormalizationConstant{Float64}"
        end
    end

    @testset "Chebyshev" begin
        T = ChebyshevT()
        w = ChebyshevWeight()
        wT = WeightedChebyshevT()
        Q = Normalized(T)

        @testset "Basic" begin
            @test MemoryLayout(Q) isa NormalizedBasisLayout
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
        @test Pn * Q[y,:] isa CachedVector

        # @test (x-y) * Q[x,1:n]'*Q[y,1:n] ≈ (x-y) * Q[x,:]'*Pn*Q[y,:] ≈ (x-y) * Q[x,:]'*Pn*Q[y,:]
        # Q[x,:]' * ((X*Pn - Pn*X)* Q[y,:])
        @test (x-y) * Q[x,1:n]'*Q[y,1:n] ≈ Q[x,n:n+1]' * (X*Pn - Pn*X)[n:n+1,n:n+1] * Q[y,n:n+1]
    end
end
