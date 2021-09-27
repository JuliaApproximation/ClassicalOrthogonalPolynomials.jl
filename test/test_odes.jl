using ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, BandedMatrices,
        SemiseparableMatrices, LazyArrays, ArrayLayouts, Test

import QuasiArrays: MulQuasiMatrix
import ClassicalOrthogonalPolynomials: oneto
import ContinuumArrays: MappedBasisLayout, MappedWeightedBasisLayout
import LazyArrays: arguments, ApplyMatrix, colsupport, MemoryLayout
import SemiseparableMatrices: VcatAlmostBandedLayout

@testset "ODEs" begin
    @testset "p-FEM" begin
        S = Jacobi(true,true)
        w = JacobiWeight(true,true)
        D = Derivative(axes(S,1))
        P = Legendre()

        @test w.*S isa QuasiArrays.BroadcastQuasiMatrix

        M = P\(D*(w.*S))
        @test M isa BandedMatrix
        @test M[1:10,1:10] == diagm(-1 => -2.0:-2:-18.0)

        N = 10
        A = D* (w.*S)[:,1:N]
        @test A.args[1] == P
        @test P\(D*(w.*S)[:,1:N]) isa ApplyMatrix{<:Any,typeof(*)}

        L = D*(w.*S)
        Δ = L'L
        @test Δ isa ApplyMatrix{<:Any,typeof(*)}
        @test Δ[1:3,1:3] isa BandedMatrix
        @test bandwidths(Δ) == (0,0)

        L = D*(w.*S)[:,1:N]

        A  = *((L').args..., L.args...)
        @test A isa ApplyMatrix{<:Any,typeof(*)}

        Δ = L'L
        @test Δ isa ApplyMatrix{<:Any,typeof(*)}
        @test bandwidths(Δ) == (0,0)
        @test BandedMatrix(Δ) == Δ
        @test BandedMatrix(Δ) isa BandedMatrix
    end

    @testset "∞-FEM" begin
        @testset "w .* P(true,true)" begin
            S = Jacobi(true,true)
            w = JacobiWeight(true,true)
            D = Derivative(axes(w,1))
            WS = w.*S
            L = D* WS
            Δ = L'L
            P = Legendre()

            f = P * Vcat(randn(10), Zeros(∞))
            (P\WS)'*(P'P)*(P\WS)
            B = BroadcastArray(+, Δ, (P\WS)'*(P'P)*(P\WS))
            @test colsupport(B,1) == 1:3

            @test axes(B.args[2].args[1]) == (oneto(∞),oneto(∞))
            @test axes(B.args[2]) == (oneto(∞),oneto(∞))
            @test axes(B) == (oneto(∞),oneto(∞))

            @test BandedMatrix(view(B,1:10,13:20)) == zeros(10,8)

            F = qr(B);
            b = Vcat(randn(10), Zeros(∞))
            @test B*(F \ b) ≈ b
        end

        @testset "simple" begin
            W = Weighted(Jacobi(1,1))
            P = Legendre()
            x = axes(W,1)
            D = Derivative(x)
            Δ = -((D*W)'*(D*W))
            u = W * (Δ \ (W'*exp.(x)))
            let x = 0.1
                @test u[x] ≈ (-1 + 2exp(1 + x) + x - exp(2)*(1 + x))/(2ℯ)
            end

            M = W'W
            u = W * ((Δ+10M) \ (W'*exp.(x)))
            let x = 0.1
                @test u[x] ≈ (csc(2sqrt(10))*sin(sqrt(10)*(-1 + x)) + ℯ*(exp(x) - ℯ*csc(2sqrt(10))*sin(sqrt(10)*(1 + x))))/(11ℯ)
            end

            V = W'*(x.*W)
            u = W * ((Δ+10V) \ (W'*exp.(x)))
            @test u[0.1] ≈ -1.6914064963142479 # mathematica
        end
    end

    @testset "Collocation" begin
        P = Chebyshev()
        D = Derivative(axes(P,1))
        n = 300
        x = cos.((0:n-2) .* π ./ (n-2))
        cfs = [P[-1,1:n]'; (D*P)[x,1:n] - P[x,1:n]] \ [exp(-1); zeros(n-1)]
        u = P[:,1:n]*cfs
        @test u[0.1] ≈ exp(0.1)

        P = Chebyshev()
        D = Derivative(axes(P,1))
        D2 = D^2 * P # could be D^2*P in the future
        n = 300
        x = cos.((1:n-2) .* π ./ (n-1)) # interior Chebyshev points
        C = [P[-1,1:n]';
            D2[x,1:n] + P[x,1:n];
            P[1,1:n]']
        cfs = C \ [1; zeros(n-2); 2] # Chebyshev coefficients
        u = P[:,1:n]*cfs  # interpret in basis
        @test u[0.1] ≈ (3cos(0.1)sec(1) + csc(1)sin(0.1))/2
    end

    @testset "∞-dimensional Dirichlet" begin
        S = Jacobi(true,true)
        w = JacobiWeight(true,true)
        D = Derivative(axes(S,1))
        X = QuasiDiagonal(axes(S,1))

        @test ((Legendre() \ S)*(S\(w.*S)))[1:10,1:10] ≈ (Legendre() \ (w .* S))[1:10,1:10]
        @test (Ultraspherical(3/2)\(D^2*(w.*S)))[1:10,1:10] ≈ diagm(0 => -(2:2:20))
    end

    @testset "rescaled" begin
        x = Inclusion(0..1)
        S = Jacobi(1.0,1.0)[2x.-1,:]
        D = Derivative(x)
        f = S*[[1,2,3]; zeros(∞)]
        g = Jacobi(1.0,1.0)*[[1,2,3]; zeros(∞)]
        @test f[0.1] ≈ g[2*0.1-1]
        h = 0.0000001
        @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=100h
        @test Jacobi(2.0,2.0)[2x.-1,:] \ (D*S).args[1] isa BandedMatrix
        @test (Jacobi(2.0,2.0)[2x.-1,:] \ (D*S))[1:10,1:10] == diagm(1 => 4:12)

        P = Legendre()[2x.-1,:]
        w = JacobiWeight(1.0,1.0)
        wS = (w .* Jacobi(1.0,1.0))[2x.-1,:]
        @test MemoryLayout(wS) isa MappedWeightedBasisLayout
        f = wS*[[1,2,3]; zeros(∞)]
        g = (w .* Jacobi(1.0,1.0))*[[1,2,3]; zeros(∞)]
        @test f[0.1] ≈ g[2*0.1-1]
        h = 0.0000001
        @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=100h
        @test P == P

        @test P.parent == (D*wS).args[1].parent
        DwS = apply(*,D,wS)
        A,B = P,arguments(DwS)[1];
        @test (A.parent\B.parent) == Eye(∞)
        @test (A \ B)[1:10,1:10] == diagm(-1 => ones(9))
        @test (P \ DwS)[1:10,1:10] == diagm(-1 => -4:-4:-36)
    end

    @testset "Beam" begin
        P = JacobiWeight(0.0,0.0) .* Jacobi(0.0,0.0)
        x = axes(P,1)
        D = Derivative(x)
        @test (D*P).args[1] == Jacobi{Float64}(1,1)
        @test (Jacobi(1,1)\(D*P))[1:10,1:10] ≈ (Jacobi(1,1) \ (D*Legendre()))[1:10,1:10]

        S = JacobiWeight(2.0,2.0) .* Jacobi(2.0,2.0)
        @test (Legendre() \ S)[1,1] ≈ 0.533333333333333333
        Δ² = (D^2*S)'*(D^2*S)
        M = S'S
    end

    @testset "Ultraspherical spectral method" begin
        T = ChebyshevT()
        U = ChebyshevU()
        x = axes(T,1)
        D = Derivative(x)
        A = U\(D*T) - U\T
        @test copyto!(BandedMatrix{Float64}(undef, (10,10), (0,2)), view(A,1:10,1:10)) == A[1:10,1:10]
        L = Vcat(T[1:1,:], A)
        @test L[1:10,1:10] isa AlmostBandedMatrix
        @test MemoryLayout(L) isa VcatAlmostBandedLayout
        u = L \ [ℯ; zeros(∞)]
        @test T[0.1,:]'u ≈ (T*u)[0.1] ≈ exp(0.1)

        C = Ultraspherical(2)
        A = C \ (D^2 * T) - C\(x .* T)
        L = Vcat(T[[-1,1],:], A)
        @test qr(L).factors[1:10,1:10] ≈ qr(L[1:13,1:10]).factors[1:10,1:10]
        u = L \ [airyai(-1); airyai(1); Zeros(∞)]
        @test T[0.1,:]'u ≈ airyai(0.1)

        ε = 0.0001
        A = ε^2 * (C \ (D^2 * T)) - C\(x .* T)
        L = Vcat(T[[-1,1],:], A)
        u = L \ [airyai(-ε^(-2/3)); airyai(ε^(2/3)); zeros(∞)]
        @test T[-0.1,:]'u ≈ airyai(-0.1*ε^(-2/3))

        @testset "auto-vcat" begin
            L = [T[[begin,end],:]; C \ (D^2 * T - x .* T)]
            c = L \ [airyai(-1); airyai(1); zeros(∞)]
            u = T*c
            @test u[0.0] ≈ airyai(0.0)
        end
    end

    @testset "combo operators" begin
        T = Chebyshev()
        C = Ultraspherical(2)
        x = axes(T,1)
        D = Derivative(x)
        L = x .* D + cos.(x) .* D^2
        M = C \ (L * T)
        @test C[0.1,:]' * (M * (T \ exp.(x))) ≈ (0.1 + cos(0.1))*exp(0.1)
    end

    @testset "Neumann" begin
        W = Weighted(Jacobi(1,1))
        x = axes(W,1)
        P = Legendre()
        D = Derivative(x)

        @testset "[x W]" begin
            Q = [x W]
            Δ = -((P\D*Q)'*(P'P)*(P\D*Q))
            @test bandwidths(Δ) == (0,0)
            c = Δ \ (Q'*exp.(x))
            u = Q * c
            @test u[0.1] ≈ -0.5922177802211208

            Δ = -((D*Q)'*(D*Q))
            @test bandwidths(Δ) == (0,0)
            c = Δ \ (Q'*exp.(x))
            u = Q * c
            @test u[0.1] ≈ -0.5922177802211208
        end

        @testset "natural" begin
            Q = [one(x) x W]
            Δ = -((D*Q)'*(D*Q))
            M = Q'Q
            @test isbanded(Δ)
            @test isbanded(M)
            @test bandwidths(M) == (2,2)
            @test bandwidths(Δ) == (0,0)
            c = (Δ + M) \ (Q'exp.(x))
            u = Q * c
            @test u[0.1] ≈ 1.104838515599687
        end

        @testset "one-sided" begin
            Q = [one(x)-x W]
            Δ = -((D*Q)'*(D*Q))
            M = Q'Q;
            c = (Δ + M) \ (Q'exp.(x))
            u = Q * c
            @test u[0.1] ≈ 1.6878004187402804
        end
    end
end