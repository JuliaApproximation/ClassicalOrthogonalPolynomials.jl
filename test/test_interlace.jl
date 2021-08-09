using ClassicalOrthogonalPolynomials, BlockArrays, LazyBandedMatrices, Test
import ClassicalOrthogonalPolynomials: PiecewiseInterlace

@testset "Piecewise" begin
    @testset "two-interval ODE" begin
        T1,T2 = chebyshevt((-1)..0), chebyshevt(0..1)
        U1,U2 = chebyshevu((-1)..0), chebyshevu(0..1)
        T = PiecewiseInterlace(T1, T2)
        U = PiecewiseInterlace(U1, U2)
        D = U \ (Derivative(axes(T,1))*T)
        C = U \ T

        A = BlockVcat(T[-1,:]',
                        BlockBroadcastArray(vcat,unitblocks(T1[end,:]),-unitblocks(T2[begin,:]))',
                        D-C)
        N = 20
        M = BlockArray(A[Block.(1:N+1), Block.(1:N)])

        u = M \ [exp(-1); zeros(size(M,1)-1)]
        x = axes(T,1)

        F = factorize(T[:,Block.(Base.OneTo(N))])
        @test F \ exp.(x) ≈ (T \ exp.(x))[Block.(1:N)] ≈ u
    end

    @testset "two-interval Weighted Derivative" begin
        T1,T2 = chebyshevt((-2)..(-1)), chebyshevt(0..1)
        U1,U2 = chebyshevu((-2)..(-1)), chebyshevu(0..1)
        W = PiecewiseInterlace(Weighted(T1), Weighted(T2))
        U = PiecewiseInterlace(U1, U2)
        x = axes(W,1)
        D = Derivative(x)
        @test_broken U\D*W isa BlockBroadcastArray
    end

    @testset "two-interval Hilbert" begin
        T1,T2 = chebyshevt((-2)..(-1)), chebyshevt(0..2)
        U1,U2 = chebyshevu((-2)..(-1)), chebyshevu(0..2)
        W = PiecewiseInterlace(Weighted(U1), Weighted(U2))
        T = PiecewiseInterlace(T1, T2)
        x = axes(W,1)
        H = T \ inv.(x .- x') * W

        c = W \ broadcast(x -> exp(x)* (0 ≤ x ≤ 2 ? sqrt(2-x)*sqrt(x) : sqrt(-1-x)*sqrt(x+2)), x)
        @test T[0.5,1:200]'*(H*c)[1:200] ≈ -6.064426633490422
    end
end
