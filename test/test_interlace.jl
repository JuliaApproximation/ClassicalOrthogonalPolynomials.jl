using ClassicalOrthogonalPolynomials, BlockArrays, LazyBandedMatrices, FillArrays, Test
import ClassicalOrthogonalPolynomials: PiecewiseInterlace

@testset "Piecewise" begin
    @testset "two-interval ODE" begin
        T1,T2 = chebyshevt(-1..0), chebyshevt(0..1)
        U1,U2 = chebyshevu(-1..0), chebyshevu(0..1)
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
        T1,T2 = chebyshevt(-2..(-1)), chebyshevt(0..1)
        U1,U2 = chebyshevu(-2..(-1)), chebyshevu(0..1)
        W = PiecewiseInterlace(Weighted(T1), Weighted(T2))
        U = PiecewiseInterlace(U1, U2)
        x = axes(W,1)
        D = Derivative(x)
        @test_broken U\D*W isa BlockBroadcastArray
    end

    @testset "two-interval p-FEM" begin
        W1,W2 = Weighted(jacobi(1,1,-1..0)), Weighted(jacobi(1,1,0..1))
        P1,P2 = legendre(-1..0), legendre(0..1)
        W = PiecewiseInterlace(W1, W2)
        P = PiecewiseInterlace(P1, P2)
        x = axes(W,1)
        D = Derivative(x)
        P\D*W
    end
end
