using ClassicalOrthogonalPolynomials, BlockArrays, LazyBandedMatrices, FillArrays, ContinuumArrays, StaticArrays, Test
import ClassicalOrthogonalPolynomials: PiecewiseInterlace, SetindexInterlace, plotgrid

@testset "Piecewise" begin
    @testset "expansion" begin
        T1,T2 = chebyshevt(-1..0), chebyshevt(0..1)
        T = PiecewiseInterlace(T1, T2)
        @test T[-0.1,1:2:10] ≈ T1[-0.1,1:5]
        @test T[0.1,2:2:10] ≈ T2[0.1,1:5]
        @test T[0.0,1:2:10] ≈ T1[0.0,1:5]
        @test T[0.0,2:2:10] ≈ T2[0.0,1:5]

        x = axes(T,1)
        u = T / T \ exp.(x)
        @test u[-0.1] ≈ exp(-0.1)
        @test u[0.1] ≈ exp(0.1)
        @test u[0.] ≈ 2
    end

    @testset "two-interval ODE" begin
        T1,T2 = chebyshevt(-1..0), chebyshevt(0..1)
        U1,U2 = chebyshevu(-1..0), chebyshevu(0..1)
        T = PiecewiseInterlace(T1, T2)
        U = PiecewiseInterlace(U1, U2)

        @test copy(T) == T

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

    @testset "two-interval p-FEM" begin
        P1,P2 = jacobi(1,1,-1..0), jacobi(1,1,0..1)
        W1,W2 = Weighted(P1), Weighted(P2)

        W = PiecewiseInterlace(W1, W2)

        x = axes(W,1)
        D = Derivative(x)
        Δ = -((D*W)'*(D*W))

        @test (W'exp.(x))[1:2:10] ≈ (W1'*exp.(axes(W1,1)))[1:5]
        @test (W'exp.(x))[2:2:10] ≈ (W2'*exp.(axes(W2,1)))[1:5]
        M = W'W

        # Δ \ (W'exp.(x))
    end

    @testset "three-interval ODE" begin
        d = (-1..0),(0..1),(1..2)
        T = PiecewiseInterlace(chebyshevt.(d)...)
        U = PiecewiseInterlace(chebyshevu.(d)...)

        D = U \ (Derivative(axes(T,1))*T)
        C = U \ T

        A = BlockVcat(T[-1,:]',
                        BlockBroadcastArray(vcat,unitblocks(T.args[1][end,:]),-unitblocks(T.args[2][begin,:]),Zeros((unitblocks(axes(T.args[3],2)),)))',
                        BlockBroadcastArray(vcat,Zeros((unitblocks(axes(T.args[1],2)),)),unitblocks(T.args[2][end,:]),-unitblocks(T.args[3][begin,:]))',
                        D-C)
        N = 20
        M = BlockArray(A[Block.(1:N+2), Block.(1:N)])

        u = M \ [exp(-1); zeros(size(M,1)-1)]
        x = axes(T,1)

        F = factorize(T[:,Block.(Base.OneTo(N))])
        @test F \ exp.(x) ≈ (T \ exp.(x))[Block.(1:N)] ≈ u
    end


    @testset "plot" begin
        T1,T2 = chebyshevt(-1..0), chebyshevt(0..1)
        T = PiecewiseInterlace(T1, T2)
        @test plotgrid(T[:,1:5]) == sort([plotgrid(T1[:,1:3]); plotgrid(T2[:,1:3])])
    end

    @testset "SetindexInterlace" begin
        T = ChebyshevT(); U = ChebyshevU();
        V = SetindexInterlace(SVector(0.0,0.0), T, U)
        @test V[0.1,1:2:6] == vcat.(T[0.1,1:3],0)
        @test V[0.1,2:2:6] == vcat.(0,U[0.1,1:3])

        C² = Ultraspherical(2)
        C³ = Ultraspherical(3)
        M = SetindexInterlace(zero(SMatrix{2,2,Float64}), T, U, C², C³)
        @test M[0.1,1:4:12] == hvcat.(2, T[0.1,1:3], 0, 0, 0)
        @test M[0.1,2:4:12] == hvcat.(2, 0, 0, U[0.1,1:3], 0)
        @test M[0.1,3:4:12] == hvcat.(2, 0, C²[0.1,1:3], 0, 0)
        @test M[0.1,4:4:12] == hvcat.(2, 0, 0, 0, C³[0.1,1:3])
    end
end
