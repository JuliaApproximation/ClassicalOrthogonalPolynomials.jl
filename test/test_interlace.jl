using ClassicalOrthogonalPolynomials, BlockArrays, LazyBandedMatrices, Test
import ClassicalOrthogonalPolynomials: PiecewiseInterlace
@testset "Piecewise" begin
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
