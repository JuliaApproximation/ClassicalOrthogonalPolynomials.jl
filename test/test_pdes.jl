@testset "2D p-FEM" begin
    W = JacobiWeight(1,1) .* Jacobi(1,1)
    x = axes(W,1)
    D = Derivative(x)

    D2 = -((D*W)'*(D*W))
    M = W'W
    A = KronTrav(D2,M)
    N = 30;
    V = view(A,Block(N,N));
    @time MemoryLayout(arguments(V)[2]) isa LazyBandedMatrices.ApplyBandedLayout{typeof(*)}

    Δ = KronTrav(D2,M) + KronTrav(M,D2-M)
    N = 100; @time L = Δ[Block.(1:N+2),Block.(1:N)];
    r = KronTrav(M,M)[Block.(1:N+2),1]
    @time F = qr(L);
    @time u = F \ r;

    # u = Δ \ [1; zeros(∞)];
end