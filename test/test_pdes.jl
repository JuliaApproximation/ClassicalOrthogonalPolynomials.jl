using ClassicalOrthogonalPolynomials, LazyBandedMatrices, Test


@testset "PDEs" begin
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

    @testset "Annuli" begin
        f = (r,θ) -> exp(r*cos(θ)+sin(θ))
        T,F = chebyshevt(ρ..1),Fourier()
        n = 1000 # create a 1000 x 1000 transform
        𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, n),ClassicalOrthogonalPolynomials.grid(F, n)
        PT,PF = plan_transform(T, (n,n), 1),plan_transform(F, (n,n), 2)
        
        @time X = PT * (PF * f.(𝐫, 𝛉'))

        @test T[0.1,1:n]'*X*F[0.2,1:n] ≈ f(0.1,0.2)
    end
end