using ClassicalOrthogonalPolynomials, FillArrays, BandedMatrices, ContinuumArrays, QuasiArrays, LazyArrays, LazyBandedMatrices, FastGaussQuadrature, Test
import ClassicalOrthogonalPolynomials: recurrencecoefficients, basis, MulQuasiMatrix, arguments, Weighted, HalfWeighted, grammatrix

@testset "Jacobi" begin
    @testset "JacobiWeight" begin
        a,b = 0.1,0.2
        w = JacobiWeight(a,b)

        @test AbstractQuasiArray{Float32}(w) ≡ AbstractQuasiVector{Float32}(w) ≡ JacobiWeight{Float32}(a, b)

        @test w.^2 == JacobiWeight(2a,2b)
        @test sqrt.(w) == JacobiWeight(a/2,b/2)
        @test JacobiWeight(0.2,0.3) .* w == JacobiWeight(a+0.2,b+0.3)
        @test LegendreWeight() .* w == w .* LegendreWeight() == w
        @test ChebyshevWeight() .* w == w .* ChebyshevWeight() == JacobiWeight(a-1/2,b-1/2)
        @test summary(w) == "(1-x)^0.1 * (1+x)^0.2 on -1..1"

        x = axes(w,1)
        @test 0 * (1 .+ x) == LegendreWeight()
        @test 1 .+ x == JacobiWeight(0,1)
        @test 1 .- x == JacobiWeight(1,0)
        @test x .- 1 ≠ JacobiWeight(1,0)
    end

    @testset "basics" begin
        @test Legendre() == Jacobi(0,0)
        @test Jacobi(0,0) == Legendre()
        @test Jacobi(1,2) .* Legendre() == Jacobi(1,2) .* Jacobi(0,0)

        J = Jacobi(1,2)
        @test AbstractQuasiArray{Float32}(J) ≡ AbstractQuasiMatrix{Float32}(J) ≡ Jacobi{Float32}(1,2)
    end

    @testset "basis" begin
        b,a = 0.1,0.2
        P = Jacobi(a,b)
        @test P[0.1,2] ≈ 0.16499999999999998
        @test summary(P) == "Jacobi(0.2, 0.1)"

        P = Jacobi(b,a)
        @test P[-0.1,2] ≈ -0.16499999999999998

        A,B,C = recurrencecoefficients(P)
        X = jacobimatrix(P)
        @testset "recurrence coefficient and jacobimatrix" begin
            @test 1/A[1] ≈ X[2,1]
            @test -B[1]/A[1] ≈ X[1,1]
            @test C[2]/A[2] ≈ X[1,2]
            @test 1/A[2] ≈ X[3,2]
            @test -B[2]/A[2] ≈ X[2,2]
            @test C[3]/A[3] ≈ X[2,3]

            @test A[1] ≈ 1/X[2,1]
            @test B[1] ≈ -X[1,1]/X[2,1]
            @test C[2] ≈ X[1,2]/X[3,2]
            @test A[2] ≈ 1/X[3,2]
            @test B[2] ≈ -X[2,2]/X[3,2]
            @test C[3] ≈ X[2,3]/X[4,3]
        end
    end

    @testset "orthogonality" begin
        @testset "legendre" begin
            P̃ = Jacobi(0,0)
            x = axes(P̃,1)
            @test P̃'exp.(x) ≈ Legendre()'exp.(x)
            @test P̃'P̃ isa Diagonal
        end

        @testset "integer" begin
            P¹ = Jacobi(1,1)
            P = Legendre()
            x = axes(P¹,1)
            @test (P¹'exp.(x))[1:10] ≈ (P¹'P)[1:10,1:10] * (P \ exp.(x))[1:10]
        end

        @testset "fractional a,b" begin
            a,b = 0.1,0.2
            x,w = gaussjacobi(3,a,b)
            P = Jacobi(a,b)

            M = P[x,1:3]'Diagonal(w)*P[x,1:3]
            @test M ≈ Diagonal(M)
            x,w = FastGaussQuadrature.gaussradau(3,a,b)
            M = P[x,1:3]'Diagonal(w)*P[x,1:3]
            @test M ≈ Diagonal(M)

            w = JacobiWeight(a,b)
            w_2 = sqrt.(w)
            @test w_2 .^ 2 == w
            @test (P' * (w .* P))[1:3,1:3] ≈ (P' * (w .* P))[1:3,1:3] ≈ ((w_2 .* P)'*(w_2 .* P))[1:3,1:3] ≈ M

            @test (Jacobi(0,0)'Jacobi(0,0))[1:10,1:10] ≈ (Legendre()'Legendre())[1:10,1:10]
        end
    end

    @testset "operators" begin
        a,b = 0.1,0.2
        S = Jacobi(a,b)

        @testset "Jacobi" begin
            x = 0.1
            @test S[x,1] === 1.0
            X = jacobimatrix(S)
            @test X[1,1] ≈ (b^2-a^2)/((a+b)*(a+b+2))
            @test X[2,1] ≈ 2/(a+b+2)
            @test S[x,2] ≈ 0.065
            @test S[x,10] ≈ 0.22071099583604945

            w = JacobiWeight(a,b)
            @test w[x] ≈ (1-x)^a * (1+x)^b
            @test OrthogonalPolynomial(w) == S
            wS = w.*S
            @test wS == Weighted(Jacobi(a,b)) == Weighted(Jacobi{Float64}(a,b))
            @test wS[0.1,1] ≈ w[0.1]
            @test wS[0.1,1:2] ≈ w[0.1] .* S[0.1,1:2]

            w_A = Weighted(Jacobi(-1/2,0))
            w_B =  Weighted(Jacobi(1/2,0))

            u = w_A * [1 ; 2; zeros(∞)]
            v = w_B * [1 ; 2; zeros(∞)]
            @test basis(u + v) == w_A
            @test (u+v)[0.1] == u[0.1] + v[0.1]
        end

        @testset "Clenshaw" begin
            x = axes(S,1)
            a = S * (S \ exp.(x))
            A = S \ (a .* S)
            @test (S * (A * (S \ a)))[0.1] ≈ exp(0.2)
            a = S * [1; Zeros(∞)]
            A = S \ (a .* S)
            @test A[1:10,1:10] == I
        end

        @testset "Conversion" begin
            A,B = Jacobi(0.25,-0.7), Jacobi(3.25,1.3)
            R = B \ A
            c = [[1,2,3,4,5]; zeros(∞)]
            @test B[0.1,:]' * (R * c) ≈ A[0.1,:]' * c
            R \ c
            Ri = A \ B
            @test Ri[1:10,1:10] ≈ inv(R[1:10,1:10])
            @test A[0.1,:]' * (Ri * c) ≈ B[0.1,:]' * c

            # special weighted conversions
            W = (JacobiWeight(-1,-1) .* Jacobi(0,0)) \ Jacobi(0,0)
            @test ((JacobiWeight(-1,-1) .* Jacobi(0,0)) * W)[0.1,1:10] ≈ Jacobi(0,0)[0.1,1:10]
        end

        @testset "Derivative" begin
            a,b,c = 0.1,0.2,0.3
            S = Jacobi(a,b)
            x = axes(S,1)
            D = Derivative(x)
            u = S \ exp.(x)
            x̃ = 0.1
            @test ((D*S) * u)[x̃] ≈ exp(x̃)
            @test (D * (JacobiWeight(0,b) .* S) * u)[x̃] ≈ exp(x̃) * (1+x̃)^(b-1) * (1+b+x̃)
            @test (D * (JacobiWeight(a,0) .* S) * u)[x̃] ≈ exp(x̃) * (1-x̃)^(a-1) * (1-a-x̃)
            @test (D * (JacobiWeight(a,b) .* S) * u)[x̃] ≈ -exp(x̃)*(1-x̃)^(-1+a)*(1+x̃)^(-1+b)*(a*(1+x̃)+(-1+x̃)*(1+b+x̃))
            @test (D * (JacobiWeight(0,c) .* S) * u)[x̃] ≈ exp(x̃) * (1+x̃)^(c-1) * (1+c+x̃)
            @test (D * (JacobiWeight(c,0) .* S) * u)[x̃] ≈ exp(x̃) * (1-x̃)^(c-1) * (1-c-x̃)
            @test (D * (JacobiWeight(0,0) .* S) * u)[x̃] ≈ exp(x̃)
            @test (D * (JacobiWeight(c,a) .* S) * u)[x̃] ≈ -exp(x̃) * (1-x̃)^(-1+c)*(1+x̃)^(-1+a)*(c*(1+x̃)+(-1+x̃)*(1+a+x̃))

            P = Jacobi(0,0)

            h = 1E-8
            @test (D * (JacobiWeight(c,a) .* S))[0.1,1:5] ≈ ((JacobiWeight(c,a) .* S)[0.1+h,1:5]-(JacobiWeight(c,a) .* S)[0.1,1:5])/h atol=1E-5
            @test (D * (JacobiWeight(c,a) .* P))[0.1,1:5] ≈ ((JacobiWeight(c,a) .* P)[0.1+h,1:5]-(JacobiWeight(c,a) .* P)[0.1,1:5])/h atol=1E-5
            @test (D * (JacobiWeight(c,a) .* P))[0.1,1:5] ≈ (D * (JacobiWeight(c,a) .* Legendre()))[0.1,1:5]
        end

        @testset "grammatrix" begin
            W = Weighted(jacobi(1,1,0..1))
            M = grammatrix(W)
            @test M[1:10,1:10] == grammatrix(Weighted(Jacobi(1,1)))[1:10,1:10]/2
        end
    end

    @testset "functions" begin
        b,a = 0.1,0.2
        P = Jacobi(a,b)
        D = Derivative(axes(P,1))

        f = P*Vcat(randn(10), Zeros(∞))
        @test (Jacobi(a,b+1) * (Jacobi(a,b+1)\f))[0.1] ≈ f[0.1]
        h = 0.0000001
        @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=100h

        (D*(JacobiWeight(a,b) .* f))
    end

    @testset "expansions" begin
        P = Jacobi(1/2,0.)
        x = axes(P,1)
        @test (P * (P \ exp.(x)))[0.1] ≈ exp(0.1)
        @test P[:,1:20] \ exp.(x) ≈ (P \ exp.(x))[1:20]

        @test P[0.1,:]' * (P \ [exp.(x) cos.(x)]) ≈ [exp(0.1) cos(0.1)]


        wP = Weighted(Jacobi(1/2,0.))
        f = @.(sqrt(1 - x) * exp(x))
        @test wP[0.1,1:100]'*(wP[:,1:100] \ f) ≈ sqrt(1-0.1) * exp(0.1)
        @test (wP * (wP \ f))[0.1] ≈ sqrt(1-0.1) * exp(0.1)

        P̃ = P[affine(Inclusion(0..1), x), :]
        x̃ = axes(P̃, 1)
        @test (P̃ * (P̃ \ exp.(x̃)))[0.1] ≈ exp(0.1)
        wP̃ = wP[affine(Inclusion(0..1), x), :]
        f̃ = @.(sqrt(1 - x̃) * exp(x̃))
        @test wP̃[0.1,1:100]'*(wP̃[:,1:100] \ f̃) ≈ sqrt(1-0.1) * exp(0.1)
        @test (wP̃ * (wP̃ \ f̃))[0.1] ≈ sqrt(1-0.1) * exp(0.1)

        @testset "bug" begin
            P = jacobi(0,-1/2,0..1)
            x = axes(P,1)
            u = P * (P \ exp.(x))
            @test u[0.1] ≈ exp(0.1)
            U = P * (P \ [exp.(x) cos.(x)])
            @test U[0.1,:] ≈ [exp(0.1),cos(0.1)]
        end

        @testset "special cases" begin
            P = Jacobi(0.1,0.2)
            x = axes(P,1)
            @test P \ zero(x) isa Zeros
            @test P \ one(x) == [1; Zeros(∞)]
            @test P \ x ≈ P \ broadcast(x -> x, x)
        end

        @testset "Weighted addition" begin
            a = Weighted(Jacobi(2,2)) * [1; 2; zeros(∞)]
            b = Weighted(Jacobi(1,1)) * [3; 4; 5; zeros(∞)]
            c = (JacobiWeight(2,2) .* Jacobi(1,1)) * [6; zeros(∞)]
            @test (a + b)[0.1] ≈ a[0.1] + b[0.1]
            @test (a + c)[0.1] ≈ (c + a)[0.1] ≈ a[0.1] + c[0.1]
        end
    end

    @testset "trivial weight" begin
        S = JacobiWeight(0.0,0.0) .* Jacobi(0.0,0.0)
        @test S == S
        @test Legendre() == S
        @test Legendre()\S isa Eye
    end

    @testset "Jacobi integer" begin
        S = Jacobi(true,true)
        D = Derivative(axes(S,1))
        P = Legendre()

        @test pinv(pinv(S)) === S
        @test P\P === pinv(P)*P === Eye(∞)

        Bi = pinv(Jacobi(2,2))
        @test Bi isa QuasiArrays.PInvQuasiMatrix

        A = Jacobi(2,2) \ (D*S)
        @test typeof(A) == typeof(pinv(Jacobi(2,2))*(D*S))
        @test A isa BandedMatrix
        @test bandwidths(A) == (-1,1)
        @test size(A) == (∞,∞)
        @test A[1:10,1:10] == diagm(1 => 2:0.5:6)

        M = @inferred(D*S)
        @test M isa MulQuasiMatrix
        @test M.args[1] == Jacobi(2,2)
        @test M.args[2][1:10,1:10] == A[1:10,1:10]
    end

    @testset "Weighted Jacobi integer" begin
        S = Jacobi(true,true)
        w̃ = JacobiWeight(false,true)
        A = Jacobi(true,false)\(w̃ .* S)
        @test A isa LazyBandedMatrices.Bidiagonal
        @test size(A) == (∞,∞)
        @test A[1:10,1:10] ≈ (Jacobi(1.0,0.0) \ (JacobiWeight(0.0,1.0) .* Jacobi(1.0,1.0)))[1:10,1:10]

        w̄ = JacobiWeight(true,false)
        A = Jacobi(false,true)\(w̄.*S)
        @test A isa LazyBandedMatrices.Bidiagonal
        @test size(A) == (∞,∞)
        @test A[1:10,1:10] ≈ (Jacobi(0.0,1.0) \ (JacobiWeight(1.0,0.0) .* Jacobi(1.0,1.0)))[1:10,1:10]

        P = Legendre()
        w̄ = JacobiWeight(true,false)
        w̄ = JacobiWeight(false,true)
        @test (P \ (w̃ .* Jacobi(false,true)))[1:10,1:10] == diagm(0 => ones(10), -1 => ones(9))

        w = JacobiWeight(true,true)
        A,B = (P'P),P\(w.*S)

        M = Mul(A,B)
        @test M[1,1] == 4/3

        M = ApplyMatrix{Float64}(*,A,B)
        M̃ = M[1:10,1:10]
        @test M̃ isa BandedMatrix
        @test bandwidths(M̃) == (2,0)

        @test A*B isa MulMatrix
        @test bandwidths(A*B) == bandwidths(B)

        A,B,C = (P\(w.*S))',(P'P),P\(w.*S)
        M = ApplyArray(*,A,B,C)
        @test bandwidths(M) == (2,2)
        @test M[1,1] ≈  1+1/15
        M = A*B*C
        @test bandwidths(M) == (2,2)
        @test M[1,1] ≈  1+1/15

        S = Jacobi(1.0,1.0)
        w = JacobiWeight(1.0,1.0)
        wS = w .* S

        W = QuasiDiagonal(w)
        @test W[0.1,0.2] ≈ 0.0
    end

    @testset "Jacobi and Chebyshev" begin
        T = ChebyshevT()
        U = ChebyshevU()
        JT = Jacobi(T)
        JU = Jacobi(U)

        @testset "recurrence degenerecies" begin
            A,B,C = recurrencecoefficients(JT)
            @test A[1] == 0.5
            @test B[1] == 0.0
        end

        @test JT[0.1,1:4] ≈ [1.0,0.05,-0.3675,-0.0925]

        @test ((T \ JT) * (JT \ T))[1:10,1:10] ≈ Eye(10)
        @test ((U \ JU) * (JU \ U))[1:10,1:10] ≈ Eye(10)

        @test T[0.1,1:10]' ≈ JT[0.1,1:10]' * (JT \ T)[1:10,1:10]
        @test U[0.1,1:10]' ≈ JU[0.1,1:10]' * (JU \ U)[1:10,1:10]

        w = JacobiWeight(1,1)
        @test (w .* U)[0.1,1:10] ≈ (T * (T \ (w .* U)))[0.1,1:10]
        @test (w .* U)[0.1,1:10] ≈ (JT * (JT \ (w .* U)))[0.1,1:10]
        @test (w .* JU)[0.1,1:10] ≈ (T * (T \ (w .* JU)))[0.1,1:10]
        @test (w .* JU)[0.1,1:10] ≈ (JT * (JT \ (w .* JU)))[0.1,1:10]
    end

    @testset "Jacobi-Chebyshev-Ultraspherical transforms" begin
        @test Jacobi(0.0,0.0) \ Legendre() == Eye(∞)
        @test ((Ultraspherical(3/2) \ Jacobi(1,1))*(Jacobi(1,1) \ Ultraspherical(3/2)))[1:10,1:10] ≈ Eye(10)
        f = Jacobi(0.0,0.0)*[[1,2,3]; zeros(∞)]
        g = (Legendre() \ f) - coefficients(f)
        @test_skip norm(g) ≤ 1E-15
        @test (Legendre() \ f) == coefficients(f)
        @test (Legendre() \ f)[1:10] ≈ coefficients(f)[1:10]
        f = Jacobi(1.0,1.0)*[[1,2,3]; zeros(∞)]
        g = Ultraspherical(3/2)*(Ultraspherical(3/2)\f)
        @test f[0.1] ≈ g[0.1]

        @testset "Chebyshev-Legendre" begin
            T = Chebyshev()
            P = Legendre()
            @test T[:,Base.OneTo(5)] \ P[:,Base.OneTo(5)] == (T\P)[1:5,1:5]

            x = axes(P,1)
            u = P * (P \ exp.(x))
            @test u[0.1] ≈ exp(0.1)

            P = Legendre{BigFloat}()
            x = axes(P,1)
            u = P * (P \ exp.(x))
            @test u[BigFloat(1)/10] ≈ exp(BigFloat(1)/10)
        end
    end

    @testset "hcat" begin
        L = LinearSpline(range(-1,1;length=2))
        S = JacobiWeight(1.0,1.0) .* Jacobi(1.0,1.0)
        P = apply(hcat,L,S)
        @test P isa ApplyQuasiArray
        @test axes(P) == axes(S)
        V = view(P,0.1,1:10)
        # @test all(arguments(V) .≈ [L[0.1,:], S[0.1,1:8]])
        @test P[0.1,1:10] == [L[0.1,:]; S[0.1,1:8]]
        D = Derivative(axes(P,1))
        # applied(*,D,P) |> typeof
        # MemoryLayout(typeof(D))
    end

    @testset "Jacobi Clenshaw" begin
        P = Jacobi(0.1,0.2)
        x = axes(P,1)
        a = P * (P \ exp.(x))
        @test a[0.1] ≈ exp(0.1)
        M = P \ (a .* P);
        u = [randn(1000); zeros(∞)];
        @test (P * (M*u))[0.1] ≈ (P*u)[0.1]*exp(0.1)
    end

    @testset "mapped" begin
        R = jacobi(0,1/2,0..1) \ jacobi(0,-1/2,0..1)
        R̃ = Jacobi(0,1/2) \ Jacobi(0,-1/2)
        @test R[1:10,1:10] == R̃[1:10,1:10]
    end

    @testset "Christoffel–Darboux" begin
        a,b = 0.1,0.2
        P = Jacobi(a,b)
        X = P\ (axes(P,1) .* P)
        Mi = inv(P'*(JacobiWeight(a,b) .* P))
        x,y = 0.1,0.2
        n = 10
        Pn = Diagonal([Ones(n); Zeros(∞)])
        Min = Pn * Mi
        @test norm((X*Min - Min*X')[1:n,1:n]) ≤ 1E-13
        β = X[n,n+1]*Mi[n+1,n+1]
        @test (x-y) * P[x,1:n]'Mi[1:n,1:n]*P[y,1:n] ≈ P[x,n:n+1]' * (X*Min - Min*X')[n:n+1,n:n+1] * P[y,n:n+1] ≈ P[x,n:n+1]' * [0 -β; β 0] * P[y,n:n+1]

        @testset "extrapolation" begin
            x,y = 0.1,3.4
            @test (x-y) * P[x,1:n]'Mi[1:n,1:n]*Base.unsafe_getindex(P,y,1:n) ≈ P[x,n:n+1]' * [0 -β; β 0] * Base.unsafe_getindex(P,y,n:n+1)
        end
    end

    @testset "special syntax" begin
        @test jacobip.(0:5, 0.1, 0.2, 0.3) == Jacobi(0.1, 0.2)[0.3, 1:6]
        @test normalizedjacobip.(0:5, 0.1, 0.2, 0.3) == Normalized(Jacobi(0.1, 0.2))[0.3, 1:6]
    end

    @testset "Weighted/HalfWeighted" begin
        x = axes(Legendre(),1)
        D = Derivative(x)
        a,b = 0.1,0.2
        B = Jacobi(a,b)
        A = Jacobi(a-1,b-1)
        D_W = Weighted(A) \ (D * Weighted(B))
        @test (A * (D_W * (B \ exp.(x))))[0.1] ≈ (-a*(1+0.1) + b*(1-0.1) + (1-0.1^2)) *exp(0.1)

        @test copy(HalfWeighted{:a}(Jacobi(a,b))) == HalfWeighted{:a}(Jacobi(a,b))

        D_a = HalfWeighted{:a}(Jacobi(a-1,b+1)) \ (D * HalfWeighted{:a}(B))
        D_b = HalfWeighted{:b}(Jacobi(a+1,b-1)) \ (D * HalfWeighted{:b}(B))
        @test (Jacobi(a-1,b+1) * (D_a * (B \ exp.(x))))[0.1] ≈ (-a + 1-0.1) *exp(0.1)
        @test (Jacobi(a+1,b-1) * (D_b * (B \ exp.(x))))[0.1] ≈ (b + 1+0.1) *exp(0.1)

        @test HalfWeighted{:a}(B) \ HalfWeighted{:a}(B) isa Eye
        @test HalfWeighted{:a}(B) \ (JacobiWeight(a,0) .* B) isa Eye

        @test HalfWeighted{:a}(B) \ (x .* HalfWeighted{:a}(B)) isa LazyBandedMatrices.Tridiagonal

        @test (D * HalfWeighted{:a}(Normalized(B)) * (Normalized(B) \ exp.(x)))[0.1] ≈ (-a + 1-0.1)*(1-0.1)^(a-1) *exp(0.1)
        @test (D * HalfWeighted{:b}(Normalized(B)) * (Normalized(B) \ exp.(x)))[0.1] ≈ (b + 1+0.1) * (1+0.1)^(b-1)*exp(0.1)

        @test (D * Weighted(Jacobi(0,0.1)))[0.1,1:10] ≈ (D * HalfWeighted{:b}(Jacobi(0,0.1)))[0.1,1:10]
        @test (D * Weighted(Jacobi(0.1,0)))[0.1,1:10] ≈ (D * HalfWeighted{:a}(Jacobi(0.1,0)))[0.1,1:10]

        @test HalfWeighted{:a}(Jacobi(0.2,0.1)) ≠ HalfWeighted{:b}(Jacobi(0.2,0.1))
        @test HalfWeighted{:a}(Jacobi(0.2,0.1)) == HalfWeighted{:a}(Jacobi(0.2,0.1))

        # @test convert(WeightedOrthogonalPolynomial, HalfWeighted{:a}(Normalized(Jacobi(0.1,0.2))))[0.1,1:10] ≈
        #     HalfWeighted{:a}(Normalized(Jacobi(0.1,0.2)))[0.1,1:10]
        # @test convert(WeightedOrthogonalPolynomial, HalfWeighted{:b}(Normalized(Jacobi(0.1,0.2))))[0.1,1:10] ≈
        #     HalfWeighted{:b}(Normalized(Jacobi(0.1,0.2)))[0.1,1:10]

        
        L = Normalized(Jacobi(0, 0)) \ HalfWeighted{:a}(Normalized(Jacobi(1, 0)))
        @test Normalized(Jacobi(0, 0))[0.1,1:11]'*L[1:11,1:10] ≈ HalfWeighted{:a}(Normalized(Jacobi(1, 0)))[0.1,1:10]'
        L = Jacobi(0, 0) \ HalfWeighted{:a}(Normalized(Jacobi(1, 0)))
        @test Jacobi(0, 0)[0.1,1:11]'*L[1:11,1:10] ≈ HalfWeighted{:a}(Normalized(Jacobi(1, 0)))[0.1,1:10]'
        L = Normalized(Jacobi(0, 0)) \ HalfWeighted{:a}(Jacobi(1, 0))
        @test Normalized(Jacobi(0, 0))[0.1,1:11]'*L[1:11,1:10] ≈ HalfWeighted{:a}(Jacobi(1, 0))[0.1,1:10]'

        @testset "different weighted" begin
            L = Weighted(Jacobi(0,0)) \ Weighted(Jacobi(1,1))
            @test L[1:10,1:10] ≈ (Legendre() \ Weighted(Jacobi(1,1)))[1:10,1:10]
            @test Weighted(Jacobi(0,0)) \ Legendre() == Eye(∞)
            @test Weighted(Jacobi(0,0)) \ (Legendre() * [1; zeros(∞)]) ≈ [1; zeros(∞)]
        end

        @testset "Mapped" begin
            x = Inclusion(0..1)
            W = Weighted(jacobi(1,1,0..1))
            D = Derivative(x)
            P¹ = Jacobi(1,1)
            @test (D * W)[0.1,1:10] ≈ (D * Weighted(P¹)[affine(x,axes(P¹,1)),:])[0.1,1:10]
        end

        @testset "==" begin
            @test HalfWeighted{:a}(Jacobi(1,2)) == JacobiWeight(1,0) .* Jacobi(1,2)
            @test JacobiWeight(1,0) .* Jacobi(1,2) == HalfWeighted{:a}(Jacobi(1,2))
            @test HalfWeighted{:a}(Jacobi(0,2)) == Jacobi(0,2)
            @test Jacobi(0,2) == HalfWeighted{:a}(Jacobi(0,2))
            @test HalfWeighted{:b}(Jacobi(2,0)) == Jacobi(2,0)
            @test Jacobi(2,0) == HalfWeighted{:b}(Jacobi(2,0))
        end

        @testset "Weighted(Normalized)" begin
            w = JacobiWeight(1,1)
            P = Jacobi(1,1)
            Q = Normalized(P)

            @test w .* Q == w .* Q

            @test Weighted(Q) \ Weighted(P) isa Diagonal
            @test Weighted(P) \ Weighted(Q) isa Diagonal
            @test Weighted(Q) \ Weighted(Q) isa Diagonal

            @test (w .* Q) \ (w .* P) isa Diagonal
            @test (w .* P) \ (w .* Q) isa Diagonal
            @test (w .* Q) \ (w .* Q) isa Diagonal

            @test (w .* Q) \ Weighted(P) isa Diagonal
            @test (w .* P) \ Weighted(Q) isa Diagonal
            @test (w .* Q) \ Weighted(Q) isa Diagonal
            @test Weighted(P) \ (w .* Q) isa Diagonal
            @test Weighted(Q) \ (w .* P) isa Diagonal
            @test Weighted(Q) \ (w .* Q) isa Diagonal
        end
    end
end