using ClassicalOrthogonalPolynomials, LazyArrays, QuasiArrays, BandedMatrices, ContinuumArrays, ForwardDiff, Test
import ClassicalOrthogonalPolynomials: recurrencecoefficients, jacobimatrix, Clenshaw, weighted, oneto, grammatrix
import QuasiArrays: MulQuasiArray

@testset "Legendre" begin
    @testset "LegendreWeight" begin
        w = LegendreWeight()
        @test AbstractQuasiArray{BigFloat}(w) ≡ AbstractQuasiVector{BigFloat}(w) ≡ LegendreWeight{BigFloat}()


        @test w.^2 isa LegendreWeight
        @test sqrt.(w) isa LegendreWeight
        @test w .* w isa LegendreWeight
        @test w[0.1] ≡ 1.0
    end

    @testset "basics" begin
        P = Legendre()

        @test AbstractQuasiArray{BigFloat}(P) ≡ AbstractQuasiMatrix{BigFloat}(P) ≡ Legendre{BigFloat}()

        @test axes(P) == (Inclusion(ChebyshevInterval()),oneto(∞))
        @test P == P == Legendre{Float32}()
        @test weighted(P) == P
        A,B,C = recurrencecoefficients(P)
        @test B isa Zeros
        P = Jacobi(0.0,0.0)
        A,B,C = recurrencecoefficients(P)
        @test B[1] == 0.0
        x = axes(P,1)
        X = P \ (x .* P)
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
            @test B[2] ≈ X[2,2]/X[3,2]
            @test C[3] ≈ X[2,3]/X[4,3]
        end
    end

    @testset "expansion" begin
        P = Legendre()
        x = axes(P,1)
        @test (P \ [exp.(x) cos.(x)])[1:10,1:2] ≈ [P\exp.(x) P\cos.(x)][1:10,:]

        # plan_transform for rect

        X = randn(10, 11)
        F1 = plan_transform(P, X, 1)
        @test size(F1) == (10,11)
        p1 = plan_transform(P, X[:,1])
        @test F1*X ≈ hcat([p1 * X[:,j] for j = 1:size(X,2)]...)
        F2 = plan_transform(P, X, 2)
        p2 = plan_transform(P, X[1,:])
        @test F2*X ≈ vcat([(p2 * X[k,:])' for k = 1:size(X,1)]...)
        F = plan_transform(P, X)
        @test F*X ≈ F2*(F1*X)
        @test F\(F*X) ≈ X
    end

    @testset "operators" begin
        P = Legendre()
        P̃ = Jacobi(0.0,0.0)
        P̄ = Ultraspherical(1/2)

        @test jacobimatrix(P̃)[1,1] == 0.0
        @test jacobimatrix(P̃)[1:10,1:10] == jacobimatrix(P)[1:10,1:10] == jacobimatrix(P̄)[1:10,1:10]

        @test P[0.1,Base.OneTo(4)] ≈ [1,0.1,-0.485,-0.1475]
        @test P̃[0.1,1:10] ≈ P[0.1,1:10] ≈ P̄[0.1,1:10]

        @test Ultraspherical(P) == P̄
        @test Jacobi(P) == P̃

        @test P̃\P === P\P̃ === P̄\P === P\P̄ === Eye(∞)
        @test_broken P̄\P̃ === P̃\P̄ === Eye(∞)

        D = Derivative(axes(P,1))
        @test Ultraspherical(3/2)\(D*P) isa BandedMatrix{Float64,<:Ones}
        @test (Ultraspherical(5/2) \ (D^2*P))[1:10,1:10] == BandedMatrix(2 => Fill(3,8))

        P = Legendre()
        x = axes(P,1)
        w = x .+ x.^2 .+ 1
        W = P \ (w .* P)
        @test W isa Clenshaw
        @test W * [1; 2; zeros(∞)] ≈ P \ (w .* (P[:,1:2] * [1,2]))

        f = expand(P, exp)
        @test (w .* f)[0.1] ≈ w[0.1]exp(0.1)

        M = P'P
        @test M isa Diagonal
        @test P'x ≈ [0; 2/3; zeros(∞)]
        @test P'f ≈ M * (P\f)

        v = f.args[2]
        @test v'*M isa Adjoint
        @test v'*M*v ≈ dot(f,f) ≈ f'f
    end

    @testset "test on functions" begin
        P = Legendre()
        D = Derivative(axes(P,1))
        f = P*Vcat(randn(10), Zeros(∞))
        P̃ = Jacobi(0.0,0.0)
        P̄ = Ultraspherical(1/2)
        @test (P̃*(P̃\f))[0.1] ≈ (P̄*(P̄\f))[0.1] ≈ f[0.1]
        C = Ultraspherical(3/2)
        @test (C*(C\f))[0.1] ≈ f[0.1]

        @test (D*f)[0.1] ≈ ForwardDiff.derivative(x -> (Legendre{eltype(x)}()*f.args[2])[x],0.1)
    end

    @testset "poly broadcast" begin
        P = Legendre()
        x = axes(P,1)

        @test (x .* P) isa MulQuasiArray

        J = P \ (x .* P)
        @test (P \ (   (1 .+ x) .* P))[1:10,1:10] ≈ (I + J)[1:10,1:10]

        x = Inclusion(0..1)
        Q = P[2x.-1,:]
        @test x .* Q isa MulQuasiArray
        @test Q \ (x .* Q) isa LazyBandedMatrices.Tridiagonal
    end

    @testset "sum" begin
        P = legendre()
        x = axes(P,1)
        w = P * (P \ exp.(x))
        @test sum(w) ≈ ℯ - inv(ℯ)
        @test sum(P[:,5]) == 0
    end

    @testset "Mapped" begin
        P = legendre(0..1)
        @test weighted(P) == P
        @test weighted(Normalized(Legendre())[parentindices(P)...]) == Normalized(Legendre())[parentindices(P)...]
        x = axes(P,1)
        X = jacobimatrix(P)
        @test X[1:10,1:10] ≈ (P \ (x .* P))[1:10,1:10]
        @test X[band(1)][1:10] ≈ [X[k,k+1] for k=1:10]
        @test X[band(0)][1:10] ≈ [X[k,k] for k=1:10]
        @test X[band(-1)][1:10] ≈ [X[k+1,k] for k=1:10]
        A,B,C = recurrencecoefficients(P)
        @test P[0.1,1] == 1
        @test P[0.1,2] ≈ A[1]*0.1 + B[1]
        @test P[0.1,3] ≈ (A[2]*0.1 + B[2])*P[0.1,2] - C[2]
        @test P[0.1,4] ≈ (A[3]*0.1 + B[3])*P[0.1,3] - C[3]*P[0.1,2]
    end

    @testset "Christoffel–Darboux" begin
        P = Legendre()
        X = P\ (axes(P,1) .* P)
        Mi = inv(P'P)
        x,y = 0.1,0.2
        n = 10
        Pn = Diagonal([Ones(n); Zeros(∞)])
        Min = Pn * Mi
        @test (X*Min - Min*X')[1:n,1:n] ≈ zeros(n,n)
        @test (x-y) * P[x,1:n]'Mi[1:n,1:n]*P[y,1:n] ≈ P[x,n:n+1]' * (X*Min - Min*X')[n:n+1,n:n+1] * P[y,n:n+1]
        β = X[n,n+1]*Mi[n+1,n+1]
        @test (x-y) * P[x,1:n]'Mi[1:n,1:n]*P[y,1:n] ≈ P[x,n:n+1]' * [0 -β; β 0] * P[y,n:n+1]
    end

    @testset "special syntax" begin
        @test legendrep.(0:5, 0.3) == Legendre()[0.3, 1:6]
    end

    @testset "Inner products" begin
        x = Inclusion(ChebyshevInterval())
        @test x'exp.(x) ≈ 2/ℯ
    end

    @testset "Heaviside and Legendre" begin
        @test Legendre() \ HeavisideSpline([-1,1]) == Vcat(1,Zeros(∞,1))
    end

    @testset "Normalized" begin
        P̃ = Normalized(Legendre())
        @test P̃'P̃ ≡ grammatrix(P̃) ≡ Eye(∞)
    end

    @testset "log sum/diff" begin
    	P = Legendre()
        z = 2 + im
        f = expand(P, exp)
        x = axes(f,1)
        @test sum(log.(z .- x) .* f) ≈ sum(expand(Legendre{ComplexF64}(), x -> log(z - x)*exp(x)))
        @test diff(log.(z .- x) .* P)[0.1,1:5] ≈ log.(z .- 0.1) .* diff(P)[0.1,1:5] - P[0.1,1:5] ./ (z .- 0.1)
        @test diff(log.(z .- x) .* f)[0.1] ≈ log(z - 0.1) * exp(0.1) - exp(0.1) / (z - 0.1)
    end

    @testset "type stable expansion" begin
        P = Legendre()
        T = ChebyshevT()
        @test @inferred(T \ P[:,1]) == Vcat([1], Zeros(∞))
        @test @inferred(P \ P[:,1]) == Vcat([1], Zeros(∞))
    end

    @testset "higher order diff" begin
        P = Legendre()
        P¹ = Ultraspherical(3/2)
        P² = Ultraspherical(5/2)
        P³ = Ultraspherical(7/2)
        @test (P¹ \ diff(P,1))[1:10,1:10] == (P¹ \ diff(P))[1:10,1:10]
        @test (P² \ diff(P,2))[1:10,1:10] ≈ (P² \ diff(diff(P)))[1:10,1:10]
        @test (P³ \ diff(P,3))[1:10,1:10] ≈ (P³ \ diff(diff(diff(P))))[1:10,1:10]

        # cover back compatibilty
        @test_throws MethodError ClassicalOrthogonalPolynomials.basis_singularities(nothing, nothing)
    end

    @testset "singularities expand" begin
        x = Inclusion(1..2)
        @test basis(expand(fill(2, x))) == basis(expand(broadcast(+, exp.(x), cos.(x), x))) == legendre(x) 
    end
    @testset "ChebyshevInterval constructor" begin
        @test legendre(ChebyshevInterval()) ≡ Legendre()
    end

    @testset "cumsum" begin
        P = Legendre()
        x = axes(P,1)
        @test (P \ cumsum(P)) * (P \ exp.(x)) ≈ P \ (exp.(x) .- exp(-1))
        f = P * (P \ exp.(x))
        @test P \ cumsum(f) ≈ P \ (exp.(x) .- exp(-1))
    end
end