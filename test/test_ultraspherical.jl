using ClassicalOrthogonalPolynomials, ContinuumArrays, BandedMatrices, LazyArrays, ForwardDiff, Test
using LazyArrays: rowsupport, colsupport
using ClassicalOrthogonalPolynomials: grammatrix

@testset "Ultraspherical" begin
    @testset "Conversion" begin
        U = Ultraspherical(1)
        w = UltrasphericalWeight(1)
        @test AbstractQuasiArray{Float32}(U) ≡ AbstractQuasiMatrix{Float32}(U) ≡ Ultraspherical{Float32}(1)
        @test AbstractQuasiArray{Float32}(w) ≡ AbstractQuasiVector{Float32}(w) ≡ UltrasphericalWeight{Float32}(1)
    end
    @testset "Transforms" begin
        U = Ultraspherical(1)
        x = axes(U,1)
        Un = U[:,Base.OneTo(5)]
        @test factorize(Un) isa ContinuumArrays.TransformFactorization
        @test (Un \ x) ≈ [0,0.5,0,0,0]
        @test (U * (U \ exp.(x)))[0.1] ≈ exp(0.1)
    end

    @testset "Operators" begin
        @testset "Lowering" begin
            λ = 1
            wC1 = Weighted(Ultraspherical(λ))
            wC2 = Weighted(Ultraspherical(λ+1))
            L = wC1 \ wC2
            @test L isa BandedMatrix
            @test bandwidths(L) == (2,0)
            @test L[1,1] ≈ 2λ*(2λ+1)/(4λ*(λ+1))
            @test L[3,1] ≈ -2/(4λ*(λ+1))
            u = [randn(10); zeros(∞)]
            @test wC1[0.1,:]'*(L*u) ≈ wC2[0.1,:]'*u
        end

        @testset "Weighted Derivative" begin
            T = Chebyshev()
            wC1 = Weighted(Ultraspherical(1))
            wC2 = Weighted(Ultraspherical(2))
            x = axes(wC2,1)
            D = Derivative(x)

            @test wC1 \ (D*wC2) isa BandedMatrix
            @test (wC1 \ (D*wC2))[1:5,1:5] == BandedMatrix(-1 => [-1.5,-4,-7.5,-12])

            u = wC2 * [randn(5); zeros(∞)]
            @test (D*u)[0.1] ≈ ((D*T) * (T\u))[0.1]
        end

        @testset "Interrelationships" begin
            @testset "Chebyshev–Ultrashperical" begin
                T = ChebyshevT()
                U = ChebyshevU()
                C = Ultraspherical(2)
                D = Derivative(axes(T,1))

                @test C\C === pinv(C)*C === Eye(∞)
                D₀ = U\(D*T)
                D₁ = C\(D*U)
                @test D₁ isa BandedMatrix
                @test (D₁*D₀)[1:10,1:10] == diagm(2 => 4:2:18)
                @test D₁*D₀ isa MulMatrix
                @test bandwidths(D₁*D₀) == (-2,2)

                S₁ = (C\U)[1:10,1:10]
                @test S₁ isa BandedMatrix{Float64}
                @test S₁ == diagm(0 => 1 ./ (1:10), 2=> -(1 ./ (3:10)))

                @test (U\C)[1:10,1:10] ≈ inv((C\U)[1:10,1:10])
                @test (T\C)[1:10,1:10] ≈ inv((C\T)[1:10,1:10])
                @test bandwidths(U\C) == bandwidths(T\C) == (0,∞)
                @test colsupport(U\C,5) == colsupport(T\C,5) == 1:5
                @test rowsupport(U\C,5) == rowsupport(T\C,5) == 5:∞
            end
            @testset "Legendre" begin
                @test Ultraspherical(0.5) \ (UltrasphericalWeight(0.0) .* Ultraspherical(0.5)) == Eye(∞)
                @test Legendre() \ (UltrasphericalWeight(0.0) .* Ultraspherical(0.5)) == Eye(∞)
                @test (Legendre() \ Ultraspherical(1.5))[1:10,1:10] ≈ inv(Ultraspherical(1.5) \ Legendre())[1:10,1:10]
                @test UltrasphericalWeight(LegendreWeight()) == UltrasphericalWeight(1/2)
            end
        end

        @testset "Conversion" begin
            R = Ultraspherical(3.5) \ Ultraspherical(0.5)
            c = [[1,2,3,4,5]; zeros(∞)]
            @test Ultraspherical(3.5)[0.1,:]' * (R * c) ≈ Ultraspherical(0.5)[0.1,:]' * c
            Ri = Ultraspherical(0.5) \ Ultraspherical(3.5)
            @test Ri[1:10,1:10] ≈ inv(R[1:10,1:10])
            @test Ultraspherical(0.5)[0.1,:]' * (Ri * c) ≈ Ultraspherical(3.5)[0.1,:]' * c
        end
    end

    @testset "test on functions" begin
        T = Chebyshev()
        U = Ultraspherical(1)
        D = Derivative(axes(T,1))
        f = T*Vcat(randn(10), Zeros(∞))
        @test (U*(U\f)).args[1] isa Ultraspherical
        @test (U*(U\f))[0.1] ≈ f[0.1]
        @test (D*f)[0.1] ≈ ForwardDiff.derivative(x -> (ChebyshevT{eltype(x)}()*f.args[2])[x],0.1)
    end

    @testset "Evaluation" begin
        C = Ultraspherical(2)
        @test @inferred(C[0.1,Base.OneTo(0)]) == Float64[]
        @test @inferred(C[0.1,Base.OneTo(1)]) == [1.0]
        @test @inferred(C[0.1,Base.OneTo(2)]) == [1.0,0.4]
        @test @inferred(C[0.1,Base.OneTo(3)]) == [1.0,0.4,-1.88]
    end

    @testset "special syntax" begin
        @test ultrasphericalc.(0:5, 2, 0.3) == Ultraspherical(2)[0.3, 1:6]
    end

    @testset "Ultraspherical vs Chebyshev and Jacobi" begin
        @test Ultraspherical(1) == ChebyshevU()
        @test ChebyshevU() == Ultraspherical(1)
        @test Ultraspherical(0) ≠ ChebyshevT()
        @test ChebyshevT() ≠ Ultraspherical(0)
        @test Ultraspherical(1) ≠ Jacobi(1/2,1/2)
        @test Jacobi(1/2,1/2) ≠ Ultraspherical(1)
        @test Ultraspherical(1/2) == Jacobi(0,0)
        @test Ultraspherical(1/2) == Legendre()
        @test Jacobi(0,0) == Ultraspherical(1/2)
        @test Legendre() == Ultraspherical(1/2)

        @test Ultraspherical(1/2) \ (JacobiWeight(0,0) .* Jacobi(0,0)) isa Diagonal
        @test (JacobiWeight(0,0) .* Jacobi(0,0)) \ Ultraspherical(1/2) isa Diagonal
    end

    @testset "D^2 * mapped" begin
        T = chebyshevt(0..1)
        C = ultraspherical(2,0..1)
        r = axes(T,1)
        D = Derivative(r)
        D₂ = C \ (D^2 * T)
        # r²D₂ = C \ (r.^2 .* (D^2 * T))

        c = [randn(100); zeros(∞)]
        @test C[0.1,:]'*(D₂ * c) ≈ 4*(Derivative(axes(ChebyshevT(),1))^2 * (ChebyshevT() * c))[2*0.1-1]
        @test_broken C[0.1,:]'*(r²D₂ * c) ≈ 0.1^2 * C[0.1,:]'*(D₂ * c)
    end
  
    @testset "BigFloat" begin
        U = Ultraspherical{BigFloat}(1)
        T = ChebyshevT{BigFloat}()
        x = axes(U,1)
        D = Derivative(x)
        
        @test Weighted(T) \ (D * Weighted(U)) isa BandedMatrix{BigFloat}

        C³ = Ultraspherical{BigFloat}(3)
        c = [1; 2; 3; zeros(BigFloat,∞)]
        @test C³[big(1)/10,:]'*(C³ \ U) * c ≈ U[big(1)/10,:]'c
    end

    @testset "show" begin
        @test stringmime("text/plain",UltrasphericalWeight(1)) == "UltrasphericalWeight(1)"
        @test stringmime("text/plain",Ultraspherical(1)) == "Ultraspherical(1)"
    end

    @testset "grammatrix" begin
        C = Ultraspherical(3/2)
        @test (C'C)[1:5,1:5] == grammatrix(C)[1:5,1:5]
    end

    @testset "Weighted derivative" begin
        T = Chebyshev()
        W = Weighted(T) \ diff(Weighted(Ultraspherical(1)))
        @test W[1:10,1:10] == diagm(-1 => -(1:9))
    end
end