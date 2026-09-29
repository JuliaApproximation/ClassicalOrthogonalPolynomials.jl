using ClassicalOrthogonalPolynomials, LazyArrays, FillArrays, Test
using ClassicalOrthogonalPolynomials: ConnectionMatrix
using LazyArrays: colsupport, rowsupport

@testset "ConnectionMatrix" begin
    T = ChebyshevT()
    P = Legendre()
    n = 10
    x = ClassicalOrthogonalPolynomials.chebyshevpoints(Float64, n, Val(1))
    ref(A, B) = A[x,1:n] \ B[x,1:n]

    for (A, B) in ((T, P), (P, T), (T, Ultraspherical(1.5)), (Ultraspherical(1.5), T),
                   (Ultraspherical(0.25), Ultraspherical(1.5)), (P, Ultraspherical(0.25)),
                   (Jacobi(0.1,0.2), Jacobi(1.1,2.3)), (T, Jacobi(0.1,0.2)), (Jacobi(0.1,0.2), T))
        C = A \ B
        @test C isa ConnectionMatrix
        @test C[1:n,1:n] ≈ ref(A, B)
        @test [C[k,j] for k=1:n, j=1:n] ≈ ref(A, B)
        @test C[3:5,2:7] ≈ ref(A, B)[3:5,2:7]
        @test C[:,4][1:n] ≈ ref(A, B)[:,4]
        @test colsupport(C, 4) == 1:4
        @test rowsupport(C, 4) == 4:∞

        c = [randn(5); zeros(∞)]
        @test C * c isa LazyArrays.ApplyArray{Float64,1,typeof(vcat)}
        @test (C * c)[1:n] ≈ ref(A, B)[:,1:5] * c[1:5]
        @test (C \ (C * c))[1:5] ≈ c[1:5]
        @test (inv(C) * (C * c))[1:5] ≈ c[1:5]
        M = [randn(5,3); zeros(∞,3)]
        @test (C * M)[1:n,:] ≈ ref(A, B)[:,1:5] * M[1:5,:]
        c = [randn(ComplexF64,5); zeros(∞)]
        @test (C * c)[1:n] ≈ ref(A, B)[:,1:5] * c[1:5]
    end

    @testset "expansions" begin
        f = expand(P, exp)
        @test (T \ f)[1:20] ≈ (T \ expand(T, exp))[1:20]
        f = expand(Ultraspherical(1.5), cos)
        @test (T \ f)[1:20] ≈ (T \ expand(T, cos))[1:20]
        @test (P \ expand(T, exp))[1:20] ≈ (P \ expand(P, exp))[1:20]
        # mapped
        g = expand(legendre(0..1), exp)
        @test (chebyshevt(0..1) \ g)[1:10] ≈ (chebyshevt(0..1) \ expand(chebyshevt(0..1), exp))[1:10]
    end

    @testset "high degree" begin
        N = 5000
        c = [randn(N); zeros(∞)]
        @test ((P \ T) * ((T \ P) * c))[1:N] ≈ c[1:N]
    end
end
