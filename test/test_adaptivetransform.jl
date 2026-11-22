using ClassicalOrthogonalPolynomials, Test
using ContinuumArrays: ContinuumArrays, basis, coefficients
using LazyArrays: paddeddata

@testset "Adaptive two-dimensional transforms" begin
    T = ChebyshevT()
    F = [exp(x*cos(y)) for x in -1..1, y in -1..1]
    C = T \ F / T'
    @test C isa AbstractMatrix
    @test axes(C) == (axes(T,2), axes(T,2))
    @test size(paddeddata(C)) isa NTuple{2,Int}
    @test iszero(C[1000,1000])
    @test (T*C*T')[0.23,-0.71] ≈ exp(0.23*cos(-0.71))

    @testset "Mapped intervals" begin
        P, Q = basis(-1..1), basis(0..1)
        F = [exp(x*cos(y)) for x in -1..1, y in 0..1]
        C = P \ F / Q'
        explicit = P*C*Q'
        x, y = [-0.4, 0.23], [0.12, 0.71]
        @test axes(C) == (axes(P,2), axes(Q,2))
        @test size(paddeddata(C)) isa NTuple{2,Int}
        @test explicit[x,y] ≈ [exp(s*cos(t)) for s in x, t in y]

        # ContinuumArrays versions with basis_expand can infer a pair of bases.
        if isdefined(ContinuumArrays, :basis_expand)
            E = expand(F)
            @test basis(E) == (P, Q)
            @test axes(E) == axes(F)
            @test paddeddata(coefficients(E)) ≈ paddeddata(C)
            @test E[x,y] ≈ explicit[x,y]
        end
    end

    # The degree-32 mode aliases to a constant on the initial 16-point grid.
    P = Legendre()
    G = [im*x*cos(32acos(y)) for x in -1..1, y in -1..1]
    D = P \ G / T'
    expected = zeros(ComplexF64, 2, 33)
    expected[2,33] = im
    @test D[1:2,1:33] ≈ expected
    @test (P \ (G / T'))[1:2,1:33] ≈ expected
    @test size(paddeddata(D), 1) < size(paddeddata(D), 2)

    Z = T \ [0.0 for x in -1..1, y in -1..1] / T'
    @test isempty(paddeddata(Z))
    @test iszero(Z[1,1])
end
