using ClassicalOrthogonalPolynomials
using Test
using ClassicalOrthogonalPolynomials: Monic, _p0, orthogonalityweight, recurrencecoefficients

@testset "Basic definition" begin
    P1 = Legendre()
    P2 = Normalized(P1)
    P3 = Monic(P1)
    @test P3.P == P2
    @test Monic(P3) === P3
    @test axes(P3) == axes(Legendre())
    @test Normalized(P3) === P3.P
    @test _p0(P3) == 1
    @test orthogonalityweight(P3) == orthogonalityweight(P1)
    @test sprint(show, MIME"text/plain"(), P3) == "Monic(Legendre())"
end

@testset "evaluation" begin
    function _pochhammer(x, n)
        y = one(x)
        for i in 0:(n-1)
            y *= (x + i)
        end
        return y
    end
    jacobi_kn = (α, β, n) -> _pochhammer(n + α + β + 1, n) / (2.0^n * factorial(n))
    ultra_kn = (λ, n) -> 2^n * _pochhammer(λ, n) / factorial(n)
    chebt_kn = n -> n == 0 ? 1.0 : 2.0 .^ (n - 1)
    chebu_kn = n -> 2.0^n
    leg_kn = n -> 2.0^n * _pochhammer(1 / 2, n) / factorial(n)
    lag_kn = n -> (-1)^n / factorial(n)
    herm_kn = n -> 2.0^n
    _Jacobi(α, β, x, n) = Jacobi(α, β)[x, n+1] / jacobi_kn(α, β, n)
    _Ultraspherical(λ, x, n) = Ultraspherical(λ)[x, n+1] / ultra_kn(λ, n)
    _ChebyshevT(x, n) = ChebyshevT()[x, n+1] / chebt_kn(n)
    _ChebyshevU(x, n) = ChebyshevU()[x, n+1] / chebu_kn(n)
    _Legendre(x, n) = Legendre()[x, n+1] / leg_kn(n)
    _Laguerre(α, x, n) = Laguerre(α)[x, n+1] / lag_kn(n)
    _Hermite(x, n) = Hermite()[x, n+1] / herm_kn(n)
    Ps = [
        Jacobi(2.0, 5.0) (x, n)->_Jacobi(2.0, 5.0, x, n)
        Ultraspherical(1.7) (x, n)->_Ultraspherical(1.7, x, n)
        ChebyshevT() _ChebyshevT
        ChebyshevU() _ChebyshevU
        Legendre() _Legendre
        Laguerre(1.5) (x, n)->_Laguerre(1.5, x, n)
        Hermite() _Hermite
    ]
    for (P, _P) in eachrow(Ps)
        Q = Monic(P)
        @test Q[0.2, 1] == 1.0
        @test Q[0.25, 2] ≈ _P(0.25, 1)
        @test Q[0.17, 3] ≈ _P(0.17, 2)
        @test Q[0.4, 17] ≈ _P(0.4, 16)
        @test Q[0.9, 21] ≈ _P(0.9, 20)
        # @inferred Q[0.2, 5] # no longer inferred
    end
end