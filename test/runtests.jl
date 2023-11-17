using Base, ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, FillArrays,
        LazyArrays, BandedMatrices, LinearAlgebra, FastTransforms, IntervalSets,
        InfiniteLinearAlgebra, Random, Test
using ForwardDiff, SemiseparableMatrices, SpecialFunctions, LazyBandedMatrices
import ContinuumArrays: BasisLayout, MappedBasisLayout
import ClassicalOrthogonalPolynomials: jacobimatrix, ∞, ChebyshevInterval, LegendreWeight,
            Clenshaw, forwardrecurrence!, singularities, OrthogonalPolynomial
import LazyArrays: ApplyStyle, colsupport, MemoryLayout, arguments
import SemiseparableMatrices: VcatAlmostBandedLayout
import QuasiArrays: MulQuasiMatrix
import ClassicalOrthogonalPolynomials: oneto
import InfiniteLinearAlgebra: KronTrav, Block
import FastTransforms: clenshaw!

Random.seed!(0)

@testset "singularities" begin
    x = Inclusion(ChebyshevInterval())
    @test singularities(x) == singularities(exp.(x)) == singularities(x.^2) == 
        singularities(x .+ 1) == singularities(1 .+ x) == singularities(x .+ x) == 
        LegendreWeight()
    @test singularities(exp.(x) .* JacobiWeight(0.1,0.2)) == 
        singularities(JacobiWeight(0.1,0.2) .* exp.(x)) ==
        JacobiWeight(0.1,0.2)

    x = Inclusion(0..1)
    @test singularities(x) == singularities(exp.(x)) == singularities(x.^2) == 
        singularities(x .+ 1) == singularities(1 .+ x) == singularities(x .+ x) == 
        legendreweight(0..1)
end

include("test_chebyshev.jl")
include("test_legendre.jl")
include("test_ultraspherical.jl")
include("test_jacobi.jl")
include("test_hermite.jl")
include("test_laguerre.jl")
include("test_fourier.jl")
include("test_odes.jl")
include("test_ratios.jl")
include("test_normalized.jl")
include("test_lanczos.jl")
include("test_interlace.jl")
include("test_choleskyQR.jl")
include("test_roots.jl")

@testset "Auto-diff" begin
    U = Ultraspherical(1)
    C = Ultraspherical(2)

    f = x -> ChebyshevT{eltype(x)}()[x,5]
    @test ForwardDiff.derivative(f,0.1) ≈ 4*U[0.1,4]
    f = x -> ChebyshevT{eltype(x)}()[x,5][1]
    @test ForwardDiff.gradient(f,[0.1]) ≈ [4*U[0.1,4]]
    @test ForwardDiff.hessian(f,[0.1]) ≈ [8*C[0.1,3]]

    f = x -> ChebyshevT{eltype(x)}()[x,1:5]
    @test ForwardDiff.derivative(f,0.1) ≈ [0;(1:4).*U[0.1,1:4]]
end

@testset "basis" begin
    for x in (Inclusion(ChebyshevInterval()), Inclusion(1 .. 2))
        a,b = first(x),last(x)
        @test sum(x) == b-a
        @test sum(x .^ 2) ≈ (b^3 - a^3)/3
        @test sum(exp.(x)) ≈ exp(b) - exp(a)
        @test dot(x, x) ≈ sum(expand(x .^2))
        @test dot(x.^2, x.^2) ≈ sum(expand(x .^4))
        @test dot(exp.(x), x.^2) ≈ sum(expand(x .^2 .* exp.(x)))
        @test dot(x, exp.(x)) ≈ dot(exp.(x), x)
    end

    # A = x .^ (0:2)'
    # sum(A; dims=1)
end

@testset "Incomplete" begin
    struct MyIncompleteJacobi <: ClassicalOrthogonalPolynomials.AbstractJacobi{Float64} end
    @test_throws ErrorException jacobimatrix(MyIncompleteJacobi())
    @test_throws ErrorException plan_transform(MyIncompleteJacobi(), 5)
end