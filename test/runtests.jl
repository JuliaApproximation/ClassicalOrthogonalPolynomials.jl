using Base, ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, FillArrays,
        LazyArrays, BandedMatrices, LinearAlgebra, FastTransforms, IntervalSets,
        InfiniteLinearAlgebra, Random, Test
using ForwardDiff, SemiseparableMatrices, SpecialFunctions, LazyBandedMatrices
import ContinuumArrays: BasisLayout, MappedBasisLayout
import ClassicalOrthogonalPolynomials: jacobimatrix, ∞, ChebyshevInterval, LegendreWeight,
            Clenshaw, forwardrecurrence!, singularities
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
include("test_stieltjes.jl")
include("test_roots.jl")
include("test_decompOPs.jl")

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