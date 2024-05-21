module ClassicalOrthogonalPolynomials
using LazyBandedMatrices: LazyBandedLayout
using InfiniteArrays: parentindices
using IntervalSets: UnitRange
using ContinuumArrays, QuasiArrays, LazyArrays, FillArrays, BandedMatrices, BlockArrays,
    IntervalSets, DomainSets, ArrayLayouts, SpecialFunctions,
    InfiniteLinearAlgebra, InfiniteArrays, LinearAlgebra, FastGaussQuadrature, FastTransforms, FFTW,
    LazyBandedMatrices, HypergeometricFunctions

import Base: @_inline_meta, axes, getindex, unsafe_getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy, setindex,
                first, last, Slice, size, length, axes, IdentityUnitRange, sum, _sum, cumsum,
                to_indices, tail, getproperty, inv, show, isapprox, summary,
                findall, searchsortedfirst, diff
import Base.Broadcast: materialize, BroadcastStyle, broadcasted, Broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, adjointlayout,
                sub_materialize, arguments, sub_paddeddata, paddeddata, AbstractPaddedLayout, PaddedColumns, resizedata!, LazyVector, ApplyLayout, call,
                _mul_arguments, CachedVector, CachedMatrix, LazyVector, LazyMatrix, axpy!, AbstractLazyLayout, BroadcastLayout,
                AbstractCachedVector, AbstractCachedMatrix, paddeddata, cache_filldata!,
                simplifiable, PaddedArray, converteltype, simplify
import ArrayLayouts: MatMulVecAdd, materialize!, _fill_lmul!, sublayout, sub_materialize, lmul!, ldiv!, ldiv, transposelayout, triangulardata,
                        subdiagonaldata, diagonaldata, supdiagonaldata, mul, rowsupport, colsupport
import LazyBandedMatrices: SymTridiagonal, Bidiagonal, Tridiagonal, unitblocks, BlockRange1, AbstractLazyBandedLayout
import LinearAlgebra: pinv, factorize, qr, adjoint, transpose, dot, mul!, reflectorApply!
import BandedMatrices: AbstractBandedLayout, AbstractBandedMatrix, _BandedMatrix, bandeddata
import FillArrays: AbstractFill, getindex_value, SquareEye
import DomainSets: components
import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle,
                    _getindex, layout_getindex, _factorize, AbstractQuasiArray, AbstractQuasiMatrix, AbstractQuasiVector,
                    AbstractQuasiFill, equals_layout, QuasiArrayLayout, PolynomialLayout, diff_layout

import InfiniteArrays: OneToInf, InfAxes, Infinity, AbstractInfUnitRange, InfiniteCardinal, InfRanges
import InfiniteLinearAlgebra: chop!, chop, pad, choplength, compatible_resize!, partialcholesky!
import ContinuumArrays: Basis, Weight, basis_axes, @simplify, Identity, AbstractAffineQuasiVector, ProjectionFactorization,
    grid, plotgrid, plotgrid_layout, plotvalues_layout, grid_layout, transform_ldiv, TransformFactorization, QInfAxes, broadcastbasis, ExpansionLayout, basismap,
    AffineQuasiVector, AffineMap, AbstractWeightLayout, AbstractWeightedBasisLayout, WeightedBasisLayout, WeightedBasisLayouts, demap, AbstractBasisLayout, BasisLayout,
    checkpoints, weight, unweighted, MappedBasisLayouts, sum_layout, invmap, plan_ldiv, layout_broadcasted, MappedBasisLayout, SubBasisLayout, broadcastbasis_layout,
    plan_grid_transform, plan_transform, MAX_PLOT_POINTS, MulPlan, grammatrix, AdjointBasisLayout, grammatrix_layout, plan_transform_layout, _cumsum
import FastTransforms: Λ, forwardrecurrence, forwardrecurrence!, _forwardrecurrence!, clenshaw, clenshaw!,
                        _forwardrecurrence_next, _clenshaw_next, check_clenshaw_recurrences, ChebyshevGrid, chebyshevpoints, Plan, ScaledPlan, th_cheb2leg

import FastGaussQuadrature: jacobimoment

import BlockArrays: blockedrange, _BlockedUnitRange, unblock, _BlockArray, block, blockindex, BlockSlice, blockvec
import BandedMatrices: bandwidths

export OrthogonalPolynomial, Normalized, LanczosPolynomial,
            Hermite, Jacobi, Legendre, Chebyshev, ChebyshevT, ChebyshevU, ChebyshevInterval, Ultraspherical, Fourier, Laurent, Laguerre,
            HermiteWeight, JacobiWeight, ChebyshevWeight, ChebyshevGrid, ChebyshevTWeight, ChebyshevUWeight, UltrasphericalWeight, LegendreWeight, LaguerreWeight,
            WeightedUltraspherical, WeightedChebyshev, WeightedChebyshevT, WeightedChebyshevU, WeightedJacobi,
            ∞, Derivative, .., Inclusion,
            chebyshevt, chebyshevu, legendre, jacobi, ultraspherical,
            legendrep, jacobip, ultrasphericalc, laguerrel,hermiteh, normalizedjacobip,
            jacobimatrix, jacobiweight, legendreweight, chebyshevtweight, chebyshevuweight, Weighted, PiecewiseInterlace, plan_transform,
            expand, transform


import Base: oneto


include("interlace.jl")
include("standardchop.jl")
include("adaptivetransform.jl")

const WeightedBasis{T, A<:AbstractQuasiVector, B<:Basis} = BroadcastQuasiMatrix{T,typeof(*),<:Tuple{A,B}}
abstract type OrthogonalPolynomial{T} <: Basis{T} end
abstract type AbstractOPLayout <: AbstractBasisLayout end
struct OPLayout <: AbstractOPLayout end
MemoryLayout(::Type{<:OrthogonalPolynomial}) = OPLayout()

Base.isassigned(P::OrthogonalPolynomial, x, n) = (x ∈ axes(P,1)) && (n ∈ axes(P,2))

sublayout(::AbstractOPLayout, ::Type{<:Tuple{<:AbstractAffineQuasiVector,<:Slice}}) = MappedOPLayout()

"""
    MappedOPLayout

represents an OP that is (usually affine) mapped OP
"""
struct MappedOPLayout <: AbstractOPLayout end

"""
    WeightedOPLayout

represents an OP multiplied by its orthogonality weight.
"""
struct WeightedOPLayout{Lay<:AbstractOPLayout} <: AbstractWeightedBasisLayout end

isorthogonalityweighted(::WeightedOPLayout, _) = true
function isorthogonalityweighted(::AbstractWeightedBasisLayout, wS)
    w,S = arguments(wS)
    w == orthogonalityweight(S)
end

isorthogonalityweighted(wS) = isorthogonalityweighted(MemoryLayout(wS), wS)


equals_layout(::MappedOPLayout, ::MappedOPLayout, P, Q) = demap(P) == demap(Q) && basismap(P) == basismap(Q)
equals_layout(::MappedOPLayout, ::MappedBasisLayouts, P, Q) = demap(P) == demap(Q) && basismap(P) == basismap(Q)
equals_layout(::MappedBasisLayouts, ::MappedOPLayout, P, Q) = demap(P) == demap(Q) && basismap(P) == basismap(Q)

broadcastbasis_layout(::typeof(+), ::MappedOPLayout, ::MappedOPLayout, P, Q) = broadcastbasis_layout(+, MappedBasisLayout(), MappedBasisLayout(), P, Q)
broadcastbasis_layout(::typeof(+), ::MappedOPLayout, M::MappedBasisLayout, P, Q) = broadcastbasis_layout(+, MappedBasisLayout(), M, P, Q)
broadcastbasis_layout(::typeof(+), L::MappedBasisLayout, ::MappedOPLayout, P, Q) = broadcastbasis_layout(+, L, MappedBasisLayout(), P, Q)
sum_layout(::MappedOPLayout, A, dims) = sum_layout(MappedBasisLayout(), A, dims)

equals_layout(::AbstractOPLayout, ::AbstractWeightedBasisLayout, _, _) = false # Weighted-Legendre doesn't exist
equals_layout(::AbstractWeightedBasisLayout, ::AbstractOPLayout, _, _) = false # Weighted-Legendre doesn't exist

equals_layout(::WeightedOPLayout, ::WeightedOPLayout, wP, wQ) = unweighted(wP) == unweighted(wQ)
equals_layout(::WeightedOPLayout, ::WeightedBasisLayout, wP, wQ) = unweighted(wP) == unweighted(wQ) && weight(wP) == weight(wQ)
equals_layout(::WeightedBasisLayout, ::WeightedOPLayout, wP, wQ) = unweighted(wP) == unweighted(wQ) && weight(wP) == weight(wQ)
equals_layout(::WeightedBasisLayout{<:AbstractOPLayout}, ::WeightedBasisLayout{<:AbstractOPLayout}, wP, wQ) = unweighted(wP) == unweighted(wQ) && weight(wP) == weight(wQ)


copy(L::Ldiv{MappedOPLayout,Lay}) where Lay = copy(Ldiv{MappedBasisLayout,Lay}(L.A,L.B))
copy(L::Ldiv{MappedOPLayout,Lay}) where Lay<:ExpansionLayout = copy(Ldiv{MappedBasisLayout,Lay}(L.A,L.B))
copy(L::Ldiv{MappedOPLayout,Lay}) where Lay<:AbstractLazyLayout = copy(Ldiv{MappedBasisLayout,Lay}(L.A,L.B))
copy(L::Ldiv{MappedOPLayout,Lay}) where Lay<:AbstractBasisLayout = copy(Ldiv{MappedBasisLayout,Lay}(L.A,L.B))
copy(L::Ldiv{MappedOPLayout,BroadcastLayout{typeof(-)}}) = copy(Ldiv{MappedBasisLayout,BroadcastLayout{typeof(-)}}(L.A,L.B))
copy(L::Ldiv{MappedOPLayout,BroadcastLayout{typeof(+)}}) = copy(Ldiv{MappedBasisLayout,BroadcastLayout{typeof(+)}}(L.A,L.B))
copy(L::Ldiv{MappedOPLayout,BroadcastLayout{typeof(*)}}) = copy(Ldiv{MappedBasisLayout,BroadcastLayout{typeof(*)}}(L.A,L.B))
copy(L::Ldiv{MappedOPLayout,ApplyLayout{typeof(hcat)}}) = copy(Ldiv{MappedBasisLayout,ApplyLayout{typeof(hcat)}}(L.A,L.B))
copy(L::Ldiv{MappedOPLayout,ApplyLayout{typeof(*)}}) = copy(Ldiv{MappedBasisLayout,ApplyLayout{typeof(*)}}(L.A,L.B))

# OPs are immutable
copy(a::OrthogonalPolynomial) = a

"""
    jacobimatrix(P)

returns the Jacobi matrix `X` associated to a quasi-matrix of orthogonal polynomials
satisfying
```julia
x = axes(P,1)
x*P == P*X
```
Note that `X` is the transpose of the usual definition of the Jacobi matrix.
"""
jacobimatrix(P, n...) = jacobimatrix_layout(MemoryLayout(P), P, n...)

jacobimatrix_layout(lay, P) = error("Override for $(typeof(P))")


"""
    _tritrunc(X,n)

does a square truncation of a tridiagonal matrix.
"""
function _tritrunc(_, X, n)
    c,a,b = subdiagonaldata(X),diagonaldata(X),supdiagonaldata(X)
    Tridiagonal(c[OneTo(n-1)],a[OneTo(n)],b[OneTo(n-1)])
end

function _tritrunc(::SymTridiagonalLayout, X, n)
    a,b = diagonaldata(X),supdiagonaldata(X)
    SymTridiagonal(a[OneTo(n)],b[OneTo(n-1)])
end

_tritrunc(X, n) = _tritrunc(MemoryLayout(X), X, n)

jacobimatrix_layout(lay, P, n) = _tritrunc(jacobimatrix(P), n)


function recurrencecoefficients_layout(lay, Q)
    T = eltype(Q)
    X = jacobimatrix(Q)
    c,a,b = subdiagonaldata(X), diagonaldata(X), supdiagonaldata(X)
    inv.(c), -(a ./ c), Vcat(zero(T), b) ./ c
end

"""
    recurrencecoefficients(P)

returns a `(A,B,C)` associated with the Orthogonal Polynomials P,
satisfying for `x in axes(P,1)`
```julia
P[x,2] == (A[1]*x + B[1])*P[x,1]
P[x,n+1] == (A[n]*x + B[n])*P[x,n] - C[n]*P[x,n-1]
```
Note that `C[1]`` is unused.

The relationship with the Jacobi matrix is:
```julia
1/A[n] == X[n+1,n]
-B[n]/A[n] == X[n,n]
C[n+1]/A[n+1] == X[n,n+1]
```
"""
recurrencecoefficients(Q) = recurrencecoefficients_layout(MemoryLayout(Q), Q)



"""
    singularities(f)

gives the singularity structure of an expansion, e.g.,
`JacobiWeight`.
"""
singularities(::AbstractWeightLayout, w) = w
singularities(lay::BroadcastLayout, a) = singularitiesbroadcast(call(a), map(singularities, arguments(lay, a))...)
singularities(::WeightedBasisLayouts, a) = singularities(BroadcastLayout{typeof(*)}(), a)
singularities(::WeightedOPLayout, a) = singularities(weight(a))
singularities(w) = singularities(MemoryLayout(w), w)
singularities(::ExpansionLayout, f) = singularities(basis(f))

singularitiesview(w, ::Inclusion) = w # for now just assume it doesn't change
singularitiesview(w, ind) = view(w, ind)
singularities(S::SubQuasiArray) = singularitiesview(singularities(parent(S)), parentindices(S)[1])

basis_axes(::Inclusion{<:Any,<:AbstractInterval}, v) = convert(AbstractQuasiMatrix{eltype(v)}, basis_singularities(singularities(v)))

struct NoSingularities end

singularities(::Number) = NoSingularities()
singularities(r::Base.RefValue) = r[] # pass through



orthogonalityweight(P::SubQuasiArray{<:Any,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}}) =
    orthogonalityweight(parent(P))[parentindices(P)[1]]


weighted(P::AbstractQuasiMatrix) = Weighted(P)

"""
gives the inner products of OPs with their weight, i.e., Weighted(P)'P.
"""
weightedgrammatrix(P) = weightedgrammatrix_layout(MemoryLayout(P), P)
function weightedgrammatrix_layout(::MappedOPLayout, P)
    Q = parent(P)
    kr,jr = parentindices(P)
    @assert kr isa AbstractAffineQuasiVector
    weightedgrammatrix(Q)/kr.A
end

grammatrix_layout(::MappedOPLayout, P) = grammatrix_layout(MappedBasisLayout(), P)
grammatrix_layout(::WeightedOPLayout{MappedOPLayout}, P) = grammatrix_layout(MappedBasisLayout(), P)

OrthogonalPolynomial(w::Weight) =error("Override for $(typeof(w))")

@simplify *(B::Identity, C::OrthogonalPolynomial) = ApplyQuasiMatrix(*, C, jacobimatrix(C))

function layout_broadcasted(::Tuple{PolynomialLayout,AbstractOPLayout}, ::typeof(*), x::Inclusion, C)
    x == axes(C,1) || throw(DimensionMismatch())
    C*jacobimatrix(C)
end



# function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::BroadcastQuasiVector, C::OrthogonalPolynomial)
#     axes(a,1) == axes(C,1) || throw(DimensionMismatch())
#     # re-expand in OP basis
#     broadcast(*, C * (C \ a), C)
# end

# function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::AbstractAffineQuasiVector, C::OrthogonalPolynomial)
#     x = axes(C,1)
#     axes(a,1) == x || throw(DimensionMismatch())
#     broadcast(*, C * (C \ a), C)
# end



function jacobimatrix_layout(::MappedOPLayout, C, n...)
    T = eltype(C)
    P = parent(C)
    kr,jr = parentindices(C)
    @assert kr isa AbstractAffineQuasiVector
    Y = jacobimatrix(P, n...)
    kr.A \ (Y - kr.b * Eye{T}(size(Y,1)))
end

function recurrencecoefficients_layout(::MappedOPLayout, C, n...)
    P = parent(C)
    kr,jr = parentindices(C)
    @assert kr isa AbstractAffineQuasiVector
    A,B,C = recurrencecoefficients(P)
    A * kr.A, A*kr.b + B, C
end

include("clenshaw.jl")
include("ratios.jl")
include("normalized.jl")
include("lanczos.jl")
include("choleskyQR.jl")

# Default is Golub–Welsch
grid_layout(::AbstractOPLayout, P, n::Integer) = eigvals(symtridiagonalize(jacobimatrix(P, n)))
grid_layout(::MappedOPLayout, P, n::Integer) = grid_layout(MappedBasisLayout(), P, n)
plotgrid_layout(::AbstractOPLayout, P, n::Integer) = grid(P, min(40n, MAX_PLOT_POINTS))
plotgrid_layout(::MappedOPLayout, P, n::Integer) = plotgrid_layout(MappedBasisLayout(), P, n)
plotvalues_layout(::ExpansionLayout{MappedOPLayout}, f, x...) = plotvalues_layout(ExpansionLayout{MappedBasisLayout}(), f, x...)

hasboundedendpoints(_) = false # assume blow up
function plotgrid_layout(::WeightedOPLayout, P, n::Integer)
    if hasboundedendpoints(weight(P))
        plotgrid(unweighted(P), n)
    else
        grid(unweighted(P), min(40n, MAX_PLOT_POINTS))
    end
end


function golubwelsch(X::AbstractMatrix)
    D, V = eigen(symtridiagonalize(X))  # Eigenvalue decomposition
    D, V[1,:].^2
end

function golubwelsch(P, n::Integer)
    x,w = golubwelsch(jacobimatrix(P, n))
    w .*= sum(orthogonalityweight(P))
    x,w
end

golubwelsch(V::SubQuasiArray) = golubwelsch(parent(V), maximum(parentindices(V)[2]))

# Default is Golub–Welsch expansion
# note this computes the grid an extra time.
function plan_transform_layout(::AbstractOPLayout, P, szs::NTuple{N,Int}, dims=ntuple(identity,Val(N))) where N
    dimsz = getindex.(Ref(szs), dims) # get the sizes of transformed dimensions
    if P isa Normalized
        MulPlan(map(dimsz) do n
            (x,w) = golubwelsch(P, n)
            P[x,oneto(n)]'*Diagonal(w)
            end, dims)
    else
        Q = normalized(P)
        MulPlan(map(dimsz) do n
            (x,w) = golubwelsch(Q, n)
            D = (P \ Q)[1:n, 1:n]
            D*Q[x,oneto(n)]'*Diagonal(w)
            end, dims)
    end
end


plan_transform_layout(::MappedOPLayout, L, szs::NTuple{N,Int}, dims=ntuple(identity,Val(N))) where N = plan_transform_layout(MappedBasisLayout(), L, szs, dims)

@simplify function \(A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}, B::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial})
    axes(A,1) == axes(B,1) || throw(DimensionMismatch())
    _,jA = parentindices(A)
    _,jB = parentindices(B)
    (parent(A) \ parent(B))[jA, jB]
end

@simplify function \(A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{Any,Slice}}, B::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{Any,Slice}})
    axes(A,1) == axes(B,1) || throw(DimensionMismatch())
    parent(A) \ parent(B)
end

# assume we can expand w_B in wA to reduce to polynomial multiplication
# function \(wA::WeightedOrthogonalPolynomial, wB::WeightedOrthogonalPolynomial)
#     _,A = arguments(wA)
#     w_B,B = arguments(wB)
#     A \ ((A * (wA \ w_B)) .* B)
# end

## special expansion routines for constants and x
function _op_ldiv(P::AbstractQuasiMatrix{V}, f::AbstractQuasiFill{T,1}) where {T,V}
    TV = promote_type(T,V)
    Vcat(getindex_value(f)/_p0(P),Zeros{TV}(∞))
end

_op_ldiv(::AbstractQuasiMatrix{V}, ::QuasiZeros{T,1}) where {T,V} = Zeros{promote_type(T,V)}(∞)

function _op_ldiv(P::AbstractQuasiMatrix{V}, f::Inclusion{T}) where {T,V}
    axes(P,1) == f || throw(DimensionMismatch())
    A,B,_ = recurrencecoefficients(P)
    TV = promote_type(T,V)
    c = inv(convert(TV,A[1]*_p0(P)))
    Vcat(-B[1]c, c, Zeros{TV}(∞))
end

include("classical/hermite.jl")
include("classical/jacobi.jl")
include("classical/chebyshev.jl")
include("classical/ultraspherical.jl")
include("classical/laguerre.jl")
include("classical/fourier.jl")
include("roots.jl")

end # module
