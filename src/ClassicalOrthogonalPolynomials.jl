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
                to_indices, _maybetail, tail, getproperty, inv, show, isapprox, summary
import Base.Broadcast: materialize, BroadcastStyle, broadcasted, Broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, adjointlayout,
                sub_materialize, arguments, sub_paddeddata, paddeddata, PaddedLayout, resizedata!, LazyVector, ApplyLayout, call,
                _mul_arguments, CachedVector, CachedMatrix, LazyVector, LazyMatrix, axpy!, AbstractLazyLayout, BroadcastLayout,
                AbstractCachedVector, AbstractCachedMatrix, paddeddata, cache_filldata!,
                simplifiable, PaddedArray, converteltype
import ArrayLayouts: MatMulVecAdd, materialize!, _fill_lmul!, sublayout, sub_materialize, lmul!, ldiv!, ldiv, transposelayout, triangulardata,
                        subdiagonaldata, diagonaldata, supdiagonaldata, mul, rowsupport, colsupport
import LazyBandedMatrices: SymTridiagonal, Bidiagonal, Tridiagonal, unitblocks, BlockRange1, AbstractLazyBandedLayout
import LinearAlgebra: pinv, factorize, qr, adjoint, transpose, dot, mul!
import BandedMatrices: AbstractBandedLayout, AbstractBandedMatrix, _BandedMatrix, bandeddata
import FillArrays: AbstractFill, getindex_value, SquareEye
import DomainSets: components
import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle,
                    _getindex, layout_getindex, _factorize, AbstractQuasiArray, AbstractQuasiMatrix, AbstractQuasiVector,
                    AbstractQuasiFill, _dot, _equals, QuasiArrayLayout, PolynomialLayout

import InfiniteArrays: OneToInf, InfAxes, Infinity, AbstractInfUnitRange, InfiniteCardinal, InfRanges
import InfiniteLinearAlgebra: chop!, chop, choplength, compatible_resize!
import ContinuumArrays: Basis, Weight, basis, @simplify, Identity, AbstractAffineQuasiVector, ProjectionFactorization,
    inbounds_getindex, grid, plotgrid, transform_ldiv, TransformFactorization, QInfAxes, broadcastbasis, ExpansionLayout, basismap,
    AffineQuasiVector, AffineMap, WeightLayout, AbstractWeightedBasisLayout, WeightedBasisLayout, WeightedBasisLayouts, demap, AbstractBasisLayout, BasisLayout,
    checkpoints, weight, unweighted, MappedBasisLayouts, __sum, invmap, plan_ldiv, layout_broadcasted, MappedBasisLayout, SubBasisLayout, _broadcastbasis,
    plan_transform, plan_grid_transform
import FastTransforms: Λ, forwardrecurrence, forwardrecurrence!, _forwardrecurrence!, clenshaw, clenshaw!,
                        _forwardrecurrence_next, _clenshaw_next, check_clenshaw_recurrences, ChebyshevGrid, chebyshevpoints, Plan

import FastGaussQuadrature: jacobimoment

import BlockArrays: blockedrange, _BlockedUnitRange, unblock, _BlockArray, block, blockindex, BlockSlice, blockvec
import BandedMatrices: bandwidths

export OrthogonalPolynomial, Normalized, orthonormalpolynomial, LanczosPolynomial,
            Hermite, Jacobi, Legendre, Chebyshev, ChebyshevT, ChebyshevU, ChebyshevInterval, Ultraspherical, Fourier, Laguerre,
            HermiteWeight, JacobiWeight, ChebyshevWeight, ChebyshevGrid, ChebyshevTWeight, ChebyshevUWeight, UltrasphericalWeight, LegendreWeight, LaguerreWeight,
            WeightedUltraspherical, WeightedChebyshev, WeightedChebyshevT, WeightedChebyshevU, WeightedJacobi,
            ∞, Derivative, .., Inclusion,
            chebyshevt, chebyshevu, legendre, jacobi, ultraspherical,
            legendrep, jacobip, ultrasphericalc, laguerrel,hermiteh, normalizedjacobip,
            jacobimatrix, jacobiweight, legendreweight, chebyshevtweight, chebyshevuweight, Weighted, PiecewiseInterlace, plan_transform


import Base: oneto


include("interlace.jl")
include("standardchop.jl")
include("adaptivetransform.jl")

const WeightedBasis{T, A<:AbstractQuasiVector, B<:Basis} = BroadcastQuasiMatrix{T,typeof(*),<:Tuple{A,B}}
abstract type OrthogonalPolynomial{T} <: Basis{T} end
abstract type AbstractOPLayout <: AbstractBasisLayout end
struct OPLayout <: AbstractOPLayout end
MemoryLayout(::Type{<:OrthogonalPolynomial}) = OPLayout()



sublayout(::AbstractOPLayout, ::Type{<:Tuple{<:AbstractAffineQuasiVector,<:Slice}}) = MappedOPLayout()

struct MappedOPLayout <: AbstractOPLayout end
struct WeightedOPLayout{Lay<:AbstractOPLayout} <: AbstractWeightedBasisLayout end

isorthogonalityweighted(::WeightedOPLayout, _) = true
function isorthogonalityweighted(::AbstractWeightedBasisLayout, wS)
    w,S = arguments(wS)
    w == orthogonalityweight(S)
end

isorthogonalityweighted(wS) = isorthogonalityweighted(MemoryLayout(wS), wS)


_equals(::MappedOPLayout, ::MappedOPLayout, P, Q) = demap(P) == demap(Q) && basismap(P) == basismap(Q)
_equals(::MappedOPLayout, ::MappedBasisLayouts, P, Q) = demap(P) == demap(Q) && basismap(P) == basismap(Q)
_equals(::MappedBasisLayouts, ::MappedOPLayout, P, Q) = demap(P) == demap(Q) && basismap(P) == basismap(Q)

_broadcastbasis(::typeof(+), ::MappedOPLayout, ::MappedOPLayout, P, Q) = _broadcastbasis(+, MappedBasisLayout(), MappedBasisLayout(), P, Q)
_broadcastbasis(::typeof(+), ::MappedOPLayout, M::MappedBasisLayout, P, Q) = _broadcastbasis(+, MappedBasisLayout(), M, P, Q)
_broadcastbasis(::typeof(+), L::MappedBasisLayout, ::MappedOPLayout, P, Q) = _broadcastbasis(+, L, MappedBasisLayout(), P, Q)
__sum(::MappedOPLayout, A, dims) = __sum(MappedBasisLayout(), A, dims)

# demap to avoid Golub-Welsch fallback
ContinuumArrays.transform_ldiv_if_columns(L::Ldiv{MappedOPLayout,Lay}, ax::OneTo) where Lay = ContinuumArrays.transform_ldiv_if_columns(Ldiv{MappedBasisLayout,Lay}(L.A,L.B), ax)
ContinuumArrays.transform_ldiv_if_columns(L::Ldiv{MappedOPLayout,ApplyLayout{typeof(hcat)}}, ax::OneTo) = ContinuumArrays.transform_ldiv_if_columns(Ldiv{MappedBasisLayout,UnknownLayout}(L.A,L.B), ax)

_equals(::AbstractOPLayout, ::AbstractWeightedBasisLayout, _, _) = false # Weighedt-Legendre doesn't exist
_equals(::AbstractWeightedBasisLayout, ::AbstractOPLayout, _, _) = false # Weighedt-Legendre doesn't exist

_equals(::WeightedOPLayout, ::WeightedOPLayout, wP, wQ) = unweighted(wP) == unweighted(wQ)
_equals(::WeightedOPLayout, ::WeightedBasisLayout, wP, wQ) = unweighted(wP) == unweighted(wQ) && weight(wP) == weight(wQ)
_equals(::WeightedBasisLayout, ::WeightedOPLayout, wP, wQ) = unweighted(wP) == unweighted(wQ) && weight(wP) == weight(wQ)
_equals(::WeightedBasisLayout{<:AbstractOPLayout}, ::WeightedBasisLayout{<:AbstractOPLayout}, wP, wQ) = unweighted(wP) == unweighted(wQ) && weight(wP) == weight(wQ)


copy(L::Ldiv{MappedOPLayout,Lay}) where Lay<:MappedBasisLayouts = copy(Ldiv{MappedBasisLayout,Lay}(L.A,L.B))

# OPs are immutable
copy(a::OrthogonalPolynomial) = a
copy(a::SubQuasiArray{<:Any,N,<:OrthogonalPolynomial}) where N = a

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
jacobimatrix(P) = error("Override for $(typeof(P))")


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
function recurrencecoefficients(Q::AbstractQuasiMatrix{T}) where T
    X = jacobimatrix(Q)
    c,a,b = subdiagonaldata(X), diagonaldata(X), supdiagonaldata(X)
    inv.(c), -(a ./ c), Vcat(zero(T), b) ./ c
end



"""
    singularities(f)

gives the singularity structure of an expansion, e.g.,
`JacobiWeight`.
"""
singularities(::WeightLayout, w) = w
singularities(lay::BroadcastLayout, a) = singularitiesbroadcast(call(a), map(singularities, arguments(lay, a))...)
singularities(::WeightedBasisLayouts, a) = singularities(BroadcastLayout{typeof(*)}(), a)
singularities(::WeightedOPLayout, a) = singularities(weight(a))
singularities(w) = singularities(MemoryLayout(w), w)
singularities(::ExpansionLayout, f) = singularities(basis(f))

singularities(S::SubQuasiArray) = singularities(parent(S))[parentindices(S)[1]]

struct NoSingularities end

singularities(::Number) = NoSingularities()
singularities(r::Base.RefValue) = r[] # pass through



orthogonalityweight(P::SubQuasiArray{<:Any,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}}) =
    orthogonalityweight(parent(P))[parentindices(P)[1]]

function massmatrix(P::SubQuasiArray{<:Any,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}})
    Q = parent(P)
    kr,jr = parentindices(P)
    massmatrix(Q)/kr.A
end

weighted(P::AbstractQuasiMatrix) = Weighted(P)

OrthogonalPolynomial(w::Weight) =error("Override for $(typeof(w))")

@simplify *(B::Identity, C::OrthogonalPolynomial) = C*jacobimatrix(C)

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



function jacobimatrix(C::SubQuasiArray{T,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}}) where T
    P = parent(C)
    kr,jr = parentindices(C)
    Y = jacobimatrix(P)
    kr.A \ (Y - kr.b * Eye{T}(size(Y,1)))
end

function recurrencecoefficients(C::SubQuasiArray{T,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}}) where T
    P = parent(C)
    kr,jr = parentindices(C)
    A,B,C = recurrencecoefficients(P)
    A * kr.A, A*kr.b + B, C
end

include("clenshaw.jl")
include("ratios.jl")
include("normalized.jl")
include("lanczos.jl")

function _tritrunc(_, X, n)
    c,a,b = subdiagonaldata(X),diagonaldata(X),supdiagonaldata(X)
    Tridiagonal(c[OneTo(n-1)],a[OneTo(n)],b[OneTo(n-1)])
end

function _tritrunc(::SymTridiagonalLayout, X, n)
    a,b = diagonaldata(X),supdiagonaldata(X)
    SymTridiagonal(a[OneTo(n)],b[OneTo(n-1)])
end

_tritrunc(X, n) = _tritrunc(MemoryLayout(X), X, n)

jacobimatrix(V::SubQuasiArray{<:Any,2,<:Any,<:Tuple{Inclusion,OneTo}}) =
    _tritrunc(jacobimatrix(parent(V)), maximum(parentindices(V)[2]))

grid(P::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{Inclusion,OneTo}}) =
    eigvals(symtridiagonalize(jacobimatrix(P)))

function grid(Pn::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{Inclusion,Any}})
    kr,jr = parentindices(Pn)
    grid(parent(Pn)[:,oneto(maximum(jr))])
end

function plotgrid(Pn::SubQuasiArray{T,2,<:OrthogonalPolynomial,<:Tuple{Inclusion,Any}}) where T
    kr,jr = parentindices(Pn)
    grid(parent(Pn)[:,oneto(40maximum(jr))])
end

function golubwelsch(X)
    D, V = eigen(symtridiagonalize(X))  # Eigenvalue decomposition
    D, V[1,:].^2
end

function golubwelsch(V::SubQuasiArray)
    x,w = golubwelsch(jacobimatrix(V))
    w .*= sum(orthogonalityweight(parent(V)))
    x,w
end

"""
    MulPlan(matrix, dims)

Takes a matrix and supports it applied to different dimensions.
"""
struct MulPlan{T, Fact, Dims} # <: Plan{T} We don't depend on AbstractFFTs
    matrix::Fact
    dims::Dims
end

MulPlan(fact, dims) = MulPlan{eltype(fact), typeof(fact), typeof(dims)}(fact, dims)

function *(P::MulPlan{<:Any,<:Any,Int}, x::AbstractVector)
    @assert P.dims == 1
    P.matrix * x
end

function *(P::MulPlan{<:Any,<:Any,Int}, X::AbstractMatrix)
    if P.dims == 1
        P.matrix * X
    else
        @assert P.dims == 2
        permutedims(P.matrix * permutedims(X))
    end
end

function *(P::MulPlan{<:Any,<:Any,Int}, X::AbstractArray{<:Any,3})
    Y = similar(X)
    if P.dims == 1
        for j in axes(X,3)
            Y[:,:,j] = P.matrix * X[:,:,j]
        end
    elseif P.dims == 2
        for k in axes(X,1)
            Y[k,:,:] = P.matrix * X[k,:,:]
        end
    else
        @assert P.dims == 3
        for k in axes(X,1), j in axes(X,2)
            Y[k,j,:] = P.matrix * X[k,j,:]
        end
    end
    Y
end

function *(P::MulPlan, X::AbstractArray)
    for d in P.dims
        X = MulPlan(P.matrix, d) * X
    end
    X
end

*(A::AbstractMatrix, P::MulPlan) = MulPlan(A*P.matrix, P.dims)


function plan_grid_transform(Q::Normalized, arr, dims=1:ndims(arr))
    L = Q[:,OneTo(size(arr,1))]
    x,w = golubwelsch(L)
    x, MulPlan(L[x,:]'*Diagonal(w), dims)
end

function plan_grid_transform(P::OrthogonalPolynomial, arr, dims...)
    Q = Normalized(P)
    x, A = plan_grid_transform(Q, arr, dims...)
    n = size(arr,1)
    D = (P \ Q)[1:n, 1:n]
    x, D * A
end


function \(A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}, B::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial})
    axes(A,1) == axes(B,1) || throw(DimensionMismatch())
    _,jA = parentindices(A)
    _,jB = parentindices(B)
    (parent(A) \ parent(B))[jA, jB]
end

function \(A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{Any,Slice}}, B::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{Any,Slice}})
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
include("stieltjes.jl")


end # module
