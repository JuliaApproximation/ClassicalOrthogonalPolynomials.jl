module ClassicalOrthogonalPolynomials
using ContinuumArrays, QuasiArrays, LazyArrays, FillArrays, BandedMatrices, BlockArrays,
    IntervalSets, DomainSets, ArrayLayouts, SpecialFunctions,
    InfiniteLinearAlgebra, InfiniteArrays, LinearAlgebra, FastGaussQuadrature, FastTransforms, FFTW,
    LazyBandedMatrices, HypergeometricFunctions

import Base: @_inline_meta, axes, getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy,
                first, last, Slice, size, length, axes, IdentityUnitRange, sum, _sum,
                to_indices, _maybetail, tail, getproperty, inv, show, isapprox, summary
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport, adjointlayout,
                sub_materialize, arguments, sub_paddeddata, paddeddata, PaddedLayout, resizedata!, LazyVector, ApplyLayout, call,
                _mul_arguments, CachedVector, CachedMatrix, LazyVector, LazyMatrix, axpy!, AbstractLazyLayout, BroadcastLayout, 
                AbstractCachedVector, AbstractCachedMatrix
import ArrayLayouts: MatMulVecAdd, materialize!, _fill_lmul!, sublayout, sub_materialize, lmul!, ldiv!, ldiv, transposelayout, triangulardata,
                        subdiagonaldata, diagonaldata, supdiagonaldata
import LazyBandedMatrices: SymTridiagonal, Bidiagonal, Tridiagonal, AbstractLazyBandedLayout
import LinearAlgebra: pinv, factorize, qr, adjoint, transpose, dot
import BandedMatrices: AbstractBandedLayout, AbstractBandedMatrix, _BandedMatrix, bandeddata
import FillArrays: AbstractFill, getindex_value

import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle,
                    _getindex, layout_getindex, _factorize

import InfiniteArrays: OneToInf, InfAxes, Infinity, AbstractInfUnitRange, InfiniteCardinal, InfRanges
import ContinuumArrays: Basis, Weight, basis, @simplify, Identity, AbstractAffineQuasiVector, ProjectionFactorization,
    inbounds_getindex, grid, plotgrid, transform, transform_ldiv, TransformFactorization, QInfAxes, broadcastbasis, Expansion,
    AffineQuasiVector, AffineMap, WeightLayout, WeightedBasisLayout, WeightedBasisLayouts, demap, AbstractBasisLayout, BasisLayout,
    checkpoints, weight, unweightedbasis, MappedBasisLayouts, __sum
import FastTransforms: Λ, forwardrecurrence, forwardrecurrence!, _forwardrecurrence!, clenshaw, clenshaw!,
                        _forwardrecurrence_next, _clenshaw_next, check_clenshaw_recurrences, ChebyshevGrid, chebyshevpoints

import FastGaussQuadrature: jacobimoment

import BlockArrays: blockedrange, _BlockedUnitRange, unblock, _BlockArray
import BandedMatrices: bandwidths

export OrthogonalPolynomial, Normalized, orthonormalpolynomial, LanczosPolynomial, 
            Hermite, Jacobi, Legendre, Chebyshev, ChebyshevT, ChebyshevU, ChebyshevInterval, Ultraspherical, Fourier, Laguerre,
            HermiteWeight, JacobiWeight, ChebyshevWeight, ChebyshevGrid, ChebyshevTWeight, ChebyshevUWeight, UltrasphericalWeight, LegendreWeight, LaguerreWeight,
            WeightedUltraspherical, WeightedChebyshev, WeightedChebyshevT, WeightedChebyshevU, WeightedJacobi,
            ∞, Derivative, .., Inclusion, 
            chebyshevt, chebyshevu, legendre, jacobi,
            legendrep, jacobip, ultrasphericalc, laguerrel,hermiteh, normalizedjacobip,
            jacobimatrix, jacobiweight, legendreweight, chebyshevtweight, chebyshevuweight

if VERSION < v"1.6-"
    oneto(n) = Base.OneTo(n)
else
    import Base: oneto
end


include("interlace.jl")


cardinality(::FullSpace{<:AbstractFloat}) = ℵ₁
cardinality(::EuclideanDomain) = ℵ₁

transform_ldiv(A, f, ::Tuple{<:Any,InfiniteCardinal{0}})  = adaptivetransform_ldiv(A, f)

function chop!(c::AbstractVector, tol::Real)
    @assert tol >= 0

    for k=length(c):-1:1
        if abs(c[k]) > tol
            resize!(c,k)
            return c
        end
    end
    resize!(c,0)
    c
end

setaxis(c, ::OneToInf) = c
setaxis(c, ax::BlockedUnitRange) = PseudoBlockVector(c, (ax,))

function adaptivetransform_ldiv(A::AbstractQuasiArray{U}, f::AbstractQuasiArray{V}) where {U,V}
    T = promote_type(U,V)

    r = checkpoints(A)
    fr = f[r]
    maxabsfr = norm(fr,Inf)

    tol = 20eps(real(T))

    for n = 2 .^ (4:∞)
        An = A[:,oneto(n)]
        cfs = An \ f
        maxabsc = maximum(abs, cfs)
        if maxabsc == 0 && maxabsfr == 0
            return zeros(T,∞)
        end

        un = A * [cfs; Zeros{T}(∞)]
        # we allow for transformed coefficients being a different size
        ##TODO: how to do scaling for unnormalized bases like Jacobi?
        if maximum(abs,@views(cfs[n-2:end])) < 10tol*maxabsc &&
                all(norm.(un[r] - fr, 1) .< tol * n * maxabsfr*1000)
            return setaxis([chop!(cfs, tol); zeros(T,∞)], axes(A,2))
        end
    end
    error("Have not converged")
end

abstract type OrthogonalPolynomial{T} <: Basis{T} end

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


const WeightedOrthogonalPolynomial{T, A<:AbstractQuasiVector, B<:OrthogonalPolynomial} = WeightedBasis{T, A, B}

function isorthogonalityweighted(wS::WeightedOrthogonalPolynomial)
    w,S = wS.args
    w == orthogonalityweight(S)
end

"""
    singularities(f)

gives the singularity structure of an expansion, e.g.,
`JacobiWeight`.
"""
singularities(::WeightLayout, w) = w
singularities(lay::BroadcastLayout, a) = singularitiesbroadcast(call(a), map(singularities, arguments(lay, a))...)
singularities(::WeightedBasisLayouts, a) = singularities(BroadcastLayout{typeof(*)}(), a)
singularities(w) = singularities(MemoryLayout(w), w)
singularities(f::Expansion) = singularities(basis(f))
singularities(S::WeightedOrthogonalPolynomial) = singularities(S.args[1])

singularities(S::SubQuasiArray) = singularities(parent(S))[parentindices(S)[1]]

struct NoSingularities end

singularities(::Number) = NoSingularities()
singularities(r::Base.RefValue) = r[] # pass through



orthogonalityweight(P::SubQuasiArray{<:Any,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}}) =
    orthogonalityweight(parent(P))[parentindices(P)[1]]

_weighted(w, P) = w .* P
weighted(P::AbstractQuasiMatrix) = _weighted(orthogonalityweight(P), P)

OrthogonalPolynomial(w::Weight) =error("Override for $(typeof(w))")

@simplify *(B::Identity, C::OrthogonalPolynomial) = C*jacobimatrix(C)

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::OrthogonalPolynomial)
    x == axes(C,1) || throw(DimensionMismatch())
    C*jacobimatrix(C)
end

# function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::BroadcastQuasiVector, C::OrthogonalPolynomial)
#     axes(a,1) == axes(C,1) || throw(DimensionMismatch())
#     # re-expand in OP basis
#     broadcast(*, C * (C \ a), C)
# end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::AbstractAffineQuasiVector, C::OrthogonalPolynomial)
    x = axes(C,1)
    axes(a,1) == x || throw(DimensionMismatch())
    broadcast(*, C * (C \ a), C)
end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::WeightedOrthogonalPolynomial)
    x == axes(C,1) || throw(DimensionMismatch())
    w,P = C.args
    P2, J = (x .* P).args
    @assert P == P2
    (w .* P) * J
end

##
# Multiplication for mapped and subviews x .* view(P,...)
##

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    T = promote_type(eltype(x), eltype(C))
    x == axes(C,1) || throw(DimensionMismatch())
    P = parent(C)
    kr,jr = parentindices(C)
    y = axes(P,1)
    Y = P \ (y .* P)
    X = kr.A \ (Y     - kr.b * Eye{T}(∞))
    P[kr, :] * view(X,:,jr)
end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Slice}})
    T = promote_type(eltype(x), eltype(C))
    x == axes(C,1) || throw(DimensionMismatch())
    P = parent(C)
    kr,_ = parentindices(C)
    y = axes(P,1)
    Y = P \ (y .* P)
    X = kr.A \ (Y     - kr.b * Eye{T}(∞))
    P[kr, :] * X
end

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


_vec(a) = vec(a)
_vec(a::InfiniteArrays.ReshapedArray) = _vec(parent(a))
_vec(a::Adjoint{<:Any,<:AbstractVector}) = a'

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

grid(P::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{Inclusion,AbstractUnitRange}}) = 
    eigvals(symtridiagonalize(jacobimatrix(P)))

function golubwelsch(X)
    D, V = eigen(symtridiagonalize(X))  # Eigenvalue decomposition
    D, V[1,:].^2
end

function golubwelsch(V::SubQuasiArray)
    x,w = golubwelsch(jacobimatrix(V))
    w .*= sum(orthogonalityweight(parent(V)))
    x,w
end

function factorize(L::SubQuasiArray{T,2,<:Normalized,<:Tuple{Inclusion,OneTo}}) where T
    x,w = golubwelsch(L)
    TransformFactorization(x, L[x,:]'*Diagonal(w))
end


function factorize(L::SubQuasiArray{T,2,<:OrthogonalPolynomial,<:Tuple{Inclusion,OneTo}}) where T
    x,w = golubwelsch(L)
    Q = Normalized(parent(L))[parentindices(L)...]
    D = L \ Q
    F = factorize(Q)
    TransformFactorization(F.grid, D*F.plan)
end

function factorize(L::SubQuasiArray{T,2,<:OrthogonalPolynomial,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) where T
    _,jr = parentindices(L)
    ProjectionFactorization(factorize(parent(L)[:,oneto(maximum(jr))]), jr)
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

function \(wA::WeightedOrthogonalPolynomial, wB::WeightedOrthogonalPolynomial)
    w_A,A = arguments(wA)
    w_B,B = arguments(wB)
    w_A == w_B || error("Not implemented")
    A\B
end


include("classical/hermite.jl")
include("classical/jacobi.jl")
include("classical/chebyshev.jl")
include("classical/ultraspherical.jl")
include("classical/laguerre.jl")
include("classical/fourier.jl")
include("stieltjes.jl")


end # module
