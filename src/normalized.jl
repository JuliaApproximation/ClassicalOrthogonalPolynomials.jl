"""
   normalizationconstant

gives the normalization constants so that the jacobi matrix is symmetric,
that is, so we have orthonormal OPs:

    Q == P*normalizationconstant(P)
"""
function normalizationconstant(μ, P::AbstractQuasiMatrix{T}) where T
    X = jacobimatrix(P)
    c,b = subdiagonaldata(X),supdiagonaldata(X)
    # hide array type to avoid crazy compilation
    Accumulate{T,1,typeof(*),Vector{T},AbstractVector{T}}(*, T[μ], Vcat(zero(T),sqrt.(c ./ b)), 1, (1,))
end

normalizationconstant(P::AbstractQuasiMatrix) = normalizationconstant(inv(sqrt(sum(orthogonalityweight(P)))), P)


abstract type OrthonormalPolynomial{T} <: OrthogonalPolynomial{T} end
struct Normalized{T, OPs<:AbstractQuasiMatrix{T}, NL} <: OrthonormalPolynomial{T}
    P::OPs
    scaling::NL # Q = P * Diagonal(scaling)
end

Normalized(P::AbstractQuasiMatrix{T}) where T = Normalized(P, normalizationconstant(P))
Normalized(Q::Normalized) = Q
normalized(P) = Normalized(P)
normalized(Q::OrthonormalPolynomial) = Q
isnormalized(P::OrthonormalPolynomial) = true
isnormalized(_) = false

"""
    AbstractNormalizedOPLayout

represents OPs that are of the form P * R where P is another family of OPs and R is upper-triangular.
"""
abstract type AbstractNormalizedOPLayout <: AbstractOPLayout end
struct NormalizedOPLayout{LAY<:AbstractBasisLayout} <: AbstractNormalizedOPLayout end

MemoryLayout(::Type{<:Normalized{<:Any, OPs}}) where OPs = NormalizedOPLayout{typeof(MemoryLayout(OPs))}()

struct QuasiQR{T, QQ, RR} <: Factorization{T}
    Q::QQ
    R::RR
end

QuasiQR(Q::AbstractQuasiMatrix{T}, R::AbstractMatrix{V}) where {T,V} =
    QuasiQR{promote_type(T,V),typeof(Q),typeof(R)}(Q, R)

Base.iterate(S::QuasiQR) = (S.Q, Val(:R))
Base.iterate(S::QuasiQR, ::Val{:R}) = (S.R, Val(:done))
Base.iterate(S::QuasiQR, ::Val{:done}) = nothing


axes(Q::Normalized) = axes(Q.P)

equals_layout(::AbstractNormalizedOPLayout, ::AbstractNormalizedOPLayout, P, Q) = orthogonalityweight(P) == orthogonalityweight(Q)
equals_layout(::AbstractNormalizedOPLayout, ::AbstractOPLayout, P, Q) = isnormalized(Q) && orthogonalityweight(P) == orthogonalityweight(Q)
equals_layout(::AbstractOPLayout, ::AbstractNormalizedOPLayout, P, Q) = isnormalized(P) && orthogonalityweight(P) == orthogonalityweight(Q)



_p0(Q::Normalized) = Q.scaling[1]


# x * p[n] = c[n-1] * p[n-1] + a[n] * p[n] + b[n] * p[n+1]
# x * q[n]/h[n] = c[n-1] * q[n-1]/h[n-1] + a[n] * q[n]/h[n] + b[n] * q[n+1]/h[n+1]
# x * q[n+1] = c[n-1] * h[n]/h[n-1] * q[n-1] + a[n] * q[n] + b[n] * h[n]/h[n+1] * q[n+1]

# q_{n+1}/h[n+1] = (A_n * x + B_n) * q_n/h[n] - C_n * p_{n-1}/h[n-1]
# q_{n+1} = (h[n+1]/h[n] * A_n * x + h[n+1]/h[n] * B_n) * q_n - h[n+1]/h[n-1] * C_n * p_{n-1}

function symtridiagonalize(X)
    c,a,b = subdiagonaldata(X), diagonaldata(X), supdiagonaldata(X)
    SymTridiagonal(a, sqrt.(b .* c))
end
jacobimatrix(Q::Normalized) = symtridiagonalize(jacobimatrix(Q.P))

orthogonalityweight(Q::Normalized) = orthogonalityweight(Q.P)
singularities(Q::Normalized) = singularities(Q.P)

function demap(Q::Normalized)
    P,D =  arguments(ApplyLayout{typeof(*)}(), Q)
    demap(P) * D
end

# Sometimes we want to expand out, sometimes we don't

QuasiArrays.ApplyQuasiArray(Q::Normalized) = ApplyQuasiArray(*, arguments(ApplyLayout{typeof(*)}(), Q)...)

ArrayLayouts.mul(Q::Normalized, C::AbstractArray) = ApplyQuasiArray(*, Q, C)

# TODO: Leverage fast transforms

# transform_ldiv(Q::Normalized, C::AbstractQuasiArray) = Q.scaling .\ (Q.P \ C)
# this leverages fast transforms for Q.P
function transform_ldiv(Q::Normalized, C::AbstractQuasiArray)
    c = paddeddata(Q.P \ C)
    [Q.scaling[axes(c,1)] .\ c; zeros(eltype(c), ∞)]
end

function transform_ldiv(V::SubQuasiArray{<:Any,2,<:Normalized}, C::AbstractQuasiArray)
    Q = parent(V)
    P = Q.P
    kr, jr = parentindices(V)
    c = transform_ldiv(view(P, kr, jr), C)
    Q.scaling[axes(c,1)] .\ c
end

arguments(::ApplyLayout{typeof(*)}, Q::Normalized) = Q.P, Diagonal(Q.scaling)
_mul_arguments(Q::Normalized) = arguments(ApplyLayout{typeof(*)}(), Q)
_mul_arguments(Q::QuasiAdjoint{<:Any,<:Normalized}) = arguments(ApplyLayout{typeof(*)}(), Q)
copy(M::Mul{<:AdjointBasisLayout{<:NormalizedOPLayout},Blay}) where Blay<:AbstractBasisLayout = copy(Mul{ApplyLayout{typeof(*)}, Blay}(M.A, M.B))

# table stable identity if A.P == B.P
@inline _normalized_ldiv(An, C, Bn) = An \ (C * Bn)
@inline _normalized_ldiv(An, C::Eye{T}, Bn) where T = FillArrays.SquareEye{promote_type(eltype(An),T,eltype(Bn))}(ℵ₀)

simplifiable(::Ldiv{<:NormalizedOPLayout,<:NormalizedOPLayout}) = Val(true)
@inline copy(L::Ldiv{<:NormalizedOPLayout,<:NormalizedOPLayout}) = _normalized_ldiv(Diagonal(L.A.scaling), L.A.P \ L.B.P, Diagonal(L.B.scaling))
@inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,<:AbstractNormalizedOPLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{Lay,<:AbstractNormalizedOPLayout}) where Lay = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,Lay}) where Lay = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
@inline copy(L::Ldiv{Lay,<:AbstractNormalizedOPLayout}) where Lay<:AbstractBasisLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,Lay}) where Lay<:AbstractBasisLayout = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
@inline copy(L::Ldiv{Lay,<:AbstractNormalizedOPLayout}) where Lay<:AbstractLazyLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,Lay}) where Lay<:AbstractLazyLayout = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
@inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,<:ExpansionLayout}) = copy(Ldiv{BasisLayout,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{ApplyLayout{typeof(*)},<:AbstractNormalizedOPLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{MappedOPLayout,<:AbstractNormalizedOPLayout}) = copy(Ldiv{MappedOPLayout,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,ApplyLayout{typeof(hcat)}}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(hcat)}}(L.A, L.B))
for Lay in (:(ApplyLayout{typeof(*)}),:(BroadcastLayout{typeof(+)}),:(BroadcastLayout{typeof(-)}))
    @eval begin
        @inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,$Lay}) = copy(Ldiv{ApplyLayout{typeof(*)},$Lay}(L.A, L.B))
        @inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,$Lay,<:Any,<:AbstractQuasiVector}) = copy(Ldiv{ApplyLayout{typeof(*)},$Lay}(L.A, L.B))
    end
end

copy(L::Ldiv{Lay,<:AbstractNormalizedOPLayout}) where Lay<:MappedBasisLayouts = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A,L.B))

# want to use special re-expansion routines without expanding Normalized basis
@inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,BroadcastLayout{typeof(*)}}) = copy(Ldiv{BasisLayout,BroadcastLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:AbstractNormalizedOPLayout,BroadcastLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = copy(Ldiv{BasisLayout,BroadcastLayout{typeof(*)}}(L.A, L.B))

# take out diagonal scaling for Weighted(::Normalized)
function _norm_expand_ldiv(A, w_B)
    w,B = w_B.args
    B̃,D = arguments(ApplyLayout{typeof(*)}(), B)
    (A \ (w .* B̃)) * D
end
simplifiable(::Ldiv{<:AbstractNormalizedOPLayout,<:WeightedBasisLayout{<:AbstractNormalizedOPLayout}}) = Val(true)
copy(L::Ldiv{<:AbstractNormalizedOPLayout,<:WeightedBasisLayout{<:AbstractNormalizedOPLayout}}) = _norm_expand_ldiv(L.A, L.B)
simplifiable(::Ldiv{OPLayout,<:WeightedBasisLayout{<:AbstractNormalizedOPLayout}}) = Val(true)
copy(L::Ldiv{OPLayout,<:WeightedBasisLayout{<:AbstractNormalizedOPLayout}}) = _norm_expand_ldiv(L.A, L.B)

###
# show
###
show(io::IO, Q::Normalized) = print(io, "Normalized($(Q.P))")
show(io::IO, ::MIME"text/plain", Q::Normalized) = show(io, Q)


abstract type AbstractWeighted{T} <: Basis{T} end




getindex(Q::AbstractWeighted, x::Union{Number,AbstractVector}, jr::Union{Number,AbstractVector}) = weight(Q)[x] .* unweighted(Q)[x,jr]

MemoryLayout(::Type{<:AbstractWeighted}) = WeightedBasisLayout{OPLayout}()
convert(::Type{WeightedBasis}, Q::AbstractWeighted) = weight(Q) .* unweighted(Q)

# make act like WeightedBasisLayout
ContinuumArrays.sublayout(::WeightedOPLayout{Lay}, inds::Type{<:Tuple{<:AbstractAffineQuasiVector,<:AbstractVector}}) where Lay = sublayout(WeightedBasisLayout{Lay}(), inds)
ContinuumArrays.sublayout(::WeightedOPLayout{Lay}, inds::Type{<:Tuple{<:Inclusion,<:AbstractVector}}) where Lay = sublayout(WeightedBasisLayout{Lay}(), inds)

ContinuumArrays.unweighted(wP::AbstractWeighted) = wP.P
# function copy(L::Ldiv{WeightedOPLayout,WeightedOPLayout})
#     L.A.P == L.B.P && return Eye{eltype(L)}(∞)
#     convert(WeightedOrthogonalPolynomial, L.A) \ convert(WeightedOrthogonalPolynomial, L.B)
# end

# copy(L::Ldiv{WeightedOPLayout,<:BroadcastLayout}) = convert(WeightedOrthogonalPolynomial, L.A) \ L.B
# copy(L::Ldiv{WeightedOPLayout,<:AbstractLazyLayout}) = convert(WeightedOrthogonalPolynomial, L.A) \ L.B
# copy(L::Ldiv{WeightedOPLayout,<:AbstractBasisLayout}) = convert(WeightedOrthogonalPolynomial, L.A) \ L.B
# copy(L::Ldiv{<:AbstractLazyLayout,WeightedOPLayout}) = L.A \ convert(WeightedOrthogonalPolynomial, L.B)
# copy(L::Ldiv{<:AbstractBasisLayout,WeightedOPLayout}) = L.A \ convert(WeightedOrthogonalPolynomial, L.B)
# copy(L::Ldiv{WeightedOPLayout,ApplyLayout{typeof(*)}}) = copy(Ldiv{UnknownLayout,ApplyLayout{typeof(*)}}(L.A, L.B))
# copy(L::Ldiv{WeightedOPLayout,<:ExpansionLayout}) = copy(Ldiv{UnknownLayout,ApplyLayout{typeof(*)}}(L.A, L.B))
# copy(L::Ldiv{WeightedOPLayout,ApplyLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = copy(Ldiv{UnknownLayout,ApplyLayout{typeof(*)}}(L.A, L.B))

copy(L::Ldiv{<:WeightedOPLayout{<:AbstractNormalizedOPLayout},Lay}) where Lay<:AbstractBasisLayout = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A,L.B))
copy(L::Ldiv{Lay,<:WeightedOPLayout{<:AbstractNormalizedOPLayout}}) where Lay<:AbstractBasisLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A,L.B))
copy(L::Ldiv{<:WeightedOPLayout{<:AbstractNormalizedOPLayout},<:WeightedOPLayout{<:AbstractNormalizedOPLayout}}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}(L.A,L.B))
copy(L::Ldiv{<:WeightedBasisLayout{<:AbstractNormalizedOPLayout},Lay}) where Lay<:AbstractBasisLayout = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A,L.B))
copy(L::Ldiv{Lay,<:WeightedBasisLayout{<:AbstractNormalizedOPLayout}}) where Lay<:AbstractBasisLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A,L.B))
copy(L::Ldiv{<:WeightedBasisLayout{<:AbstractNormalizedOPLayout},<:WeightedBasisLayout{<:AbstractNormalizedOPLayout}}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}(L.A,L.B))
copy(L::Ldiv{<:WeightedOPLayout{<:AbstractNormalizedOPLayout},<:WeightedBasisLayout{<:AbstractNormalizedOPLayout}}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}(L.A,L.B))
copy(L::Ldiv{<:WeightedBasisLayout{<:AbstractNormalizedOPLayout},<:WeightedOPLayout{<:AbstractNormalizedOPLayout}}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}(L.A,L.B))
copy(L::Ldiv{<:AbstractNormalizedOPLayout, <:WeightedOPLayout{<:AbstractNormalizedOPLayout}}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}(L.A, L.B))

# function layout_broadcasted(::ExpansionLayout{WeightedOPLayout}, ::OPLayout, ::typeof(*), a, P)
#     axes(a,1) == axes(P,1) || throw(DimensionMismatch())
#     wQ,c = arguments(a)
#     w,Q = arguments(wQ)
#     (w .* P) * Clenshaw(Q * c, P)
# end


"""
    OrthonormalWeighted(P)

is the orthonormal with respect to L^2 basis given by
`sqrt.(orthogonalityweight(P)) .* Normalized(P)`.
"""
struct OrthonormalWeighted{T, PP<:AbstractQuasiMatrix{T}} <: AbstractWeighted{T}
    P::Normalized{T, PP}
end

function OrthonormalWeighted(P)
    Q = normalized(P)
    OrthonormalWeighted{eltype(Q),typeof(P)}(Q)
end

axes(Q::OrthonormalWeighted) = axes(Q.P)
copy(Q::OrthonormalWeighted) = Q

==(A::OrthonormalWeighted, B::OrthonormalWeighted) = A.P == B.P


weight(Q::OrthonormalWeighted) = sqrt.(orthogonalityweight(Q.P))

broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, Q::OrthonormalWeighted) = Q * (Q.P \ (x .* Q.P))

grammatrix(A::OrthonormalWeighted{T}) where T = Eye{T}(∞)


"""
    Weighted(P)

is equivalent to `orthogonalityweight(P) .* P`
"""
struct Weighted{T, PP<:AbstractQuasiMatrix{T}} <: AbstractWeighted{T}
    P::PP
end

AbstractQuasiArray{T}(W::Weighted) where T = Weighted(AbstractQuasiArray{T}(W.P))
AbstractQuasiMatrix{T}(W::Weighted) where T = Weighted(AbstractQuasiArray{T}(W.P))

axes(Q::Weighted) = axes(Q.P)
copy(Q::Weighted) = Q

weight(wP::Weighted) = orthogonalityweight(wP.P)

_weightedmemorylayout(::PP) where PP<:AbstractOPLayout =  WeightedOPLayout{PP}()
_weightedmemorylayout(::PP) where PP =  WeightedBasisLayout{PP}()
MemoryLayout(::Type{<:Weighted{<:Any,PP}}) where PP = _weightedmemorylayout(MemoryLayout(PP))

function arguments(::ApplyLayout{typeof(*)}, Q::BroadcastQuasiMatrix{<:Any,typeof(*),<:Tuple{Weight,Normalized}})
    w,Q = Q.args
    P,D = arguments(*,Q)
    (w .* P),D
end
function arguments(::ApplyLayout{typeof(*)}, Q::Weighted{<:Any,<:Normalized})
    P,D = arguments(*, Q.P)
    Weighted(P),D
end
_mul_arguments(Q::Weighted{<:Any,<:Normalized}) = arguments(ApplyLayout{typeof(*)}(), Q)

# convert(::Type{WeightedOrthogonalPolynomial}, P::Weighted) = weight(P) .* unweighted(P)

broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, Q::Weighted) = Q * (Q.P \ (x .* Q.P))


@simplify function *(Ac::QuasiAdjoint{<:Any,<:Weighted}, B::AbstractQuasiVector)
    P = (Ac').P
    weightedgrammatrix(P) * (P\B)
end

@simplify function *(Ac::QuasiAdjoint{<:Any,<:Weighted{<:Any,<:SubQuasiArray}}, B::Weighted{<:Any,<:SubQuasiArray})
    P = (Ac').P
    Q = B.P
    V = view(Weighted(parent(P)), parentindices(P)...)
    W = view(Weighted(parent(Q)), parentindices(Q)...)
    V'W
end

show(io::IO, Q::Weighted) = print(io, "Weighted($(Q.P))")

sum_layout(::AbstractNormalizedOPLayout, A, dims) = sum_layout(ApplyLayout{typeof(*)}(), A, dims)
function sum_layout(::WeightedOPLayout, A, dims)
    @assert dims == 1
    Hcat(sum(weight(A)), Zeros{eltype(A)}(1,∞))
end

_sum(p::SubQuasiArray{T,1,<:Weighted,<:Tuple{Inclusion,Int}}, ::Colon) where T = 
    parentindices(p)[2] == 1 ? convert(T, sum(weight(parent(p)))) : zero(T)

demap(W::Weighted) = Weighted(demap(W.P))
basismap(W::Weighted) = basismap(W.P)
const MappedOPLayouts = Union{MappedOPLayout,WeightedOPLayout{MappedOPLayout}}
diff_layout(::MappedOPLayouts, A, dims...) = diff_layout(MappedBasisLayout(), A, dims...)

diff_layout(::NormalizedOPLayout, A, dims...) = diff_layout(ApplyLayout{typeof(*)}(), A, dims...)