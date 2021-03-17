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


struct Normalized{T, OPs<:AbstractQuasiMatrix{T}, NL} <: OrthogonalPolynomial{T}
    P::OPs
    scaling::NL # Q = P * Diagonal(scaling)
end

Normalized(P::AbstractQuasiMatrix{T}) where T = Normalized(P, normalizationconstant(P))
Normalized(Q::Normalized) = Q
normalized(P) = Normalized(P)

struct NormalizedBasisLayout{LAY<:AbstractBasisLayout} <: AbstractBasisLayout end

MemoryLayout(::Type{<:Normalized{<:Any, OPs}}) where OPs = NormalizedBasisLayout{typeof(MemoryLayout(OPs))}()

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
==(A::Normalized, B::Normalized) = A.P == B.P

# There is no point in a Normalized OP thats ==, so just return false
==(A::Normalized, B::OrthogonalPolynomial) = false
==(A::OrthogonalPolynomial, B::Normalized) = false
==(A::Normalized, B::AbstractQuasiMatrix) = false
==(A::AbstractQuasiMatrix, B::Normalized) = false
==(A::Normalized, B::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}) = false
==(A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}, B::Normalized) = false

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

grid(Q::SubQuasiArray{<:Any,2,<:Normalized}) = grid(view(parent(Q).P, parentindices(Q)...))

# transform_ldiv(Q::Normalized, C::AbstractQuasiArray) = Q.scaling .\ (Q.P \ C)
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

# table stable identity if A.P == B.P
@inline _normalized_ldiv(An, C, Bn) = An \ (C * Bn)
@inline _normalized_ldiv(An, C::Eye{T}, Bn) where T = FillArrays.SquareEye{promote_type(eltype(An),T,eltype(Bn))}(ℵ₀)
@inline copy(L::Ldiv{<:NormalizedBasisLayout,<:NormalizedBasisLayout}) = _normalized_ldiv(Diagonal(L.A.scaling), L.A.P \ L.B.P, Diagonal(L.B.scaling))
@inline copy(L::Ldiv{Lay,<:NormalizedBasisLayout}) where Lay = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:NormalizedBasisLayout,Lay}) where Lay = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
@inline copy(L::Ldiv{<:NormalizedBasisLayout,Lay,<:Any,<:AbstractQuasiVector}) where Lay = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
@inline copy(L::Ldiv{Lay,<:NormalizedBasisLayout}) where Lay<:AbstractBasisLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:NormalizedBasisLayout,Lay}) where Lay<:AbstractBasisLayout = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
@inline copy(L::Ldiv{Lay,<:NormalizedBasisLayout}) where Lay<:AbstractLazyLayout = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:NormalizedBasisLayout,Lay}) where Lay<:AbstractLazyLayout = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
@inline copy(L::Ldiv{<:NormalizedBasisLayout,Lay,<:Any,<:AbstractQuasiVector}) where Lay<:AbstractLazyLayout = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
@inline copy(L::Ldiv{ApplyLayout{typeof(*)},<:NormalizedBasisLayout}) = copy(Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}(L.A, L.B))
for Lay in (:(ApplyLayout{typeof(*)}),:(BroadcastLayout{typeof(+)}),:(BroadcastLayout{typeof(-)}))
    @eval begin
        @inline copy(L::Ldiv{<:NormalizedBasisLayout,$Lay}) = copy(Ldiv{ApplyLayout{typeof(*)},$Lay}(L.A, L.B))
        @inline copy(L::Ldiv{<:NormalizedBasisLayout,$Lay,<:Any,<:AbstractQuasiVector}) = copy(Ldiv{ApplyLayout{typeof(*)},$Lay}(L.A, L.B))
    end
end

# want to use special re-expansion routines without expanding Normalized basis
@inline copy(L::Ldiv{<:NormalizedBasisLayout,BroadcastLayout{typeof(*)}}) = copy(Ldiv{BasisLayout,BroadcastLayout{typeof(*)}}(L.A, L.B))
@inline copy(L::Ldiv{<:NormalizedBasisLayout,BroadcastLayout{typeof(*)},<:Any,<:AbstractQuasiVector}) = copy(Ldiv{BasisLayout,BroadcastLayout{typeof(*)}}(L.A, L.B))

###
# show
###
show(io::IO, Q::Normalized) = print(io, "Normalized($(Q.P))")
show(io::IO, ::MIME"text/plain", Q::Normalized) = show(io, Q)




"""
    OrthonormalWeighted(P)

is the orthonormal with respect to L^2 basis given by
`sqrt.(orthogonalityweight(P)) .* Normalized(P)`.
"""
struct OrthonormalWeighted{T, PP<:AbstractQuasiMatrix{T}} <: Basis{T}
    P::Normalized{T, PP}
end

function OrthonormalWeighted(P)
    Q = normalized(P)
    OrthonormalWeighted{eltype(Q),typeof(P)}(Q)
end

axes(Q::OrthonormalWeighted) = axes(Q.P)
copy(Q::OrthonormalWeighted) = Q

==(A::OrthonormalWeighted, B::OrthonormalWeighted) = A.P == B.P

function getindex(Q::OrthonormalWeighted, x::Union{Number,AbstractVector}, jr::Union{Number,AbstractVector})
    w = orthogonalityweight(Q.P)
    sqrt.(w[x]) .* Q.P[x,jr]
end
broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, Q::OrthonormalWeighted) = Q * (Q.P \ (x .* Q.P))


abstract type AbstractWeighted{T} <: Basis{T} end

MemoryLayout(::Type{<:AbstractWeighted}) = WeightedBasisLayout()
ContinuumArrays.unweightedbasis(wP::AbstractWeighted) = wP.P
\(w_A::AbstractWeighted, w_B::AbstractWeighted) = convert(WeightedOrthogonalPolynomial, w_A) \ convert(WeightedOrthogonalPolynomial, w_B)
\(w_A::AbstractWeighted, B::AbstractQuasiArray) = convert(WeightedOrthogonalPolynomial, w_A) \ B
\(A::AbstractQuasiArray, w_B::AbstractWeighted) = A \ convert(WeightedOrthogonalPolynomial, w_B)

"""
    Weighted(P)

is equivalent to `orthogonalityweight(P) .* P`
"""
struct Weighted{T, PP<:AbstractQuasiMatrix{T}} <: AbstractWeighted{T}
    P::PP
end

axes(Q::Weighted) = axes(Q.P)
copy(Q::Weighted) = Q

==(A::Weighted, B::Weighted) = A.P == B.P

weight(wP::Weighted) = orthogonalityweight(wP.P)

convert(::Type{WeightedOrthogonalPolynomial}, P::Weighted) = weight(P) .* unweightedbasis(P)

getindex(Q::Weighted, x::Union{Number,AbstractVector}, jr::Union{Number,AbstractVector}) = weight(Q)[x] .* unweightedbasis(Q)[x,jr]
broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, Q::Weighted) = Q * (Q.P \ (x .* Q.P))


@simplify *(Ac::QuasiAdjoint{<:Any,<:Weighted}, wB::Weighted) = 
    convert(WeightedOrthogonalPolynomial, parent(Ac))' * convert(WeightedOrthogonalPolynomial, wB)

summary(io::IO, Q::Weighted) = print(io, "Weighted($(Q.P))")