"""
    ConnectionMatrix(A, B)

represents the upper triangular conversion matrix `A \\ B` between two classical orthogonal polynomial bases
whose entries are dense, e.g. `ChebyshevT() \\ Legendre()`. Multiplying or dividing coefficients with
finitely many non-zeros uses the Toeplitz-dot-Hankel transforms in FastTransforms.jl.
"""
struct ConnectionMatrix{T, AA, BB} <: LayoutMatrix{T}
    A::AA
    B::BB
end

ConnectionMatrix{T}(A, B) where T = ConnectionMatrix{T,typeof(A),typeof(B)}(A, B)
ConnectionMatrix(A, B) = ConnectionMatrix{promote_type(eltype(A), eltype(B))}(A, B)

size(::ConnectionMatrix) = (ℵ₀, ℵ₀)
axes(::ConnectionMatrix) = (oneto(ℵ₀), oneto(ℵ₀))
copy(C::ConnectionMatrix) = C
inv(C::ConnectionMatrix{T}) where T = ConnectionMatrix{T}(C.B, C.A)

struct ConnectionLayout <: AbstractLazyLayout end
MemoryLayout(::Type{<:ConnectionMatrix}) = ConnectionLayout()

colsupport(::ConnectionLayout, C, j) = oneto(maximum(j))
rowsupport(::ConnectionLayout, C, k) = minimum(k):∞

###
# connection_transform(A, B, c) returns the coefficients in A of B*c, applied to the columns of c
###

connection_transform(A, B, c::AbstractVecOrMat) = connection_transform(A, B, convert(AbstractArray{eltype(ConnectionMatrix(A,B))}, c))
function connection_transform(A, B, c::AbstractVecOrMat{T}) where T<:Complex
    complex.(connection_transform(A, B, convert(AbstractArray{real(T)}, real(c))),
             connection_transform(A, B, convert(AbstractArray{real(T)}, imag(c))))
end

for Typ in (:Float16, :Float32, :Float64, :BigFloat, :AbstractFloat)
    @eval function connection_transform(A, B, c::AbstractVecOrMat{<:$Typ})
        isempty(c) && return copy(c)
        _connection_transform(A, B, c)
    end
end

_connection_transform(::ChebyshevT, ::Legendre, c) = th_leg2cheb(c, 1)
_connection_transform(::Legendre, ::ChebyshevT, c) = th_cheb2leg(c, 1)
_connection_transform(A::Ultraspherical, B::Ultraspherical, c::AbstractVecOrMat{T}) where T = th_ultra2ultra(c, convert(T,B.λ), convert(T,A.λ), 1)
_connection_transform(A::Jacobi, B::Jacobi, c::AbstractVecOrMat{T}) where T = th_jac2jac(c, convert(T,B.a), convert(T,B.b), convert(T,A.a), convert(T,A.b), 1)
_connection_transform(::ChebyshevT, B::Jacobi, c::AbstractVecOrMat{T}) where T = th_jac2cheb(c, convert(T,B.a), convert(T,B.b), 1)
_connection_transform(A::Jacobi, ::ChebyshevT, c::AbstractVecOrMat{T}) where T = th_cheb2jac(c, convert(T,A.a), convert(T,A.b), 1)

# Ultraspherical(λ) is a rescaling of Jacobi(λ-1/2, λ-1/2)
_ultraspherical_jacobi_scale(C::Ultraspherical, n) = (C[1,oneto(n)] ./ Jacobi(C)[1,oneto(n)])
_connection_transform(A::ChebyshevT, B::Ultraspherical, c) = _connection_transform(A, Jacobi(B), _ultraspherical_jacobi_scale(B, size(c,1)) .* c)
_connection_transform(A::Ultraspherical, B::ChebyshevT, c) = _connection_transform(Jacobi(A), B, c) ./ _ultraspherical_jacobi_scale(A, size(c,1))

###
# multiplication and division of coefficients with finitely many non-zeros
###

simplifiable(::Mul{ConnectionLayout,<:PaddedColumns}) = Val(true)
function copy(M::Mul{ConnectionLayout,<:PaddedColumns})
    C, b = M.A, M.B
    padrows(connection_transform(C.A, C.B, paddeddata(b)), axes(C,1))
end
copy(L::Ldiv{ConnectionLayout,<:PaddedColumns}) = inv(L.A) * L.B

###
# entries
###

# columns of the identity, so that finite sections are computed with a single transform
_connection_block(C::ConnectionMatrix{T}, jr) where T = connection_transform(C.A, C.B, Matrix{T}(I, maximum(jr; init=0), maximum(jr; init=0))[:, jr])

function _connection_rows(C::ConnectionMatrix{T}, kr, jr) where T
    B = _connection_block(C, jr)
    [k ≤ size(B,1) ? B[k,j] : zero(T) for k in kr, j in axes(B,2)]
end

sub_materialize(::ConnectionLayout, V::AbstractMatrix, ::Tuple{OneTo{Int},OneTo{Int}}) = _connection_rows(parent(V), parentindices(V)...)
sub_materialize(::ConnectionLayout, V::AbstractVector, ::Tuple{OneTo{Int}}) = vec(_connection_rows(parent(V), parentindices(V)[1], parentindices(V)[2]:parentindices(V)[2]))
function sub_materialize(::ConnectionLayout, V::AbstractVector, ::Tuple{OneToInf{Int}})
    C = parent(V)
    j = parentindices(V)[2]
    Vcat(vec(_connection_block(C, j:j)), Zeros{eltype(C)}(∞))
end

getindex(C::ConnectionMatrix{T}, k::Integer, j::Integer) where T = k > j ? zero(T) : connection_getindex(C.A, C.B, T, k, j)
connection_getindex(A, B, ::Type{T}, k, j) where T = _connection_block(ConnectionMatrix{T}(A, B), j:j)[k]

# closed form for entries of Chebyshev in terms of Ultraspherical:
# C_n^(λ)(cos θ) = Σ_j (λ)_j (λ)_{n-j} / (j! (n-j)!) cos((n-2j)θ)
function _chebyshev_ultraspherical_getindex(λ::T, k, j) where T
    iseven(k) == iseven(j) || return zero(T)
    a = Λ(convert(T,j-k)/2, λ, one(T)) * Λ(convert(T,k+j-2)/2, λ, one(T)) / gamma(λ)^2
    k == 1 ? a : 2a
end
connection_getindex(::ChebyshevT, B::Ultraspherical, ::Type{T}, k, j) where T = _chebyshev_ultraspherical_getindex(convert(T, B.λ), k, j)
connection_getindex(::ChebyshevT, ::Legendre, ::Type{T}, k, j) where T = _chebyshev_ultraspherical_getindex(one(T)/2, k, j)
