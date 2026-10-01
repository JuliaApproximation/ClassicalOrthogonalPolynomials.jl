"""
    ConnectionMatrix(A, B)

represents the upper triangular conversion matrix `A \\ B` between two classical orthogonal polynomial bases
whose entries are dense, e.g. `ChebyshevT() \\ Legendre()`. Multiplying or dividing coefficients with
finitely many non-zeros uses the Toeplitz-dot-Hankel transforms in FastTransforms.jl, whereas entries use explicit formulas.
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
Base.BroadcastStyle(::Type{<:ConnectionMatrix}) = LazyArrays.LazyArrayStyle{2}()

colsupport(::ConnectionLayout, C, j) = oneto(maximum(j))
rowsupport(::ConnectionLayout, C, k) = minimum(k):∞

# transposes arise when indexing rows of products, e.g. stieltjes(Legendre(), z) * (Legendre() \ ChebyshevT())
struct TransposeConnectionLayout <: AbstractLazyLayout end
transposelayout(::ConnectionLayout) = TransposeConnectionLayout()
transposelayout(::TransposeConnectionLayout) = ConnectionLayout()

colsupport(::TransposeConnectionLayout, C, j) = minimum(j):∞
rowsupport(::TransposeConnectionLayout, C, k) = oneto(maximum(k))

# finite sections, which are computed from explicit formulas for the entries
struct ConnectionBlockLayout <: AbstractLazyLayout end
sublayout(::Union{ConnectionLayout,TransposeConnectionLayout}, ::Type{<:Tuple{AbstractUnitRange{Int},Union{Int,AbstractUnitRange{Int}}}}) = ConnectionBlockLayout()

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
# entries, which use explicit formulas
###

getindex(C::ConnectionMatrix{T}, k::Integer, j::Integer) where T = k > j ? zero(T) : connection_getindex(C.A, C.B, T, k, j)

_connection_rows(C::ConnectionMatrix{T}, kr, jr) where T = connection_block(C.A, C.B, T, kr, jr)
_connection_rows(C::Union{Adjoint{<:Real,<:ConnectionMatrix},Transpose{<:Any,<:ConnectionMatrix}}, kr, jr) = transpose(_connection_rows(parent(C), jr, kr))

connection_block(A, B, ::Type{T}, kr, jr) where T = T[k > j ? zero(T) : connection_getindex(A, B, T, k, j) for k in kr, j in jr]

sub_materialize(::ConnectionBlockLayout, V::AbstractMatrix, ::Tuple{OneTo{Int},OneTo{Int}}) = _connection_rows(parent(V), parentindices(V)...)
sub_materialize(::ConnectionBlockLayout, V::AbstractVector, ::Tuple{OneTo{Int}}) = vec(_connection_rows(parent(V), parentindices(V)[1], parentindices(V)[2]:parentindices(V)[2]))
function sub_materialize(::ConnectionBlockLayout, V::SubArray{<:Any,1,<:ConnectionMatrix}, ::Tuple{OneToInf{Int}})
    C = parent(V)
    j = parentindices(V)[2]
    Vcat(vec(_connection_rows(C, oneto(j), j:j)), Zeros{eltype(C)}(∞))
end

# materialize finite sections of broadcasted arguments, e.g. in (Legendre() \ ChebyshevT()) / 2, as entries are expensive
const ConnectionMatrices = Union{ConnectionMatrix, Adjoint{<:Real,<:ConnectionMatrix}, Transpose{<:Any,<:ConnectionMatrix}}
LazyArrays._viewifmutable(C::ConnectionMatrices, kr, jr) = isfinite(length(kr)) && isfinite(length(jr)) ? C[kr, jr] : view(C, kr, jr)

# (x)_m / m!
function _pochhammer_factorial(x::T, m) where T
    if isapproxinteger(x) && x ≤ 0
        p = round(Int, -x)
        return m > p ? zero(T) : convert(T, (-1)^m * binomial(p, m))
    end
    Λ(convert(T, m), x, one(T)) / gamma(x)
end

# (2k+λ)Γ(k+λ)/Γ(k+μ), using λΓ(λ) = Γ(λ+1) for k = 0 to avoid 0*Inf when λ = 0
_twoklambda(k, λ::T, μ) where T = iszero(k) ? Λ(zero(T), λ+1, μ) : (2k+λ)*Λ(convert(T,k), λ, μ)

# Ultraspherical family, where ChebyshevT corresponds to λ = 0 and Legendre to λ = 1/2
const UltrasphericalFamily = Union{ChebyshevT,Legendre,Ultraspherical}
_ultraspherical_parameter(::ChebyshevT{T}) where T = zero(T)
_ultraspherical_parameter(::Legendre{T}) where T = one(T)/2
_ultraspherical_parameter(C::Ultraspherical) = C.λ

connection_getindex(A::UltrasphericalFamily, B::UltrasphericalFamily, ::Type{T}, k, j) where T =
    _ultraspherical_connection(convert(T, _ultraspherical_parameter(A)), convert(T, _ultraspherical_parameter(B)), k-1, j-1)

# coefficient of C_m^(μ) in C_n^(λ) (DLMF 18.18.16), where λ = 0 or μ = 0 correspond to Chebyshev T_n = n/2 * lim_{λ → 0} C_n^(λ)/λ
function _ultraspherical_connection(μ::T, λ::T, m, n) where T
    (m ≤ n && iseven(n-m)) || return zero(T)
    ℓ = (n-m) ÷ 2
    if iszero(μ)
        a = Λ(convert(T,ℓ), λ, one(T)) * Λ(convert(T,n-ℓ), λ, one(T)) / gamma(λ)^2
        m == 0 ? a : 2a
    elseif iszero(λ)
        n == 0 ? one(T) : n * _pochhammer_factorial(-μ, ℓ) * Λ(convert(T,n-ℓ), zero(T), μ+1) * gamma(μ) * (μ+m) / 2
    else
        _pochhammer_factorial(λ-μ, ℓ) * Λ(convert(T,n-ℓ), λ, μ+1) * gamma(μ) * (μ+m) / gamma(λ)
    end
end

# coefficient of P_k^(α,β) in P_n^(γ,β), which is a balanced 3F2 summed by Pfaff–Saalschütz.
# n = 0 is special-cased as the formula is 0*Inf when γ+β+1 = 0, e.g. when converting to Chebyshev
function _jacobi_connection_a(α::T, β::T, γ::T, k, n) where T
    n == 0 && return one(T)
    _twoklambda(k, α+β+1, β+1) * Λ(convert(T,n+k), γ+β+1, α+β+2) * _pochhammer_factorial(γ-α, n-k) * Λ(convert(T,n), β+1, γ+β+1)
end
# coefficient of P_k^(α,β) in P_n^(α,δ), using P_n^(a,b)(-x) = (-1)^n P_n^(b,a)(x)
_jacobi_connection_b(α::T, β::T, δ::T, k, n) where T = (-1)^(n+k) * _jacobi_connection_a(β, α, δ, k, n)

# P_n(1) relative to the Jacobi polynomial with the same weight
_value_at_one(::Union{ChebyshevT{T},Legendre{T}}, n) where T = one(T)
_value_at_one(C::Ultraspherical, n) = _pochhammer_factorial(2C.λ, n)
_value_at_one(P::Jacobi, n) = _pochhammer_factorial(P.a+1, n)
_jacobi_normalization(P, ::Type{T}, n) where T = convert(T, _value_at_one(P, n) / _value_at_one(Jacobi(P), n))

# Jacobi(α,β) \ Jacobi(γ,δ) = (Jacobi(α,β) \ Jacobi(γ,β)) * (Jacobi(γ,β) \ Jacobi(γ,δ))
_jacobi_connection_a_block(α::T, β, γ, kr, jr) where T = T[k ≤ j ? _jacobi_connection_a(α, β, γ, k-1, j-1) : zero(T) for k in kr, j in jr]
_jacobi_connection_b_block(α::T, β, δ, kr, jr) where T = T[k ≤ j ? _jacobi_connection_b(α, β, δ, k-1, j-1) : zero(T) for k in kr, j in jr]

function _jacobi_connection(α::T, β::T, γ::T, δ::T, kr, jr) where T
    γ == α && return _jacobi_connection_b_block(α, β, δ, kr, jr)
    δ == β && return _jacobi_connection_a_block(α, β, γ, kr, jr)
    ir = minimum(kr; init=1):maximum(jr; init=0)
    _jacobi_connection_a_block(α, β, γ, kr, ir) * _jacobi_connection_b_block(γ, β, δ, ir, jr)
end

function connection_block(A::AbstractJacobi, B::AbstractJacobi, ::Type{T}, kr, jr) where T
    JA, JB = Jacobi(A), Jacobi(B)
    M = _jacobi_connection(convert(T, JA.a), convert(T, JA.b), convert(T, JB.a), convert(T, JB.b), kr, jr)
    M ./ _jacobi_normalization.(Ref(A), T, kr .- 1) .* transpose(_jacobi_normalization.(Ref(B), T, jr .- 1))
end
connection_getindex(A::AbstractJacobi, B::AbstractJacobi, ::Type{T}, k, j) where T = connection_block(A, B, T, k:k, j:j)[1]
connection_block(A::UltrasphericalFamily, B::UltrasphericalFamily, ::Type{T}, kr, jr) where T = T[k > j ? zero(T) : connection_getindex(A, B, T, k, j) for k in kr, j in jr]
