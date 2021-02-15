# We roughly follow DLMF notation
# Q[x,n] = k[n] * x^(n-1)
# Note that we have
# Q[x,n] = (1/γ[n] - β[n-1]/γ[n]) * x * Q[x,n-1] - γ[n-1]/γ[n] * Q[x,n]
#
#

function lanczos!(Ns, X::AbstractMatrix{T}, W::AbstractMatrix{T}, γ::AbstractVector{T}, β::AbstractVector{T}, R::AbstractMatrix{T}) where T
    for n = Ns
        v = view(R,:,n);
        p1 = view(R,:,n-1);
        muladd!(one(T), X, p1, zero(T), v); # TODO: `mul!(v, X, p1)`
        β[n-1] = -dot(v,W,p1)
        axpy!(β[n-1],p1,v);
        if n > 2
            p0 = view(R,:,n-2)
            axpy!(-γ[n-1],p0,v)
        end
        γ[n] = sqrt(dot(v,W,v));
        lmul!(inv(γ[n]), v)
    end
    γ,β,R
end

const PaddedVector{T} = CachedVector{T,Vector{T},Zeros{T,1,Tuple{OneToInf{Int}}}}
const PaddedMatrix{T} = CachedMatrix{T,Matrix{T},Zeros{T,2,NTuple{2,OneToInf{Int}}}}

mutable struct LanczosData{T,XX,WW}
    X::XX
    W::WW
    γ::PaddedVector{T}
    β::PaddedVector{T}
    R::UpperTriangular{T,PaddedMatrix{T}}
    ncols::Int

    function LanczosData{T,XX,WW}(X, W, γ, β, R) where {T,XX,WW}
        R[1,1] = 1;
        p0 = view(R,:,1);
        γ[1] = sqrt(dot(p0,W,p0))
        lmul!(inv(γ[1]), p0)
        new{T,XX,WW}(X, W, γ, β, R, 1)
    end
end

LanczosData(X::XX, W::WW, γ::AbstractVector{T}, β, R) where {T,XX,WW} = LanczosData{T,XX,WW}(X, W, γ, β, R)
LanczosData(X::AbstractMatrix{T}, W::AbstractMatrix{T}) where T = LanczosData(X, W, zeros(T,∞), zeros(T,∞), UpperTriangular(zeros(T,∞,∞)))

function LanczosData(w::AbstractQuasiVector, P::AbstractQuasiMatrix)
    x = axes(P,1)
    wP = weighted(P)
    X = jacobimatrix(P)
    W = Clenshaw(P * (wP \ w), P)
    LanczosData(X, W)
end

function resizedata!(L::LanczosData, N)
    N ≤ L.ncols && return L
    resizedata!(L.R, N, N)
    resizedata!(L.γ, N)
    resizedata!(L.β, N)
    lanczos!(L.ncols+1:N, L.X, L.W, L.γ, L.β, L.R)
    L.ncols = N
    L
end

struct LanczosConversion{T,XX,WW} <: LayoutMatrix{T}
    data::LanczosData{T,XX,WW}
end

size(::LanczosConversion) = (∞,∞)
bandwidths(::LanczosConversion) = (0,∞)
colsupport(L::LanczosConversion, j) = 1:maximum(j)
rowsupport(L::LanczosConversion, j) = minimum(j):∞

function _lanczosconversion_getindex(R, k, j)
    resizedata!(R.data, max(maximum(k), maximum(j)))
    R.data.R[k,j]
end

getindex(R::LanczosConversion, k::Integer, j::Integer) = _lanczosconversion_getindex(R, k, j)
getindex(R::LanczosConversion, k::AbstractUnitRange, j::AbstractUnitRange) = _lanczosconversion_getindex(R, k, j)

inv(R::LanczosConversion) = ApplyArray(inv, R)


Base.BroadcastStyle(::Type{<:LanczosConversion}) = LazyArrays.LazyArrayStyle{2}()

struct LanczosConversionLayout <: AbstractLazyLayout end

LazyArrays.simplifiable(::Mul{LanczosConversionLayout,<:PaddedLayout}) = Val(true)
function copy(M::Mul{LanczosConversionLayout,<:PaddedLayout})
    resizedata!(M.A.data, maximum(colsupport(M.B)))
    M.A.data.R * M.B
end

function copy(M::Ldiv{LanczosConversionLayout,<:PaddedLayout})
    resizedata!(M.A.data, maximum(colsupport(M.B)))
    M.A.data.R \ M.B
end

function getindex(L::Ldiv{LanczosConversionLayout,<:AbstractBandedLayout}, ::Colon, j::Integer)
    m = maximum(colrange(L.B,j))
    [L.A[1:m,1:m] \ L.B[1:m,j]; Zeros{eltype(L)}(∞)]
end

MemoryLayout(::Type{<:LanczosConversion}) = LanczosConversionLayout()
triangulardata(R::LanczosConversion) = R
sublayout(::LanczosConversionLayout, ::Type{<:Tuple{KR,Integer}}) where KR = 
    sublayout(PaddedLayout{UnknownLayout}(), Tuple{KR})

function sub_paddeddata(::LanczosConversionLayout, S::SubArray{<:Any,1,<:AbstractMatrix})
    P = parent(S)
    (kr,j) = parentindices(S)
    resizedata!(P.data, j)
    paddeddata(view(UpperTriangular(P.data.R.data.data), kr, j))
end

# struct LanczosJacobiMatrix{T,XX,WW} <: AbstractBandedMatrix{T}
#     data::LanczosData{T,XX,WW}
# end

struct LanczosJacobiBand{T,XX,WW} <: LazyVector{T}
    data::LanczosData{T,XX,WW}
    diag::Symbol
end

size(P::LanczosJacobiBand) = (∞,)
resizedata!(A::LanczosJacobiBand, n) = resizedata!(A.data, n)


# γ[n+1]*Q[x,n+1] + β[n]*Q[x,n] + γ[n-1]*Q[x,n-1] = x  * Q[x,n]
function _lanczos_getindex(C::LanczosJacobiBand, I)
    resizedata!(C, maximum(I)+1)
    if C.diag == :du
        C.data.γ.data[I .+ 1]
    else # :d
        -C.data.β.data[I]
    end
end

getindex(A::LanczosJacobiBand, I::Integer) = _lanczos_getindex(A, I)
getindex(A::LanczosJacobiBand, I::AbstractVector) = _lanczos_getindex(A, I)
getindex(K::LanczosJacobiBand, k::AbstractInfUnitRange) = view(K, k)
getindex(K::SubArray{<:Any,1,<:LanczosJacobiBand}, k::AbstractInfUnitRange) = view(K, k)

copy(A::LanczosJacobiBand) = A # immutable

struct LanczosRecurrence{ABC,T,XX,WW} <: LazyVector{T}
    data::LanczosData{T,XX,WW}
end

LanczosRecurrence{ABC}(data::LanczosData{T,XX,WW}) where {ABC,T,XX,WW} = LanczosRecurrence{ABC,T,XX,WW}(data)

size(P::LanczosRecurrence) = (∞,)

resizedata!(A::LanczosRecurrence, n) = resizedata!(A.data, n)



# Q[x,n] = (x/γ[n] - β[n-1]/γ[n])  * Q[x,n-1] - γ[n-1]/γ[n] * Q[x,n]


function _lanczos_getindex(A::LanczosRecurrence{:A}, I)
    resizedata!(A, maximum(I)+1)
    inv.(A.data.γ.data[I .+ 1])
end

function _lanczos_getindex(B::LanczosRecurrence{:B}, I)
    resizedata!(B, maximum(I)+1)
    B.data.β.data[I] ./ B.data.γ.data[I .+ 1]
end

function _lanczos_getindex(C::LanczosRecurrence{:C}, I)
    resizedata!(C, maximum(I)+1)
    C.data.γ.data[I] ./ C.data.γ.data[I  .+ 1]
end

getindex(A::LanczosRecurrence, I::Integer) = _lanczos_getindex(A, I)
getindex(A::LanczosRecurrence, I::AbstractVector) = _lanczos_getindex(A, I)
getindex(K::LanczosRecurrence, k::InfRanges{<:Integer}) = view(K, k)
getindex(K::SubArray{<:Any,1,<:LanczosRecurrence}, k::InfRanges{<:Integer}) = view(K, k)


struct LanczosPolynomial{T,XX,WW,Weight,Basis} <: OrthogonalPolynomial{T}
    w::Weight # Weight of orthogonality
    P::Basis # Basis we use to represent the OPs
    data::LanczosData{T,XX,WW}
end

==(A::LanczosPolynomial, B::LanczosPolynomial) = A.w == B.w
==(::LanczosPolynomial, ::OrthogonalPolynomial) = false # TODO: fix
==(::OrthogonalPolynomial, ::LanczosPolynomial) = false # TODO: fix
==(::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}, ::LanczosPolynomial) = false # TODO: fix
==(::LanczosPolynomial, ::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}) = false # TODO: fix


normalize(Q::LanczosPolynomial) = Q
normalize(Q::AbstractQuasiMatrix) = Normalized(Q)

OrthogonalPolynomial(w::AbstractQuasiVector) = LanczosPolynomial(w)
orthogonalpolynomial(w::AbstractQuasiVector) = OrthogonalPolynomial(w)
orthogonalpolynomial(w::SubQuasiArray) = orthogonalpolynomial(parent(w))[parentindices(w)[1],:]
orthonormalpolynomial(w::AbstractQuasiVector) = normalize(orthogonalpolynomial(w))

function LanczosPolynomial(w_in::AbstractQuasiVector, P::AbstractQuasiMatrix)
    Q = normalize(P)
    wQ = weighted(Q)
    w = wQ * (wQ \ w_in) # expand weight in basis
    LanczosPolynomial(w, Q, LanczosData(w, Q))
end

LanczosPolynomial(w::AbstractQuasiVector) = LanczosPolynomial(w, orthonormalpolynomial(singularities(w)))

orthogonalityweight(Q::LanczosPolynomial) = Q.w

axes(Q::LanczosPolynomial) = (axes(Q.w,1),OneToInf())

_p0(Q::LanczosPolynomial) = inv(Q.data.γ[1])*_p0(Q.P)

recurrencecoefficients(Q::LanczosPolynomial) = LanczosRecurrence{:A}(Q.data),LanczosRecurrence{:B}(Q.data),LanczosRecurrence{:C}(Q.data)
jacobimatrix(Q::LanczosPolynomial) = SymTridiagonal(LanczosJacobiBand(Q.data, :d), LanczosJacobiBand(Q.data, :du))

Base.summary(io::IO, Q::LanczosPolynomial{T}) where T = print(io, "LanczosPolynomial{$T} with weight with singularities $(singularities(Q.w))")
Base.show(io::IO, Q::LanczosPolynomial{T}) where T = summary(io, Q)

Base.array_summary(io::IO, C::SymTridiagonal{T,<:LanczosJacobiBand}, inds::Tuple{Vararg{OneToInf{Int}}}) where T =
    print(io, Base.dims2string(length.(inds)), " SymTridiagonal{$T} Jacobi operator from Lanczos")

Base.array_summary(io::IO, C::LanczosConversion{T}, inds::Tuple{Vararg{OneToInf{Int}}}) where T =
    print(io, Base.dims2string(length.(inds)), " LanczosConversion{$T}")

Base.array_summary(io::IO, C::LanczosJacobiBand{T}, inds::Tuple{Vararg{OneToInf{Int}}}) where T =
    print(io, Base.dims2string(length.(inds)), " LanczosJacobiBand{$T}")


# Sometimes we want to expand out, sometimes we don't

QuasiArrays.ApplyQuasiArray(Q::LanczosPolynomial) = ApplyQuasiArray(*, arguments(ApplyLayout{typeof(*)}(), Q)...)


function \(A::LanczosPolynomial{T}, B::LanczosPolynomial{V}) where {T,V}
    A == B && return Eye{promote_type(T,V)}(∞)
    inv(LanczosConversion(A.data)) * (A.P \ B.P)  * LanczosConversion(B.data)
end
\(A::OrthogonalPolynomial, Q::LanczosPolynomial) = (A \ Q.P) * LanczosConversion(Q.data)
\(A::Normalized, Q::LanczosPolynomial) = (A \ Q.P) * LanczosConversion(Q.data)
\(Q::LanczosPolynomial, A::OrthogonalPolynomial) = inv(LanczosConversion(Q.data)) * (Q.P \ A)
\(Q::LanczosPolynomial, A::Normalized) = inv(LanczosConversion(Q.data)) * (Q.P \ A)
\(Q::LanczosPolynomial, A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}) = inv(LanczosConversion(Q.data)) * (Q.P \ A)
\(A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}, Q::LanczosPolynomial) = (A \ Q.P) * LanczosConversion(Q.data)

ArrayLayouts.mul(Q::LanczosPolynomial, C::AbstractArray) = ApplyQuasiArray(*, Q, C)
function ldiv(Qn::SubQuasiArray{<:Any,2,<:LanczosPolynomial,<:Tuple{<:Inclusion,<:Any}}, C::AbstractQuasiArray)
    _,jr = parentindices(Qn)
    Q = parent(Qn)
    LanczosConversion(Q.data)[jr,jr] \ (Q.P[:,jr] \ C)
end
arguments(::ApplyLayout{typeof(*)}, Q::LanczosPolynomial) = Q.P, LanczosConversion(Q.data)
LazyArrays._mul_arguments(Q::LanczosPolynomial) = arguments(ApplyLayout{typeof(*)}(), Q)
LazyArrays._mul_arguments(Q::QuasiAdjoint{<:Any,<:LanczosPolynomial}) = arguments(ApplyLayout{typeof(*)}(), Q)

\(A::LanczosPolynomial, x::AbstractQuasiVector) = ApplyQuasiArray(A) \ x
