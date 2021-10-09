
##
# For Chebyshev T. Note the shift in indexing is fine due to the AbstractFill
##
Base.@propagate_inbounds _forwardrecurrence_next(n, A::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, B::Zeros, C::Ones, x, p0, p1) = 
    _forwardrecurrence_next(n, A.args[2], B, C, x, p0, p1)

Base.@propagate_inbounds _clenshaw_next(n, A::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, B::Zeros, C::Ones, x, c, bn1, bn2) = 
    _clenshaw_next(n, A.args[2], B, C, x, c, bn1, bn2)

# Assume 1 normalization
_p0(A) = one(eltype(A))

function initiateforwardrecurrence(N, A, B, C, x, μ)
    T = promote_type(eltype(A), eltype(B), eltype(C), typeof(x))
    p0 = convert(T, μ)
    p1 = convert(T, muladd(A[1],x,B[1])*p0)
    @inbounds for n = 2:N
        p1,p0 = _forwardrecurrence_next(n, A, B, C, x, p0, p1),p1
    end
    p0,p1
end

for (get, vie) in ((:getindex, :view), (:(Base.unsafe_getindex), :(Base.unsafe_view)))
    @eval begin
        Base.@propagate_inbounds @inline $get(P::OrthogonalPolynomial{T}, x::Number, n::OneTo) where T =
            copyto!(Vector{T}(undef,length(n)), $vie(P, x, n))

        $get(P::OrthogonalPolynomial{T}, x::AbstractVector, n::AbstractUnitRange{Int}) where T =
            copyto!(Matrix{T}(undef,length(x),length(n)), $vie(P, x, n))
    end
end

function copyto!(dest::AbstractVector, V::SubArray{<:Any,1,<:OrthogonalPolynomial,<:Tuple{<:Number,<:OneTo}})
    P = parent(V)
    x,n = parentindices(V)
    A,B,C = recurrencecoefficients(P)
    forwardrecurrence!(dest, A, B, C, x, _p0(P))
end

function forwardrecurrence_copyto!(dest::AbstractMatrix, V)
    checkbounds(dest, axes(V)...)
    P = parent(V)
    xr,jr = parentindices(V)
    A,B,C = recurrencecoefficients(P)
    shift = first(jr)
    Ã,B̃,C̃ = A[shift:∞],B[shift:∞],C[shift:∞]
    for (k,x) = enumerate(xr)
        p0, p1 = initiateforwardrecurrence(shift, A, B, C, x, _p0(P))
        _forwardrecurrence!(view(dest,k,:), Ã, B̃, C̃, x, p0, p1)
    end
    dest
end
copyto!(dest::AbstractMatrix, V::SubArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{<:AbstractVector,<:AbstractUnitRange}}) = forwardrecurrence_copyto!(dest, V)
copyto!(dest::LayoutMatrix, V::SubArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{<:AbstractVector,<:AbstractUnitRange}}) = forwardrecurrence_copyto!(dest, V)

function copyto!(dest::AbstractVector, V::SubArray{<:Any,1,<:OrthogonalPolynomial,<:Tuple{<:Number,<:UnitRange}})
    checkbounds(dest, axes(V)...)
    P = parent(V)
    x,jr = parentindices(V)
    A,B,C = recurrencecoefficients(P)
    shift = first(jr)
    Ã,B̃,C̃ = A[shift:∞],B[shift:∞],C[shift:∞]
    p0, p1 = initiateforwardrecurrence(shift, A, B, C, x, _p0(P))
    _forwardrecurrence!(dest, Ã, B̃, C̃, x, p0, p1)
    dest
end

getindex(P::OrthogonalPolynomial, x::Number, n::AbstractVector) = layout_getindex(P, x, n)
getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractVector) = layout_getindex(P, x, n)
getindex(P::SubArray{<:Any,1,<:OrthogonalPolynomial}, x::AbstractVector) = layout_getindex(P, x)
getindex(P::OrthogonalPolynomial, x::Number, n::Number) = P[x,oneto(n)][end]

unsafe_layout_getindex(A...) = sub_materialize(Base.unsafe_view(A...))

Base.unsafe_getindex(P::OrthogonalPolynomial, x::Number, n::AbstractUnitRange) = unsafe_layout_getindex(P, x, n)
Base.unsafe_getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractUnitRange) = unsafe_layout_getindex(P, x, n)
Base.unsafe_getindex(P::OrthogonalPolynomial, x::Number, n::AbstractVector) = Base.unsafe_getindex(P,x,oneto(maximum(n)))[n]
Base.unsafe_getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractVector) = Base.unsafe_getindex(P,x,oneto(maximum(n)))[:,n]
Base.unsafe_getindex(P::OrthogonalPolynomial, x::AbstractVector, n::Number) = Base.unsafe_getindex(P, x, 1:n)[:,end]
Base.unsafe_getindex(P::OrthogonalPolynomial, x::Number, ::Colon) = Base.unsafe_getindex(P, x, axes(P,2))
Base.unsafe_getindex(P::OrthogonalPolynomial, x::Number, n::Number) = Base.unsafe_getindex(P,x,oneto(n))[end]

getindex(P::OrthogonalPolynomial, x::Number, jr::AbstractInfUnitRange{Int}) = view(P, x, jr)
getindex(P::OrthogonalPolynomial, x::AbstractVector, jr::AbstractInfUnitRange{Int}) = view(P, x, jr)
Base.unsafe_getindex(P::OrthogonalPolynomial{T}, x::Number, jr::AbstractInfUnitRange{Int}) where T = 
    BroadcastVector{T}(Base.unsafe_getindex, Ref(P), x, jr)

function getindex(P::BroadcastVector{<:Any,typeof(Base.unsafe_getindex), <:Tuple{Ref{<:OrthogonalPolynomial},Number,AbstractVector{Int}}}, kr::AbstractVector)
    Pr, x, jr = P.args
    Base.unsafe_getindex(Pr[], x, jr[kr])
end

###
# Clenshaw
###

function unsafe_getindex(f::Mul{<:AbstractOPLayout,<:PaddedLayout}, x::Number)
    P,c = f.A,f.B
    _p0(P)*clenshaw(paddeddata(c), recurrencecoefficients(P)..., x)
end

function unsafe_getindex(f::Mul{<:AbstractOPLayout,<:PaddedLayout}, x::Number, jr)
    P,c = f.A,f.B
    _p0(P)*clenshaw(view(paddeddata(c),:,jr), recurrencecoefficients(P)..., x)
end

Base.@propagate_inbounds function getindex(f::Mul{<:AbstractOPLayout,<:PaddedLayout}, x::Number, j...)
    @inbounds checkbounds(ApplyQuasiArray(*,f.A,f.B), x, j...)
    unsafe_getindex(f, x, j...)
end

# const ExpansionMatrix{T,P<:AbstractQuasiMatrix,M<:AbstractMatrix} = ApplyQuasiMatrix{T,typeof(*),<:Tuple{P,M}}

# function unsafe_getindex(f::ExpansionMatrix{<:Any,<:OrthogonalPolynomial}, x::Union{Number,AbstractVector{<:Number}}, jr::Union{Colon,AbstractVector{Int}})
#     P,c = arguments(f)
#     _p0(P)*clenshaw(view(paddeddata(c),:,jr), recurrencecoefficients(P)..., x)
# end


# Base.@propagate_inbounds function getindex(f::ExpansionMatrix{<:Any,<:OrthogonalPolynomial}, x::Union{Number,AbstractVector{<:Number}}, jr::Union{Colon,AbstractVector{Int}})
#     @inbounds checkbounds(f, x, jr)
#     unsafe_getindex(f, x, jr)
# end


# getindex(f::Expansion{T,<:OrthogonalPolynomial}, x::AbstractVector{<:Number}) where T = 
#     copyto!(Vector{T}(undef, length(x)), view(f, x))

# function copyto!(dest::AbstractVector{T}, v::SubArray{<:Any,1,<:Expansion{<:Any,<:OrthogonalPolynomial}, <:Tuple{AbstractVector{<:Number}}}) where T
#     f = parent(v)
#     (x,) = parentindices(v)
#     P,c = arguments(f)
#     clenshaw!(paddeddata(c), recurrencecoefficients(P)..., x, Fill(_p0(P), length(x)), dest)
# end

###
# Operator clenshaw
###


Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractFill, ::Zeros, C::Ones, x::AbstractMatrix, c, bn1::AbstractMatrix{T}, bn2::AbstractMatrix{T}) where T
    muladd!(getindex_value(A), x, bn1, -one(T), bn2)
    view(bn2,band(0)) .+= c[n]
    bn2
end

Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractVector, ::Zeros, C::AbstractVector, x::AbstractMatrix, c, bn1::AbstractMatrix{T}, bn2::AbstractMatrix{T}) where T
    muladd!(A[n], x, bn1, -C[n+1], bn2)
    view(bn2,band(0)) .+= c[n]
    bn2
end

Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractMatrix, c, bn1::AbstractMatrix{T}, bn2::AbstractMatrix{T}) where T
    # bn2 .= B[n] .* bn1 .- C[n+1] .* bn2
    lmul!(-C[n+1], bn2)
    BLAS.axpy!(B[n], bn1, bn2)
    muladd!(A[n], x, bn1, one(T), bn2)
    view(bn2,band(0)) .+= c[n]
    bn2
end

# Operator * f Clenshaw
Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractFill, ::Zeros, C::Ones, X::AbstractMatrix, c, f::AbstractVector, bn1::AbstractVector{T}, bn2::AbstractVector{T}) where T
    muladd!(getindex_value(A), X, bn1, -one(T), bn2)
    bn2 .+= c[n] .* f
    bn2
end

Base.@propagate_inbounds function _clenshaw_next!(n, A, ::Zeros, C, X::AbstractMatrix, c, f::AbstractVector, bn1::AbstractVector{T}, bn2::AbstractVector{T}) where T
    muladd!(A[n], X, bn1, -C[n+1], bn2)
    bn2 .+= c[n] .* f
    bn2
end

Base.@propagate_inbounds function _clenshaw_next!(n, A, B, C, X::AbstractMatrix, c, f::AbstractVector, bn1::AbstractVector{T}, bn2::AbstractVector{T}) where T
    bn2 .= B[n] .* bn1 .- C[n+1] .* bn2 .+ c[n] .* f
    muladd!(A[n], X, bn1, one(T), bn2)
    bn2
end

# allow special casing first arg, for ChebyshevT in ClassicalOrthogonalPolynomials
Base.@propagate_inbounds function _clenshaw_first!(A, ::Zeros, C, X, c, bn1, bn2) 
    muladd!(A[1], X, bn1, -C[2], bn2)
    view(bn2,band(0)) .+= c[1]
    bn2
end

Base.@propagate_inbounds function _clenshaw_first!(A, B, C, X, c, bn1, bn2) 
    lmul!(-C[2], bn2)
    BLAS.axpy!(B[1], bn1, bn2)
    muladd!(A[1], X, bn1, one(eltype(bn2)), bn2)
    view(bn2,band(0)) .+= c[1]
    bn2
end

Base.@propagate_inbounds function _clenshaw_first!(A, ::Zeros, C, X, c, f::AbstractVector, bn1, bn2) 
    muladd!(A[1], X, bn1, -C[2], bn2)
    bn2 .+= c[1] .* f
    bn2
end

Base.@propagate_inbounds function _clenshaw_first!(A, B, C, X, c, f::AbstractVector, bn1, bn2) 
    bn2 .= B[1] .* bn1 .- C[2] .* bn2 .+ c[1] .* f
    muladd!(A[1], X, bn1, one(eltype(bn2)), bn2)
    bn2
end

_clenshaw_op(::AbstractBandedLayout, Z, N) = BandedMatrix(Z, (N-1,N-1))

function clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix)
    N = length(c)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),eltype(X))
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    m = size(X,1)
    m == size(X,2) || throw(DimensionMismatch("X must be square"))
    N == 0 && return zero(T)
    bn2 = _clenshaw_op(MemoryLayout(X), Zeros{T}(m, m), N)
    bn1 = _clenshaw_op(MemoryLayout(X), c[N]*Eye{T}(m), N)
    _clenshaw_op!(c, A, B, C, X, bn1, bn2)
end

function clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix, f::AbstractVector)
    N = length(c)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),eltype(X))
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    m = size(X,1)
    m == size(X,2) || throw(DimensionMismatch("X must be square"))
    m == length(f) || throw(DimensionMismatch("Dimensions must match"))
    N == 0 && return zero(T)
    bn2 = zeros(T,m)
    bn1 = Vector{T}(undef,m)
    bn1 .= c[N] .* f
    _clenshaw_op!(c, A, B, C, X, f, bn1, bn2)
end

function _clenshaw_op!(c, A, B, C, X, bn1, bn2)
    N = length(c)
    N == 1 && return bn1
    @inbounds begin
        for n = N-1:-1:2
            bn1,bn2 = _clenshaw_next!(n, A, B, C, X, c, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first!(A, B, C, X, c, bn1, bn2)
    end
    bn1
end

function _clenshaw_op!(c, A, B, C, X, f::AbstractVector, bn1, bn2)
    N = length(c)
    N == 1 && return bn1
    @inbounds begin
        for n = N-1:-1:2
            bn1,bn2 = _clenshaw_next!(n, A, B, C, X, c, f, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first!(A, B, C, X, c, f, bn1, bn2)
    end
    bn1
end



"""
    Clenshaw(a, X)

represents the operator `a(X)` where a is a polynomial.
Here `a` is to stored as a quasi-vector.
"""
struct Clenshaw{T, Coefs<:AbstractVector, AA<:AbstractVector, BB<:AbstractVector, CC<:AbstractVector, Jac<:AbstractMatrix} <: AbstractBandedMatrix{T}
    c::Coefs
    A::AA
    B::BB
    C::CC
    X::Jac
    p0::T
end

Clenshaw(c::AbstractVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix{T}, p0::T) where T = 
    Clenshaw{T,typeof(c),typeof(A),typeof(B),typeof(C),typeof(X)}(c, A, B, C, X, p0)

Clenshaw(c::Number, A, B, C, X, p) = Clenshaw([c], A, B, C, X, p)

function Clenshaw(a::AbstractQuasiVector, X::AbstractQuasiMatrix)
    P,c = arguments(a)
    Clenshaw(paddeddata(c), recurrencecoefficients(P)..., jacobimatrix(X), _p0(P))
end

copy(M::Clenshaw) = M
size(M::Clenshaw) = size(M.X)
axes(M::Clenshaw) = axes(M.X)
bandwidths(M::Clenshaw) = (length(M.c)-1,length(M.c)-1)

Base.array_summary(io::IO, C::Clenshaw{T}, inds::Tuple{Vararg{OneToInf{Int}}}) where T =
    print(io, Base.dims2string(length.(inds)), " Clenshaw{$T} with $(length(C.c)) degree polynomial")

struct ClenshawLayout <: AbstractLazyBandedLayout end
MemoryLayout(::Type{<:Clenshaw}) = ClenshawLayout()
sublayout(::ClenshawLayout, ::Type{<:NTuple{2,AbstractUnitRange{Int}}}) = ClenshawLayout()
sublayout(::ClenshawLayout, ::Type{<:Tuple{AbstractUnitRange{Int},Union{Slice,AbstractInfUnitRange{Int}}}}) = LazyBandedLayout()
sublayout(::ClenshawLayout, ::Type{<:Tuple{Union{Slice,AbstractInfUnitRange{Int}},AbstractUnitRange{Int}}}) = LazyBandedLayout()
sublayout(::ClenshawLayout, ::Type{<:Tuple{Union{Slice,AbstractInfUnitRange{Int}},Union{Slice,AbstractInfUnitRange{Int}}}}) = LazyBandedLayout()
sub_materialize(::ClenshawLayout, V) = BandedMatrix(V)

function _BandedMatrix(::ClenshawLayout, V::SubArray{<:Any,2})
    M = parent(V)
    kr,jr = parentindices(V)
    b = bandwidth(M,1)
    jkr=max(1,min(first(jr),first(kr))-b÷2):max(last(jr),last(kr))+b÷2
    # relationship between jkr and kr, jr
    kr2,jr2 = kr.-first(jkr).+1,jr.-first(jkr).+1
    lmul!(M.p0, clenshaw(M.c, M.A, M.B, M.C, M.X[jkr, jkr])[kr2,jr2])
end

function getindex(M::Clenshaw{T}, kr::AbstractUnitRange, j::Integer) where T
    b = bandwidth(M,1)
    jkr=max(1,min(j,first(kr))-b÷2):max(j,last(kr))+b÷2
    # relationship between jkr and kr, jr
    kr2,j2 = kr.-first(jkr).+1,j-first(jkr)+1
    f = [Zeros{T}(j2-1); one(T); Zeros{T}(length(jkr)-j2)]
    lmul!(M.p0, clenshaw(M.c, M.A, M.B, M.C, M.X[jkr, jkr], f)[kr2])
end

getindex(M::Clenshaw, k::Int, j::Int) = M[k:k,j][1]

transposelayout(M::ClenshawLayout) = LazyBandedMatrices.LazyBandedLayout()
# TODO: generalise for layout, use Base.PermutedDimsArray
Base.permutedims(M::Clenshaw{<:Number}) = transpose(M)


function materialize!(M::MatMulVecAdd{<:ClenshawLayout,<:PaddedLayout,<:PaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch("Dimensions must match"))
    length(x) == size(A,2) || throw(DimensionMismatch("Dimensions must match"))
    x̃ = paddeddata(x);
    m = length(x̃)
    b = bandwidth(A,1)
    jkr=1:m+b
    p = [x̃; zeros(eltype(x̃),length(jkr)-m)];
    Ax = lmul!(A.p0, clenshaw(A.c, A.A, A.B, A.C, A.X[jkr, jkr], p))
    _fill_lmul!(β,y)
    resizedata!(y, last(jkr))
    v = view(paddeddata(y),jkr)
    BLAS.axpy!(α, Ax, v)
    y
end

# TODO: generalise this to be trait based
function layout_broadcasted(::Tuple{ExpansionLayout{<:AbstractOPLayout},AbstractOPLayout}, ::typeof(*), a, P)
    axes(a,1) == axes(P,1) || throw(DimensionMismatch())
    P * Clenshaw(a, P)
end

# TODO: layout_broadcasted
function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{AbstractAffineQuasiVector,Slice}},Any}}, V::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{AbstractAffineQuasiVector,Any}})
    axes(a,1) == axes(V,1) || throw(DimensionMismatch())
    kr,jr = parentindices(V)
    P = view(parent(V),kr,:)
    P * Clenshaw(a, P)[:,jr]
end


layout_broadcasted(::Tuple{BroadcastLayout{typeof(*)},AbstractOPLayout}, ::typeof(*), a, P) =
    _broadcasted_layout_broadcasted_mul(map(MemoryLayout,arguments(BroadcastLayout{typeof(*)}(),a)),a,P)

function _broadcasted_layout_broadcasted_mul(::Tuple{WeightLayout,PolynomialLayout}, wv, P)
    w,v = arguments(wv)
    Q = OrthogonalPolynomial(w)
    a = (w .* Q) * (Q \ v)
    a .* P
end


##
# Banded dot is slow
###

LinearAlgebra.dot(x::AbstractVector, A::Clenshaw, y::AbstractVector) = dot(x, mul(A, y))