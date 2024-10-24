
# Assume 1 normalization
_p0(A) = one(eltype(A))


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
        forwardrecurrence!(view(dest,k,:), Ã, B̃, C̃, x, p0, p1)
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
    forwardrecurrence!(dest, Ã, B̃, C̃, x, p0, p1)
    dest
end

getindex(P::OrthogonalPolynomial, x::Number, n::AbstractVector) = layout_getindex(P, x, n)
getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractVector) = layout_getindex(P, x, n)
getindex(P::SubArray{<:Any,1,<:OrthogonalPolynomial}, x::AbstractVector) = layout_getindex(P, x)
Base.@propagate_inbounds function getindex(P::OrthogonalPolynomial, x::Number, n::Number)
    @boundscheck checkbounds(P, x, n)
    Base.unsafe_getindex(P, x, n)
end


unsafe_layout_getindex(A...) = sub_materialize(Base.unsafe_view(A...))

Base.unsafe_getindex(P::OrthogonalPolynomial, x::Number, n::AbstractUnitRange) = unsafe_layout_getindex(P, x, n)
Base.unsafe_getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractUnitRange) = unsafe_layout_getindex(P, x, n)
Base.unsafe_getindex(P::OrthogonalPolynomial, x::Number, n::AbstractVector) = Base.unsafe_getindex(P,x,oneto(maximum(n)))[n]
Base.unsafe_getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractVector) = Base.unsafe_getindex(P,x,oneto(maximum(n)))[:,n]
Base.unsafe_getindex(P::OrthogonalPolynomial, x::AbstractVector, n::Number) = Base.unsafe_getindex(P, x, 1:n)[:,end]
Base.unsafe_getindex(P::OrthogonalPolynomial, x::Number, ::Colon) = Base.unsafe_getindex(P, x, axes(P,2))
Base.unsafe_getindex(P::OrthogonalPolynomial, x::Number, n::Number) = initiateforwardrecurrence(n-1, recurrencecoefficients(P)..., x, _p0(P))[end]

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

function unsafe_getindex(f::Mul{<:AbstractOPLayout,<:AbstractPaddedLayout}, x::Number)
    P,c = f.A,f.B
    _p0(P)*clenshaw(paddeddata(c), recurrencecoefficients(P)..., x)
end

function unsafe_getindex(f::Mul{<:AbstractOPLayout,<:AbstractPaddedLayout}, x::Number, jr)
    P,c = f.A,f.B
    _p0(P)*clenshaw(view(paddeddata(c),:,jr), recurrencecoefficients(P)..., x)
end

Base.@propagate_inbounds function getindex(f::Mul{<:AbstractOPLayout,<:AbstractPaddedLayout}, x::Number, j...)
    @inbounds checkbounds(ApplyQuasiArray(*,f.A,f.B), x, j...)
    unsafe_getindex(f, x, j...)
end

Base.@propagate_inbounds getindex(f::Mul{<:WeightedOPLayout,<:AbstractPaddedLayout}, x::Number, j...) =
    weight(f.A)[x] * (unweighted(f.A) * f.B)[x, j...]


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

function _broadcasted_layout_broadcasted_mul(::Tuple{AbstractWeightLayout,PolynomialLayout}, wv, P)
    w,v = arguments(wv)
    Q = OrthogonalPolynomial(w)
    a = (w .* Q) * (Q \ v)
    a .* P
end

# constructor for Clenshaw
function Clenshaw(a::AbstractQuasiVector, X::AbstractQuasiMatrix)
    P,c = arguments(a)
    Clenshaw(paddeddata(c), recurrencecoefficients(P)..., jacobimatrix(X), _p0(P))
end