
# Helper routines

function reverseeven!(x::AbstractVector)
    n = length(x)
    if iseven(n)
        @inbounds @simd for k=2:2:n÷2
            x[k],x[n+2-k] = x[n+2-k],x[k]
        end
    else
        @inbounds @simd for k=2:2:n÷2
            x[k],x[n+1-k] = x[n+1-k],x[k]
        end
    end
    x
end

function negateeven!(x::AbstractVector)
    @inbounds @simd for k = 2:2:length(x)
        x[k] *= -1
    end
    x
end



### In-place O(n) interlacing

function highestleader(n::Int)
    i = 1
    while 3i < n i *= 3 end
    i
end

function nextindex(i::Int,n::Int)
    i <<= 1
    while i > n
        i -= n + 1
    end
    i
end

function cycle_rotate!(v::AbstractVector, leader::Int, it::Int, twom::Int)
    i = nextindex(leader, twom)
    while i != leader
        idx1, idx2 = it + i - 1, it + leader - 1
        @inbounds v[idx1], v[idx2] = v[idx2], v[idx1]
        i = nextindex(i, twom)
    end
    v
end

function right_cyclic_shift!(v::AbstractVector, it::Int, m::Int, n::Int)
    itpm = it + m
    itpmm1 = itpm - 1
    itpmpnm1 = itpmm1 + n
    reverse!(v, itpm, itpmpnm1)
    reverse!(v, itpm, itpmm1 + m)
    reverse!(v, itpm + m, itpmpnm1)
    v
end

"""
This function implements the algorithm described in:

    P. Jain, "A simple in-place algorithm for in-shuffle," arXiv:0805.1598, 2008.
"""
function interlace!(v::AbstractVector,offset::Int)
    N = length(v)
    if N < 2 + offset
        return v
    end

    it = 1 + offset
    m = 0
    n = 1

    while m < n
        twom = N + 1 - it
        h = highestleader(twom)
        m = h > 1 ? h÷2 : 1
        n = twom÷2

        right_cyclic_shift!(v,it,m,n)

        leader = 1
        while leader < 2m
            cycle_rotate!(v, leader, it, 2m)
            leader *= 3
        end

        it += 2m
    end
    v
end


abstract type AbstractInterlaceBasis{T} <: Basis{T} end
copy(A::AbstractInterlaceBasis) = interlacebasis(A, map(copy, A.args))

"""
    PiecewiseInterlace(args...)

is an analogue of `Basis` that takes the union of the first axis,
and the second axis is a blocked interlace of args.
If there is overlap, it uses the first in order.
"""
struct PiecewiseInterlace{T, Args} <: AbstractInterlaceBasis{T}
    args::Args
end

PiecewiseInterlace{T}(args...) where T = PiecewiseInterlace{T,typeof(args)}(args)
PiecewiseInterlace(args...) = PiecewiseInterlace{mapreduce(eltype,promote_type,args)}(args...)
PiecewiseInterlace{T}(args::AbstractVector) where T = PiecewiseInterlace{T,typeof(args)}(args)
PiecewiseInterlace(args::AbstractVector) = PiecewiseInterlace{eltype(eltype(args))}(args)

interlacebasis(::PiecewiseInterlace, args...) = PiecewiseInterlace(args...)


axes(A::PiecewiseInterlace) = (union(axes.(A.args,1)...), LazyBandedMatrices._block_vcat_axes(unitblocks.(axes.(A.args,2))...))

==(A::PiecewiseInterlace, B::PiecewiseInterlace) = all(A.args .== B.args)


function QuasiArrays._getindex(::Type{IND}, A::PiecewiseInterlace{T}, (x,j)::IND) where {IND,T}
    Jj = findblockindex(axes(A,2), j)
    @boundscheck x in axes(A,1) || throw(BoundsError(A, (x,j)))
    J = Int(block(Jj))
    i = blockindex(Jj)
    x in axes(A.args[i],1) && return A.args[i][x, J]
    zero(T)
end

function \(A::AbstractInterlaceBasis, B::AbstractInterlaceBasis)
    axes(A,1) == axes(B,1) || throw(DimensionMismatch())
    T = promote_type(eltype(A),eltype(B))
    A == B && return Eye{T}((axes(A,2),))
    BlockBroadcastArray{T}(Diagonal, unitblocks.((\).(A.args, B.args))...)
end

@simplify function *(D::Derivative, S::AbstractInterlaceBasis)
    axes(D,2) == axes(S,1) || throw(DimensionMismatch())
    args = arguments.(*, Derivative.(axes.(S.args,1)) .* S.args)
    all(length.(args) .== 2) || error("Not implemented")
    interlacebasis(S, map(first, args)...) * BlockBroadcastArray{promote_type(eltype(D),eltype(S))}(Diagonal, unitblocks.(last.(args))...)
end

struct PiecewiseFactorization{T,FF,Ax} <: Factorization{T}
    factorizations::FF
    axes::Ax
end

PiecewiseFactorization{T}(fac, ax) where T = PiecewiseFactorization{T,typeof(fac),typeof(ax)}(fac, ax)

function \(F::PiecewiseFactorization{T}, v::AbstractQuasiVector) where {T}
    BlockBroadcastArray{T}(vcat, unitblocks.((\).(F.factorizations, getindex.(Ref(v), F.axes)))...)
end

function factorize(V::SubQuasiArray{T,2,<:AbstractInterlaceBasis,<:Tuple{Inclusion,BlockSlice{BlockRange1{OneTo{Int}}}}}) where T
    P = parent(V)
    _,jr = parentindices(V)
    N = Int(last(jr.block))
    PiecewiseFactorization{T}(factorize.(view.(P.args, :, Ref(Base.OneTo(N)))), axes.(P.args,1))
end

function factorize(V::SubQuasiArray{<:Any,2,<:AbstractInterlaceBasis,<:Tuple{Inclusion,AbstractVector{Int}}})
    P = parent(V)
    _,jr = parentindices(V)
    J = findblock(axes(P,2),maximum(jr))
    ProjectionFactorization(factorize(P[:,Block.(OneTo(Int(J)))]), jr)
end