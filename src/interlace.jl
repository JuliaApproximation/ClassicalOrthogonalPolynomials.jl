
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
