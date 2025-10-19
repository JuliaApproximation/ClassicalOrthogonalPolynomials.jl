# If f(x) = P[x,1:n+1] * c
# normalized so that c[n+1] == 1 so that 
# P[x,n+1] == -P[x,1:n]*c[1:n]
# we have 
# x*P[x,1:n] == J[1:n+1,1:n] * P[x,1:n+1]
#    == J[1:n,1:n] * P[x,1:n] + [zeros(n-1); J[n+1,n]*P[x,n+1]]
#    == J[1:n,1:n] * P[x,1:n] + [zeros(n-1); -J[n+1,n]*c[1:n]']*P[x,1:n]
# I.e. (J[1:n,1:n] + [zeros(n-1,n); -J[n+1,n]*c[1:n]'])*P[x,1:n] = x*P[x,1:n]
# 

function colleaguematrix(P, c)
    cₙ = paddeddata(c)
    n = findlast(!iszero, cₙ)-1
    J = jacobimatrix(P)'
    C = Matrix(J[1:n,1:n])
    C[end,:] .-= J[n+1,n] .* view(cₙ,1:n) ./ cₙ[n+1]
    C
end


function _findall(::typeof(iszero), ::ExpansionLayout{<:AbstractOPLayout}, f)
    C = colleaguematrix(f.args...)
    ax = axes(f,1)
    convert(Vector{eltype(ax)}, filter!(in(ax), eigvals(C)))
end
findall(f::Function, v::AbstractQuasiVector) = _findall(f, MemoryLayout(v), v)

# gives a generalization of midpoint for when `a` or `b` is infinite
function genmidpoint(a::T, b::T) where T
    if isinf(a) && isinf(b)
        zero(T)
    elseif isinf(a)
        b - 100
    elseif isinf(b)
        a + 100
    else
        (a+b)/2
    end
end


function _searchsortedfirst(::ExpansionLayout{<:AbstractOPLayout}, f, x; iterations=47)
    d = axes(f,1)
    a,b = first(d), last(d)

    for k=1:iterations  #TODO: decide 47
        m= genmidpoint(a,b)
        (f[m] ≤ x) ? (a = m) : (b = m)
    end
    (a+b)/2
end
searchsortedfirst(f::AbstractQuasiVector, x; kwds...) = _searchsortedfirst(MemoryLayout(f), f, x; kwds...)

sample(f::AbstractQuasiArray, n...) = sample_layout(MemoryLayout(f), f, n...)

function sample_layout(_, f::AbstractQuasiVector, n...)
    g = cumsum(f)
    searchsortedfirst.(Ref(g/last(g)), rand(n...))
end

function sample_layout(_, f::AbstractQuasiMatrix, n...)
    @assert size(f,2) == 1 # TODO generalise 
    sample(f[:,1], n...)
end

####
# min/max/extrema
####
function minimum(f::AbstractQuasiVector)
    r = findall(iszero, diff(f))
    min(first(f), minimum(f[r]), last(f))
end

function maximum(f::AbstractQuasiVector)
    r = findall(iszero, diff(f))
    max(first(f), maximum(f[r]), last(f))
end

function extrema(f::AbstractQuasiVector)
    r = findall(iszero, diff(f))
    extrema([first(f); f[r]; last(f)])
end