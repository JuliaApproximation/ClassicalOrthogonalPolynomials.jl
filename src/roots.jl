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


function _roots(::ExpansionLayout{<:AbstractOPLayout}, f)
    C = colleaguematrix(f.args...)
    ax = axes(f,1)
    convert(Vector{eltype(ax)}, filter!(in(ax), eigvals(C)))
end
findall(::typeof(iszero), f::AbstractQuasiVector) = _roots(MemoryLayout(f), f)