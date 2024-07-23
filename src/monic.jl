struct Monic{T,OPs<:AbstractQuasiMatrix{T},NL} <: OrthogonalPolynomial{T}
    P::Normalized{T,OPs,NL}
    # X::AbstractMatrix{T} # Should X be stored? Probably not. 
    # X would be the Jacobi matrix of the normalised polynomials, not for the monic polynomials.
end
# Will need to figure out what this should be exactly. 
# Consider this internal for now until it stabilises.

Monic(P::AbstractQuasiMatrix) = Monic(Normalized(P))
Monic(P::Monic) = P

Normalized(P::Monic) = P.P

axes(P::Monic) = axes(P.P)

orthogonalityweight(P::Monic) = orthogonalityweight(P.P)

_p0(::Monic{T}) where {T} = one(T)

show(io::IO, P::Monic) = print(io, "Monic($(P.P.P))")
show(io::IO, ::MIME"text/plain", P::Monic) = show(io, P)

function getindex(P::Monic{T}, x::Number, n::Int)::T where {T}
    # TODO: Rewrite this to be more efficient using forwardrecurrence!
    p0 = _p0(P)
    n == 1 && return p0
    t = convert(T, x)
    J = jacobimatrix(P.P, n)
    α = diagonaldata(J)
    β = supdiagonaldata(J)
    p1 = ((t - α[1]) * p0)::T
    n == 2 && return p1
    for i in 2:(n-1)
        _p1 = p0::T
        p0 = p1::T
        p1 = ((t - α[i]) * p1 - β[i-1]^2 * _p1)::T
    end
    return p1
end
# Should a method be written that makes this more efficient when requesting multiple n?


