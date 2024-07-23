struct Monic{T,OPs<:AbstractQuasiMatrix{T},NL} <: OrthogonalPolynomial{T}
    P::Normalized{T,OPs,NL}
    α::AbstractVector{T} # diagonal of Jacobi matrix of P
    β::AbstractVector{T} # squared supdiagonal of Jacobi matrix of P
end

Monic(P::AbstractQuasiMatrix) = Monic(Normalized(P))
function Monic(P::Normalized)
    X = jacobimatrix(P)
    α = diagonaldata(X)
    β = supdiagonaldata(X)
    return Monic(P, α, β.^2)
end
Monic(P::Monic) = Monic(P.P, P.α, P.β)

Normalized(P::Monic) = P.P

axes(P::Monic) = axes(P.P)

orthogonalityweight(P::Monic) = orthogonalityweight(P.P)

_p0(::Monic{T}) where {T} = one(T)

show(io::IO, P::Monic) = print(io, "Monic($(P.P.P))")
show(io::IO, ::MIME"text/plain", P::Monic) = show(io, P)

function recurrencecoefficients(P::Monic{T}) where {T}
    α = P.α
    β = P.β
    return _monicrecurrencecoefficients(α, β) # function barrier 
end
function _monicrecurrencecoefficients(α::AbstractVector{T}, β) where {T}
    A = Ones{T}(∞)
    B = -α
    C = Vcat(zero(T), β)
    return A, B, C
end

