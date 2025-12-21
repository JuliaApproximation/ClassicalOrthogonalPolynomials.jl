"""
   LaguerreWeight(α)

is a quasi-vector representing `x^α * exp(-x)` on `0..Inf`.
"""
struct LaguerreWeight{T} <: Weight{T}
    α::T
end

LaguerreWeight{T}() where T = LaguerreWeight{T}(zero(T))
LaguerreWeight() = LaguerreWeight{Float64}()

AbstractQuasiArray{T}(w::LaguerreWeight) where T = LaguerreWeight{T}(w.α)
AbstractQuasiVector{T}(w::LaguerreWeight) where T = LaguerreWeight{T}(w.α)


axes(::LaguerreWeight{T}) where T = (Inclusion(ℝ),)
function getindex(w::LaguerreWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    x^w.α * exp(-x)
end

sum(L::LaguerreWeight{T}) where T = gamma(L.α + 1)

struct Laguerre{T} <: OrthogonalPolynomial{T} 
    α::T
    Laguerre{T}(α) where T = new{T}(convert(T, α))
end
Laguerre{T}() where T = Laguerre{T}(zero(T))
Laguerre() = Laguerre{Float64}()
Laguerre(α::T) where T = Laguerre{float(T)}(α)

AbstractQuasiArray{T}(w::Laguerre) where T = Laguerre{T}(w.α)
AbstractQuasiMatrix{T}(w::Laguerre) where T = Laguerre{T}(w.α)


orthogonalityweight(L::Laguerre)= LaguerreWeight(L.α)

==(L1::Laguerre, L2::Laguerre) = L1.α == L2.α
axes(::Laguerre{T}) where T = (Inclusion(HalfLine{T}()), oneto(∞))

"""
     laguerrel(n, α, z)

computes the `n`-th generalized Laguerre polynomial, orthogonal with 
respec to `x^α * exp(-x)`, at `z`.
"""
laguerrel(n::Integer, α, z::Number) = Base.unsafe_getindex(Laguerre{polynomialtype(typeof(α), typeof(z))}(α), z, n+1)

"""
     laguerrel(n, z)

computes the `n`-th Laguerre polynomial, orthogonal with 
respec to `exp(-x)`, at `z`.
"""
laguerrel(n::Integer, z::Number) = laguerrel(n, 0, z)


# L_{n+1} = (-1/(n+1) x + (2n+α+1)/(n+1)) L_n - (n+α)/(n+1) L_{n-1}
# - (n+α) L_{n-1} + (2n+α+1)* L_n -(n+1) L_{n+1} = x  L_n
# x*[L_0 L_1 L_2 …] = [L_0 L_1 L_2 …] * [(α+1)    -(α+1); -1  (α+3)     -(α+2);0  -2   (α+5) -(α+3); …]   
function jacobimatrix(L::Laguerre{T}) where T
    α = L.α
    Tridiagonal(-(1:∞), (α+1):2:∞, -(α+1:∞))
end

recurrencecoefficients(L::Laguerre{T}) where T = ((-one(T)) ./ (1:∞), ((L.α+1):2:∞) ./ (1:∞), (L.α:∞) ./ (1:∞))

##########
# Derivatives
##########

function diff(L::Laguerre{T}; dims=1) where T
    D = _BandedMatrix(Fill(-one(T),1,∞), ∞, -1,1)
    ApplyQuasiMatrix(*, Laguerre(L.α+1), D)
end



##########
# grammatrix
##########

function weightedgrammatrix(L::Laguerre{T}) where T
    α = L.α
    iszero(α) && return Eye{T}(∞)
    isone(α) && return Diagonal(convert(T,1):∞)
    Diagonal(exp.(loggamma.((1:∞) .+ α) .- loggamma.(1:∞)))
end