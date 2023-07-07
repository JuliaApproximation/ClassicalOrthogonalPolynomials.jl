"""
   LaguerreWeight(α)

is a quasi-vector representing `x^α * exp(-x)` on `0..Inf`.
"""
struct LaguerreWeight{T} <: Weight{T}
    α::T
end

LaguerreWeight{T}() where T = LaguerreWeight{T}(zero(T))
LaguerreWeight() = LaguerreWeight{Float64}()
# axes(::LaguerreWeight{T}) where T = (Inclusion(ℝ),)
axes(::LaguerreWeight{T}) where T = (Inclusion(HalfLine{T}()),)
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
orthogonalityweight(L::Laguerre)= LaguerreWeight(L.α)

==(L1::Laguerre, L2::Laguerre) = L1.α == L2.α
axes(::Laguerre{T}) where T = (Inclusion(HalfLine{T}()), oneto(∞))

"""
     laguerrel(n, α, z)

computes the `n`-th generalized Laguerre polynomial, orthogonal with 
respec to `x^α * exp(-x)`, at `z`.
"""
laguerrel(n::Integer, α, z::Number) = Base.unsafe_getindex(Laguerre{promote_type(typeof(α), typeof(z))}(α), z, n+1)

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

@simplify function *(D::Derivative, L::Laguerre)
    T = promote_type(eltype(D),eltype(L))
    D = _BandedMatrix(Fill(-one(T),1,∞), ∞, -1,1)
    Laguerre(L.α+1)*D
end

@simplify function *(D::Derivative, w_A::Weighted{<:Any,<:Laguerre})
    T = promote_type(eltype(D),eltype(w_A))
    D = BandedMatrix(-1=>one(T):∞)
    Weighted(Laguerre{T}(w_A.P.α-1))*D
end

##########
# Conversion
##########

function \(L::Laguerre, K::Laguerre)
    T = promote_type(eltype(L), eltype(K))
    if L.α ≈ K.α
        Eye{T}(∞)
    elseif L.α ≈ K.α + 1
        BandedMatrix(0=>Fill{T}(one(T), ∞), 1=>Fill{T}(-one(T), ∞))
    else
        error("Not implemented for this choice of L.α and K.α.")
    end
end

function \(w_A::Weighted{<:Any,<:Laguerre}, w_B::Weighted{<:Any,<:Laguerre})
    T = promote_type(eltype(w_A), eltype(w_B))
    if w_A.P.α ≈ w_B.P.α
        Eye{T}(∞)
    elseif w_A.P.α + 1 ≈ w_B.P.α
        BandedMatrix(0=>w_B.P.α:∞, -1=>-one(T):-one(T):-∞)
    else
        error("Not implemented for this choice of w_A.P.α and w_B.P.α.")
    end
end