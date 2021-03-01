"""
   HermiteWeight()

is a quasi-vector representing `exp(-x^2)` on ℝ.
"""
struct HermiteWeight{T} <: Weight{T} end

HermiteWeight() = HermiteWeight{Float64}()
axes(::HermiteWeight{T}) where T = (Inclusion(ℝ),)
function getindex(w::HermiteWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    exp(-x^2)
end

sum(::HermiteWeight{T}) where T = sqrt(convert(T, π))

struct Hermite{T} <: OrthogonalPolynomial{T} end
Hermite() = Hermite{Float64}()
orthogonalityweight(::Hermite{T}) where T = HermiteWeight{T}()

==(::Hermite, ::Hermite) = true
axes(::Hermite{T}) where T = (Inclusion(ℝ), oneto(∞))

"""
     hermiteh(n, z)

computes the `n`-th Hermite polynomial, orthogonal with 
respec to `exp(-x^2)`, at `z`.
"""
hermiteh(n::Integer, z::Number) = Base.unsafe_getindex(Hermite{typeof(z)}(), z, n+1)

# H_{n+1} = 2x H_n - 2n H_{n-1}
# 1/2 * H_{n+1} + n H_{n-1} = x H_n 
# x*[H_0 H_1 H_2 …] = [H_0 H_1 H_2 …] * [0    1; 1/2  0     2; 1/2   0  3; …]   
jacobimatrix(H::Hermite{T}) where T = Tridiagonal(Fill(one(T)/2,∞), Zeros{T}(∞), one(T):∞)
recurrencecoefficients(H::Hermite{T}) where T = Fill{T}(2,∞), Zeros{T}(∞), zero(T):2:∞

@simplify function *(Ac::QuasiAdjoint{<:Any,<:Hermite}, B::WeightedBasis{<:Any,<:HermiteWeight,<:Hermite})  
    T = promote_type(eltype(Ac), eltype(B))
    Diagonal(sqrt(convert(T,π)) .* convert(T,2) .^ (0:∞) .* factorial.(convert(T,0):∞))
end

##########
# Derivatives
##########

@simplify function *(D::Derivative, H::Hermite)
    T = promote_type(eltype(D),eltype(H))
    D = _BandedMatrix((zero(T):2:∞)', ℵ₀, -1,1)
    H*D
end

@simplify function *(D::Derivative, Q::OrthonormalWeighted{<:Any,<:Hermite})
    X = jacobimatrix(Q.P)
    Q * Tridiagonal(-X.ev, X.dv, X.ev)
end