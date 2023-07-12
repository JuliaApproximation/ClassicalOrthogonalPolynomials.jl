"""
   HermiteWeight()

is a quasi-vector representing `exp(-x^2)` on ℝ.
"""
struct HermiteWeight{T} <: Weight{T} end

HermiteWeight() = HermiteWeight{Float64}()
axes(::HermiteWeight{T}) where T = (Inclusion{T}(ℝ),)
==(::HermiteWeight, ::HermiteWeight) = true

function getindex(w::HermiteWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    exp(-x^2)
end

sum(::HermiteWeight{T}) where T = sqrt(convert(T, π))

function broadcasted(::typeof(^), H::HermiteWeight, k::Number)
    x = axes(H,1)
    exp.((-k) .* x .^2)
end

broadcasted(::typeof(sqrt), H::HermiteWeight{T}) where T = H .^ (one(T)/2)

struct Hermite{T} <: OrthogonalPolynomial{T} end
Hermite() = Hermite{Float64}()
orthogonalityweight(::Hermite{T}) where T = HermiteWeight{T}()

==(::Hermite, ::Hermite) = true
axes(::Hermite{T}) where T = (Inclusion{T}(ℝ), oneto(∞))

"""
     hermiteh(n, z)

computes the `n`-th Hermite polynomial, orthogonal with 
respec to `exp(-x^2)`, at `z`.
"""
hermiteh(n::Integer, z::Number) = Base.unsafe_getindex(Hermite{typeof(z)}(), z, n+1)

broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), ::HermiteWeight{T}, ::Hermite{V}) where {T,V} = Weighted(Hermite{promote_type(T,V)}())

# H_{n+1} = 2x H_n - 2n H_{n-1}
# 1/2 * H_{n+1} + n H_{n-1} = x H_n 
# x*[H_0 H_1 H_2 …] = [H_0 H_1 H_2 …] * [0    1; 1/2  0     2; 1/2   0  3; …]   
jacobimatrix(H::Hermite{T}) where T = Tridiagonal(Fill(one(T)/2,∞), Zeros{T}(∞), one(T):∞)
recurrencecoefficients(H::Hermite{T}) where T = Fill{T}(2,∞), Zeros{T}(∞), zero(T):2:∞

weightedgrammatrix(::Hermite{T}) where T = Diagonal(sqrt(convert(T,π)) .* convert(T,2) .^ (0:∞) .* gamma.(one(T):∞))

@simplify *(A::QuasiAdjoint{<:Any,<:Weighted{<:Any,<:Hermite}}, B::Hermite) = weightedgrammatrix(Hermite{promote_type(eltype(A),eltype(B))}())

##########
# Derivatives
#####

diff(H::Hermite{T}; dims=1) where T = ApplyQuasiMatrix(*, H, _BandedMatrix((zero(T):2:∞)', ℵ₀, -1,1))
diff(W::Weighted{T,<:Hermite}; dims=1) where T = ApplyQuasiMatrix(*, W, _BandedMatrix(Fill(-one(T), 1, ∞), ℵ₀, 1,-1))
function diff(Q::OrthonormalWeighted{<:Any,<:Hermite}; dims=1)
    X = jacobimatrix(Q.P)
    ApplyQuasiMatrix(*, Q, Tridiagonal(-X.ev, X.dv, X.ev))
end