


##
# Ultraspherical
##

struct UltrasphericalWeight{T,Λ} <: AbstractJacobiWeight{T}
    λ::Λ
end

UltrasphericalWeight{T}(λ) where T = UltrasphericalWeight{T,typeof(λ)}(λ)
UltrasphericalWeight(λ) = UltrasphericalWeight{float(typeof(λ)),typeof(λ)}(λ)
UltrasphericalWeight(::LegendreWeight{T}) where T = UltrasphericalWeight(one(T)/2)

AbstractQuasiArray{T}(w::UltrasphericalWeight) where T = UltrasphericalWeight{T}(w.λ)
AbstractQuasiVector{T}(w::UltrasphericalWeight) where T = UltrasphericalWeight{T}(w.λ)

show(io::IO, w::UltrasphericalWeight) = summary(io, w)
summary(io::IO, w::UltrasphericalWeight) = print(io, "UltrasphericalWeight($(w.λ))")

==(a::UltrasphericalWeight, b::UltrasphericalWeight) = a.λ == b.λ

function getindex(w::UltrasphericalWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x^2)^(w.λ-one(w.λ)/2)
end

sum(w::UltrasphericalWeight{T}) where T = sqrt(convert(T,π))*exp(loggamma(one(T)/2 + w.λ)-loggamma(1+w.λ))

hasboundedendpoints(w::UltrasphericalWeight) = 2w.λ ≥ 1


struct Ultraspherical{T,Λ} <: AbstractJacobi{T}
    λ::Λ
end
Ultraspherical{T}(λ::Λ) where {T,Λ} = Ultraspherical{T,Λ}(λ)
Ultraspherical(λ::Λ) where Λ = Ultraspherical{Float64,Λ}(λ)
Ultraspherical(P::Legendre{T}) where T = Ultraspherical(one(T)/2)
function Ultraspherical(P::Jacobi{T}) where T
    P.a == P.b || throw(ArgumentError("$P is not ultraspherical"))
    Ultraspherical(P.a+one(T)/2)
end

Ultraspherical(::ChebyshevU{T}) where T = Ultraspherical{T}(1)

AbstractQuasiArray{T}(w::Ultraspherical) where T = Ultraspherical{T}(w.λ)
AbstractQuasiMatrix{T}(w::Ultraspherical) where T = Ultraspherical{T}(w.λ)

show(io::IO, w::Ultraspherical) = summary(io, w)
summary(io::IO, w::Ultraspherical) = print(io, "Ultraspherical($(w.λ))")

const WeightedUltraspherical{T} = WeightedBasis{T,<:UltrasphericalWeight,<:Ultraspherical}

orthogonalityweight(C::Ultraspherical) = UltrasphericalWeight(C.λ)

ultrasphericalc(n::Integer, λ, z) = Base.unsafe_getindex(Ultraspherical{polynomialtype(typeof(λ),typeof(z))}(λ), z, n+1)
ultraspherical(λ, d::AbstractInterval{T}) where T = Ultraspherical{float(promote_type(eltype(λ),T))}(λ)[affine(d,ChebyshevInterval{T}()), :]
ultraspherical(λ, d::ChebyshevInterval{T}) where T = Ultraspherical{float(promote_type(eltype(λ),T))}(λ)

==(a::Ultraspherical, b::Ultraspherical) = a.λ == b.λ
==(::Ultraspherical, ::ChebyshevT) = false
==(::ChebyshevT, ::Ultraspherical) = false
==(C::Ultraspherical, ::ChebyshevU) = isone(C.λ)
==(::ChebyshevU, C::Ultraspherical) = isone(C.λ)
==(P::Ultraspherical, Q::Jacobi) = isone(2P.λ) && Jacobi(P) == Q
==(P::Jacobi, Q::Ultraspherical) = isone(2Q.λ) && P == Jacobi(Q)
==(P::Ultraspherical, Q::Legendre) = isone(2P.λ)
==(P::Legendre, Q::Ultraspherical) = isone(2Q.λ)



###
# transforms
###

plan_transform(P::Ultraspherical{T}, szs::NTuple{N,Int}, dims...) where {T,N} = JacobiTransformPlan(FastTransforms.plan_th_ultra2ultra!(T, szs, one(P.λ), P.λ, dims...), plan_chebyshevutransform(T, szs, dims...))

###
# interrelationships
###

Jacobi(C::Ultraspherical{T}) where T = Jacobi(C.λ-one(T)/2,C.λ-one(T)/2)



######
# Weighted Gram Matrix
######

# 2^(1-2λ)*π*gamma(n+2λ)/((n+λ)*gamma(λ)^2 * n!)
function weightedgrammatrix(P::Ultraspherical{T}) where T
    λ = P.λ
    n = 0:∞
    c = 2^(1-2λ) * convert(T,π)/gamma(λ)^2
    Diagonal(c * exp.(loggamma.(n .+ 2λ) .- loggamma.(n .+ 1) ) ./ (n .+ λ))
end

########
# Jacobi Matrix
########

function jacobimatrix(P::Ultraspherical{T}) where T
    λ = P.λ
    Tridiagonal((one(T):∞) ./ (2 .*((zero(T):∞) .+ λ)),
                Zeros{T}(∞),
                ((2λ):∞) ./ (2 .*((one(T):∞) .+ λ)))
end

# These return vectors A[k], B[k], C[k] are from DLMF. Cause of MikaelSlevinsky we need an extra entry in C ... for now.
function recurrencecoefficients(C::Ultraspherical{T}) where T
    λ = convert(T,C.λ)
    n = 0:∞
    (2(n .+ λ) ./ (n .+ 1), Zeros{typeof(λ)}(∞), (n .+ (2λ-1)) ./ (n .+ 1))
end


##########
# Derivatives
##########

# Ultraspherical(1)\(D*Chebyshev())
diff(S::ChebyshevU, m...; dims=1) = diff(Ultraspherical(S), m...; dims)
diff(S::Legendre, m...; dims=1) = diff(Ultraspherical(S), m...; dims)


# Ultraspherical(1/2)\(D*Legendre())
# Special cased as its a Ones
function diff(S::Legendre{T}; dims=1) where T
    A = _BandedMatrix(Ones{T}(1,∞), ℵ₀, -1,1)
    ApplyQuasiMatrix(*, Ultraspherical{T}(convert(T,3)/2), A)
end


# Ultraspherical(λ+1)\(D*Ultraspherical(λ))
function diff(S::Ultraspherical{T}; dims=1) where T
    A = _BandedMatrix(Fill(2convert(T,S.λ),1,∞), ℵ₀, -1,1)
    ApplyQuasiMatrix(*, Ultraspherical{T}(S.λ+1), A)
end

# higher order 

function diff(S::ChebyshevT{T}, m::Integer; dims=1) where T
    iszero(m) && return S
    isone(m) && return diff(S)
    μ = pochhammer(one(T),m-1)*convert(T,2)^(m-1)
    D = _BandedMatrix((μ * (0:∞))', ℵ₀, -m, m)
    ApplyQuasiMatrix(*, Ultraspherical{T}(m), D)
end

function diff(C::Ultraspherical{T}, m::Integer; dims=1) where T
    μ = pochhammer(convert(T,C.λ),m)*convert(T,2)^m
    D = _BandedMatrix(Fill(μ,1,∞), ℵ₀, -m, m)
    ApplyQuasiMatrix(*, Ultraspherical{T}(C.λ+m), D)
end

# Ultraspherical(λ-1)\ (D*wUltraspherical(λ))
function diff(WS::Weighted{T,<:Ultraspherical}; dims=1) where T
    S = WS.P
    λ = S.λ
    if λ == 1
        A = _BandedMatrix((-(one(T):∞))', ℵ₀, 1,-1)
        ApplyQuasiMatrix(*, Weighted(ChebyshevT{T}()), A)
    else
        n = (0:∞)
        A = _BandedMatrix((-one(T)/(2*(λ-1)) * ((n.+1) .* (n .+ (2λ-1))))', ℵ₀, 1,-1)
        if λ == 3/2
            ApplyQuasiMatrix(*, Legendre{T}(), A)
        else
            ApplyQuasiMatrix(*, Weighted(Ultraspherical{T}(λ-1)), A)
        end
    end
end

# Ultraspherical(λ-1)\ (D*w*Ultraspherical(λ))
function diff(WS::WeightedUltraspherical{T}; dims=1) where T
    w,S = WS.args
    λ = S.λ
    if iszero(w.λ)
        diff(S)
    elseif isorthogonalityweighted(WS) # weights match
        diff(Weighted(S))
    else
        error("Not implemented")
    end
end

function _cumsum(P::Legendre{V}, dims) where V
    @assert dims == 1
    Σ = Bidiagonal(Vcat(1, Zeros{V}(∞)), Fill(-one(V), ∞), :L)
    ApplyQuasiArray(*, Ultraspherical(-one(V)/2), Σ)
end


##########
# Conversion
##########

\(A::Ultraspherical, B::Legendre) = A\Ultraspherical(B)
\(A::Legendre, B::Ultraspherical) = Ultraspherical(A)\B
\(A::Legendre, B::Weighted{<:Any,<:Ultraspherical}) = Weighted(Ultraspherical(A))\B

function \(A::Ultraspherical, B::Jacobi)
    Ã = Jacobi(A)
    Diagonal(Ã[1,:]./A[1,:]) * (Ã\B)
end
function \(A::Jacobi, B::Ultraspherical)
    if B == Ultraspherical(-1/2) && (A == Jacobi(-1, 0) || A == Jacobi(0, -1))
        # In this case, Jacobi(-1, -1) is (currently) undefined, so the conversion via B̃ = Jacobi(B) leads to NaNs 
        # from evaluating in B̃[1, :]
        T = promote_type(eltype(A), eltype(B))
        n = -2one(T) ./ (2 .* (2:∞) .- one(T))
        sgn = A == Jacobi(-1, 0) ? one(T) : -one(T)
        dv = Vcat(one(T), -2one(T), n)
        ev = Vcat(-sgn, sgn .* n)
        LazyBandedMatrices.Bidiagonal(dv, ev, :U)
    else
        B̃ = Jacobi(B)
        (A\B̃)*Diagonal(B[1,:]./B̃[1,:])
    end 
end

function \(wA::Weighted{<:Any,<:Ultraspherical}, wB::Weighted{<:Any,<:Jacobi})
    A = wA.P
    Ã = Jacobi(A)
    Diagonal(Ã[1,:]./A[1,:]) * (Weighted(Ã)\wB)
end

function \(wA::Weighted{<:Any,<:Jacobi}, wB::Weighted{<:Any,<:Ultraspherical})
    B = wB.P
    B̃ = Jacobi(B)
    (wA\Weighted(B̃))*Diagonal(B[1,:]./B̃[1,:])
end



function \(U::Ultraspherical{<:Any,<:Integer}, C::ChebyshevT)
    T = promote_type(eltype(U), eltype(C))
    (U\Ultraspherical{T}(1)) * (ChebyshevU{T}()\C)
end

function \(U::Ultraspherical{<:Any,<:Integer}, C::ChebyshevU)
    T = promote_type(eltype(U), eltype(C))
    U\Ultraspherical(C)
end



\(T::Chebyshev, C::Ultraspherical) = inv(C \ T)

function \(C2::Ultraspherical{<:Any,<:Integer}, C1::Ultraspherical{<:Any,<:Integer})
    T = promote_type(eltype(C2), eltype(C1))
    λ_Int = C1.λ
    λ = convert(T,λ_Int)
    if C2.λ == λ_Int+1
        _BandedMatrix( Vcat(-(λ ./ ((0:∞) .+ λ))', Zeros{T}(1,∞), (λ ./ ((0:∞) .+ λ))'), ℵ₀, 0, 2)
    elseif C2.λ == λ_Int
        Eye{T}(∞)
    elseif C2.λ > λ_Int
        (C2 \ Ultraspherical(λ_Int+1)) * (Ultraspherical(λ_Int+1)\C1)
    else
        error("Not implemented")
    end
end

function \(C2::Ultraspherical, C1::Ultraspherical)
    T = promote_type(eltype(C2), eltype(C1))
    λ_Int = C1.λ
    λ = convert(T,λ_Int)
    if C2.λ == λ+1
        _BandedMatrix( Vcat(-(λ ./ ((0:∞) .+ λ))', Zeros{T}(1,∞), (λ ./ ((0:∞) .+ λ))'), ℵ₀, 0, 2)
    elseif C2.λ == λ_Int
        Eye{T}(∞)
    elseif isinteger(C2.λ-λ_Int) && C2.λ > λ_Int
        Cm = Ultraspherical{T}(λ_Int+1)
        (C2 \ Cm) * (Cm \ C1)
    elseif isinteger(C2.λ-λ_Int)
        inv(C1 \ C2)
    else
        error("Not implemented")
    end
end

function \(w_A::Weighted{<:Any,<:Ultraspherical}, w_B::Weighted{<:Any,<:Ultraspherical})
    A = w_A.P
    B = w_B.P
    T = promote_type(eltype(w_A),eltype(w_B))

    if A == B
        SquareEye{T}(ℵ₀)
    elseif B.λ == A.λ+1
        λ = convert(T,A.λ)
        _BandedMatrix(Vcat(((2λ:∞) .* ((2λ+1):∞) ./ (4λ .* (λ+1:∞)))',
                            Zeros{T}(1,∞),
                            (-(1:∞) .* (2:∞) ./ (4λ .* (λ+1:∞)))'), ℵ₀, 2,0)
    elseif B.λ > A.λ+1
        J = Weighted(Ultraspherical(B.λ-1))
        (w_A\J) * (J\w_B)
    else
        error("not implemented for $w_A and $w_B")
    end
end



function \(w_A::WeightedUltraspherical, w_B::WeightedUltraspherical)
    wA,A = w_A.args
    wB,B = w_B.args
    T = promote_type(eltype(w_A),eltype(w_B))

    if wA == wB
        A \ B
    elseif wA.λ == A.λ && wB.λ == B.λ # weighted
        Weighted(A) \ Weighted(B)
    elseif wB.λ ≥ wA.λ+1 # lower
        J = UltrasphericalWeight(wB.λ-1) .* Ultraspherical(B.λ-1)
        (w_A\J) * (J\w_B)
    else
        error("not implemented for $w_A and $w_B")
    end
end

\(w_A::WeightedUltraspherical, w_B::Weighted{<:Any,<:Ultraspherical}) = w_A \ convert(WeightedBasis,w_B)
\(w_A::Weighted{<:Any,<:Ultraspherical}, w_B::WeightedUltraspherical) = convert(WeightedBasis,w_A) \ w_B
\(A::Ultraspherical, w_B::Weighted{<:Any,<:Ultraspherical}) = A \ convert(WeightedBasis,w_B)
\(A::Ultraspherical, w_B::WeightedUltraspherical) = (UltrasphericalWeight(one(A.λ)/2) .* A) \ w_B
\(A::Legendre, wB::WeightedUltraspherical) = Ultraspherical(A) \ wB

