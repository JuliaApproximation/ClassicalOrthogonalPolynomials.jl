


##
# Ultraspherical
##

struct UltrasphericalWeight{T,Λ} <: AbstractJacobiWeight{T}
    λ::Λ
end

UltrasphericalWeight{T}(λ) where T = UltrasphericalWeight{T,typeof(λ)}(λ)
UltrasphericalWeight(λ) = UltrasphericalWeight{float(typeof(λ)),typeof(λ)}(λ)

==(a::UltrasphericalWeight, b::UltrasphericalWeight) = a.λ == b.λ

function getindex(w::UltrasphericalWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x^2)^(w.λ-one(w.λ)/2)
end

sum(w::UltrasphericalWeight{T}) where T = sqrt(convert(T,π))*exp(loggamma(one(T)/2 + w.λ)-loggamma(1+w.λ))


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

const WeightedUltraspherical{T} = WeightedBasis{T,<:UltrasphericalWeight,<:Ultraspherical}

orthogonalityweight(C::Ultraspherical) = UltrasphericalWeight(C.λ)

ultrasphericalc(n::Integer, λ, z::Number) = Base.unsafe_getindex(Ultraspherical{promote_type(typeof(λ),typeof(z))}(λ), z, n+1)
ultraspherical(λ, d::AbstractInterval{T}) where T = Ultraspherical{float(promote_type(eltype(λ),T))}(λ)[affine(d,ChebyshevInterval{T}()), :]

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
# interrelationships
###

Jacobi(C::Ultraspherical{T}) where T = Jacobi(C.λ-one(T)/2,C.λ-one(T)/2)


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
@simplify *(D::Derivative{<:Any,<:ChebyshevInterval}, S::ChebyshevU) = D * Ultraspherical(S)

# Ultraspherical(1/2)\(D*Legendre())
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::Legendre)
    T = promote_type(eltype(D),eltype(S))
    A = _BandedMatrix(Ones{T}(1,∞), ℵ₀, -1,1)
    ApplyQuasiMatrix(*, Ultraspherical{T}(convert(T,3)/2), A)
end


# Ultraspherical(λ+1)\(D*Ultraspherical(λ))
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::Ultraspherical)
    T = promote_type(eltype(D),eltype(S))
    A = _BandedMatrix(Fill(2convert(T,S.λ),1,∞), ℵ₀, -1,1)
    ApplyQuasiMatrix(*, Ultraspherical{T}(S.λ+1), A)
end

# Ultraspherical(λ-1)\ (D*wUltraspherical(λ))
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::Weighted{<:Any,<:Ultraspherical})
    S = WS.P
    λ = S.λ
    T = promote_type(eltype(D),eltype(WS))
    if λ == 1
        A = _BandedMatrix((-(one(T):∞))', ℵ₀, 1,-1)
        ApplyQuasiMatrix(*, Weighted(ChebyshevT{T}()), A)
    else
        n = (0:∞)
        A = _BandedMatrix((-one(T)/(2*(λ-1)) * ((n.+1) .* (n .+ (2λ-1))))', ℵ₀, 1,-1)
        ApplyQuasiMatrix(*, Weighted(Ultraspherical{T}(λ-1)), A)
    end
end

# Ultraspherical(λ-1)\ (D*w*Ultraspherical(λ))
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::WeightedUltraspherical)
    w,S = WS.args
    λ = S.λ
    T = eltype(WS)
    if iszero(w.λ)
        D*S
    elseif isorthogonalityweighted(WS) # weights match
        D * Weighted(S)
    else
        error("Not implemented")
    end
end


##########
# Conversion
##########

\(A::Ultraspherical, B::Legendre) = A\Ultraspherical(B)
\(A::Legendre, B::Ultraspherical) = Ultraspherical(A)\B

function \(A::Ultraspherical, B::Jacobi)
    Ã = Jacobi(A)
    Diagonal(Ã[1,:]./A[1,:]) * (Ã\B)
end
function \(A::Jacobi, B::Ultraspherical)
    B̃ = Jacobi(B)
    (A\B̃)*Diagonal(B[1,:]./B̃[1,:])
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

function \(w_A::WeightedUltraspherical, w_B::WeightedUltraspherical)
    wA,A = w_A.args
    wB,B = w_B.args
    T = promote_type(eltype(w_A),eltype(w_B))

    if wA == wB
        A \ B
    elseif B.λ == A.λ+1 && wB.λ == wA.λ+1 # Lower
        λ = convert(T,A.λ)
        _BandedMatrix(Vcat(((2λ:∞) .* ((2λ+1):∞) ./ (4λ .* (λ+1:∞)))',
                            Zeros{T}(1,∞),
                            (-(1:∞) .* (2:∞) ./ (4λ .* (λ+1:∞)))'), ℵ₀, 2,0)
    else
        error("not implemented for $A and $wB")
    end
end

\(w_A::Weighted{<:Any,<:Ultraspherical}, w_B::Weighted{<:Any,<:Ultraspherical}) = convert(WeightedBasis, w_A) \ convert(WeightedBasis, w_B)

\(A::Legendre, wB::WeightedUltraspherical) = Ultraspherical(A) \ wB

function \(A::Ultraspherical, w_B::WeightedUltraspherical) 
    (UltrasphericalWeight(zero(A.λ)) .* A) \ w_B
end


