abstract type AbstractJacobiWeight{T} <: Weight{T} end

axes(::AbstractJacobiWeight{T}) where T = (Inclusion(ChebyshevInterval{T}()),)

==(w::AbstractJacobiWeight, v::AbstractJacobiWeight) = w.a == v.a && w.b == v.b

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(*), w::AbstractJacobiWeight, v::AbstractJacobiWeight) =
    JacobiWeight(w.a + v.a, w.b + v.b)

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(sqrt), w::AbstractJacobiWeight) =
    JacobiWeight(w.a/2, w.b/2)

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, w::AbstractJacobiWeight, ::Base.RefValue{Val{k}}) where k = 
    JacobiWeight(k * w.a, k * w.b)

struct JacobiWeight{T} <: AbstractJacobiWeight{T}
    a::T
    b::T
    JacobiWeight{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end

JacobiWeight(a::V, b::T) where {T,V} = JacobiWeight{promote_type(T,V)}(a,b)
jacobiweight(a,b, d::AbstractInterval{T}) where T = JacobiWeight(a,b)[affine(d,ChebyshevInterval{T}())]

==(A::JacobiWeight, B::JacobiWeight) = A.b == B.b && A.a == B.a

function getindex(w::JacobiWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x)^w.a * (1+x)^w.b
end

summary(io::IO, w::JacobiWeight) = print(io, "(1-x)^$(w.a) * (1+x)^$(w.b) on -1..1")

sum(P::JacobiWeight) = jacobimoment(P.a, P.b)

struct LegendreWeight{T} <: AbstractJacobiWeight{T} end
LegendreWeight() = LegendreWeight{Float64}()
legendreweight(d::AbstractInterval{T}) where T = LegendreWeight{float(T)}()[affine(d,ChebyshevInterval{T}())]

function getindex(w::LegendreWeight{T}, x::Number) where T
    x ∈ axes(w,1) || throw(BoundsError())
    one(T)
end

getproperty(w::LegendreWeight{T}, ::Symbol) where T = zero(T)

sum(::LegendreWeight{T}) where T = 2one(T)

_weighted(::LegendreWeight, P) = P
_weighted(::SubQuasiArray{<:Any,1,<:LegendreWeight}, P) = P

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(*), ::LegendreWeight{T}, ::LegendreWeight{V}) where {T,V} =
    LegendreWeight{promote_type(T,V)}()

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(sqrt), w::LegendreWeight{T}) where T = w

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, w::LegendreWeight, ::Base.RefValue{Val{k}}) where k = w

# support auto-basis determination

singularities(a::AbstractAffineQuasiVector) = singularities(a.x)
singularitiesbroadcast(_, L::LegendreWeight) = L # Assume we stay smooth
singularitiesbroadcast(::typeof(exp), L::LegendreWeight) = L
singularitiesbroadcast(::typeof(Base.literal_pow), ::typeof(^), L::LegendreWeight, ::Val) = L
for op in (:+, :-, :*)
    @eval begin
        singularitiesbroadcast(::typeof($op), ::LegendreWeight{T}, ::LegendreWeight{V}) where {T,V} = LegendreWeight{promote_type(T,V)}()
        singularitiesbroadcast(::typeof($op), L::LegendreWeight, ::NoSingularities) = L
        singularitiesbroadcast(::typeof($op), ::NoSingularities, L::LegendreWeight) = L
    end
end
singularitiesbroadcast(::typeof(^), L::LegendreWeight, ::NoSingularities) = L
singularitiesbroadcast(::typeof(/), ::NoSingularities, L::LegendreWeight) = L # can't find roots

_parent(::NoSingularities) = NoSingularities()
_parent(a) = parent(a)
_parentindices(a::NoSingularities, b...) = _parentindices(b...)
_parentindices(a, b...) = parentindices(a)
singularitiesbroadcast(F::Function, G::Function, V::SubQuasiArray, K) = singularitiesbroadcast(F, G, parent(V), K)[parentindices(V)...]
singularitiesbroadcast(F, V::Union{NoSingularities,SubQuasiArray}...) = singularitiesbroadcast(F, map(_parent,V)...)[_parentindices(V...)...]


singularitiesbroadcast(::typeof(*), ::LegendreWeight, b::AbstractJacobiWeight) = b
singularitiesbroadcast(::typeof(*), a::AbstractJacobiWeight, ::LegendreWeight) = a

abstract type AbstractJacobi{T} <: OrthogonalPolynomial{T} end

singularities(::AbstractJacobi{T}) where T = LegendreWeight{T}()
singularities(::Inclusion{T,<:AbstractInterval}) where T = LegendreWeight{T}()
singularities(d::Inclusion{T,<:Interval}) where T = LegendreWeight{T}()[affine(d,ChebyshevInterval{T}())]

struct Legendre{T} <: AbstractJacobi{T} end
Legendre() = Legendre{Float64}()

legendre() = Legendre()
legendre(d::AbstractInterval{T}) where T = Legendre{float(T)}()[affine(d,ChebyshevInterval{T}()), :]

"""
     legendrep(n, z)

computes the `n`-th Legendre polynomial at `z`.
"""
legendrep(n::Integer, z::Number) = Base.unsafe_getindex(Legendre{typeof(z)}(), z, n+1)


==(::Legendre, ::Legendre) = true

OrthogonalPolynomial(w::LegendreWeight{T}) where {T} = Legendre{T}()
orthogonalityweight(::Legendre{T}) where T = LegendreWeight{T}()

function qr(P::Legendre)
    Q = Normalized(P)
    QuasiQR(Q, Diagonal(Q.scaling))
end

struct Jacobi{T} <: AbstractJacobi{T}
    a::T
    b::T
    Jacobi{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end

Jacobi(a::V, b::T) where {T,V} = Jacobi{float(promote_type(T,V))}(a, b)

jacobi(a,b) = Jacobi(a,b)
jacobi(a,b, d::AbstractInterval{T}) where T = Jacobi{float(promote_type(eltype(a),eltype(b),T))}(a,b)[affine(d,ChebyshevInterval{T}()), :]

Jacobi(P::Legendre{T}) where T = Jacobi(zero(T), zero(T))

"""
     jacobip(n, a, b, z)

computes the `n`-th Jacobi polynomial, orthogonal with 
respec to `(1-x)^a*(1+x)^b`, at `z`.
"""
jacobip(n::Integer, a, b, z::Number) = Base.unsafe_getindex(Jacobi{promote_type(typeof(a), typeof(b), typeof(z))}(a,b), z, n+1)
normalizedjacobip(n::Integer, a, b, z::Number) = Base.unsafe_getindex(Normalized(Jacobi{promote_type(typeof(a), typeof(b), typeof(z))}(a,b)), z, n+1)

OrthogonalPolynomial(w::JacobiWeight) = Jacobi(w.a, w.b)
orthogonalityweight(P::Jacobi) = JacobiWeight(P.a, P.b)

const WeightedJacobi{T} = WeightedBasis{T,<:JacobiWeight,<:Jacobi}

WeightedJacobi(a,b) = JacobiWeight(a,b) .* Jacobi(a,b)
WeightedJacobi{T}(a,b) where T = JacobiWeight{T}(a,b) .* Jacobi{T}(a,b)


"""
    HalfWeighted{lr}(Jacobi(a,b))

is equivalent to `JacobiWeight(a,0) .* Jacobi(a,b)` (`lr = :a`) or
`JacobiWeight(0,b) .* Jacobi(a,b)` (`lr = :b`)
"""
struct HalfWeighted{lr, T, PP<:AbstractQuasiMatrix{T}} <: AbstractWeighted{T}
    P::PP
end

HalfWeighted{lr}(P) where lr = HalfWeighted{lr,eltype(P),typeof(P)}(P)

axes(Q::HalfWeighted) = axes(Q.P)
copy(Q::HalfWeighted) = Q

==(A::HalfWeighted{lr}, B::HalfWeighted{lr}) where lr = A.P == B.P
==(A::HalfWeighted, B::HalfWeighted) = false

convert(::Type{WeightedOrthogonalPolynomial}, Q::HalfWeighted{:a,T,<:Jacobi}) where T = JacobiWeight(Q.P.a,zero(T)) .* Q.P
convert(::Type{WeightedOrthogonalPolynomial}, Q::HalfWeighted{:b,T,<:Jacobi}) where T = JacobiWeight(zero(T),Q.P.b) .* Q.P
function convert(::Type{WeightedOrthogonalPolynomial}, Q::HalfWeighted{lr,T,<:Normalized}) where {T,lr}
    w,_ = arguments(convert(WeightedOrthogonalPolynomial, HalfWeighted{lr}(Q.P.P)))
    w .* Q.P
end

getindex(Q::HalfWeighted, x::Union{Number,AbstractVector}, jr::Union{Number,AbstractVector}) = convert(WeightedOrthogonalPolynomial, Q)[x,jr]

broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, Q::HalfWeighted) = Q * (Q.P \ (x .* Q.P))

\(w_A::HalfWeighted, w_B::HalfWeighted) = convert(WeightedOrthogonalPolynomial, w_A) \ convert(WeightedOrthogonalPolynomial, w_B)
\(w_A::HalfWeighted, B::AbstractQuasiArray) = convert(WeightedOrthogonalPolynomial, w_A) \ B
\(A::AbstractQuasiArray, w_B::HalfWeighted) = A \ convert(WeightedOrthogonalPolynomial, w_B)

function _norm_expand_ldiv(A, w_B)
    w,B = w_B.args
    B̃,D = arguments(ApplyLayout{typeof(*)}(), B)
    (A \ (w .* B̃)) * D
end
\(A::AbstractQuasiArray, w_B::WeightedOrthogonalPolynomial{<:Any,<:Weight,<:Normalized}) = _norm_expand_ldiv(A, w_B)
\(A::WeightedOrthogonalPolynomial, w_B::WeightedOrthogonalPolynomial{<:Any,<:Weight,<:Normalized}) = _norm_expand_ldiv(A, w_B)

axes(::AbstractJacobi{T}) where T = (Inclusion{T}(ChebyshevInterval{real(T)}()), oneto(∞))
==(P::Jacobi, Q::Jacobi) = P.a == Q.a && P.b == Q.b
==(P::Legendre, Q::Jacobi) = Jacobi(P) == Q
==(P::Jacobi, Q::Legendre) = P == Jacobi(Q)
==(A::WeightedJacobi, B::WeightedJacobi) = A.args == B.args
==(A::WeightedJacobi, B::Jacobi{T}) where T = A == JacobiWeight(zero(T),zero(T)).*B
==(A::WeightedJacobi, B::Legendre) = A == Jacobi(B)
==(A::Legendre, B::WeightedJacobi) = Jacobi(A) == B
==(A::Jacobi{T}, B::WeightedJacobi) where T = JacobiWeight(zero(T),zero(T)).*A == B
==(A::Legendre, B::Weighted{<:Any,<:AbstractJacobi}) = A == B.P
==(A::Weighted{<:Any,<:AbstractJacobi}, B::Legendre) = A.P == B


summary(io::IO, P::Jacobi) = print(io, "Jacobi($(P.a), $(P.b))")

###
# transforms
###

function grid(Pn::SubQuasiArray{T,2,<:AbstractJacobi,<:Tuple{Inclusion,Any}}) where T
    kr,jr = parentindices(Pn)
    ChebyshevGrid{1,T}(maximum(jr))
end

function plotgrid(Pn::SubQuasiArray{T,2,<:AbstractJacobi,<:Tuple{Inclusion,Any}}) where T
    kr,jr = parentindices(Pn)
    ChebyshevGrid{2,T}(40maximum(jr))
end


function ldiv(::Legendre{V}, f::AbstractQuasiVector) where V
    T = ChebyshevT{V}()
    [cheb2leg(paddeddata(T \ f)); zeros(V,∞)]
end

function ldiv(P::Jacobi{V}, f::AbstractQuasiVector) where V
    T = ChebyshevT{V}()
    [cheb2jac(paddeddata(T \ f), P.a, P.b); zeros(V,∞)]
end


########
# Mass Matrix
#########

legendre_massmatrix(::Type{T}) where T = Diagonal(convert(T,2) ./ (2(0:∞) .+ 1))

function legendre_massmatrix(Ac, B)
    A = parent(Ac)
    P = Legendre{eltype(B)}()
    (P\A)'*legendre_massmatrix(eltype(P))*(P\B)
end

@simplify *(Ac::QuasiAdjoint{<:Any,<:Legendre}, B::Legendre) = legendre_massmatrix(Ac, B)
@simplify *(Ac::QuasiAdjoint{<:Any,<:AbstractJacobi}, B::AbstractJacobi) = legendre_massmatrix(Ac,B)

# 2^{a + b + 1} {\Gamma(n+a+1) \Gamma(n+b+1) \over (2n+a+b+1) \Gamma(n+a+b+1) n!}.

function jacobi_massmatrix(a, b)
    n = 0:∞
    Diagonal(2^(a+b+1) .* (exp.(loggamma.(n .+ (a+1)) .+ loggamma.(n .+ (b+1)) .- loggamma.(n .+ (a+b+1)) .- loggamma.(n .+ 1)) ./ (2n .+ (a+b+1))))
end


@simplify function *(wAc::QuasiAdjoint{<:Any,<:WeightedBasis{<:Any,<:AbstractJacobiWeight}}, wB::WeightedBasis{<:Any,<:AbstractJacobiWeight})
    w,A = arguments(parent(wAc))
    v,B = arguments(wB)
    A'*((w .* v) .* B)
end
@simplify function *(Ac::QuasiAdjoint{<:Any,<:AbstractJacobi}, wB::WeightedBasis{<:Any,<:AbstractJacobiWeight,<:AbstractJacobi})
    A = parent(Ac)
    w,B = arguments(wB)
    P = Jacobi(w.a, w.b)
    (P\A)' * jacobi_massmatrix(w.a, w.b) * (P \ B)
end

########
# Jacobi Matrix
########

jacobimatrix(::Legendre{T}) where T =  Tridiagonal((one(T):∞)./(1:2:∞), Zeros{T}(∞), (one(T):∞)./(3:2:∞))

# These return vectors A[k], B[k], C[k] are from DLMF. Cause of MikaelSlevinsky we need an extra entry in C ... for now.
function recurrencecoefficients(::Legendre{T}) where T
    n = zero(T):∞
    ((2n .+ 1) ./ (n .+ 1), Zeros{T}(∞), n ./ (n .+ 1))
end

function jacobimatrix(J::Jacobi)
    b,a = J.b,J.a
    n = 0:∞
    B = Vcat(2 / (a+b+2),  2 .* (n .+ 2) .* (n .+ (a+b+2)) ./ ((2n .+ (a+b+3)) .* (2n .+ (a+b+4))))
    A = Vcat((b-a) / (a+b+2), (b^2-a^2) ./ ((2n .+ (a+b+2)) .* (2n .+ (a+b+4))))
    C = 2 .* (n .+ (a + 1)) .* (n .+ (b + 1)) ./ ((2n .+ (a+b+2)) .* (2n .+ (a+b+3)))

    Tridiagonal(B,A,C)
end

function recurrencecoefficients(P::Jacobi)
    n = 0:∞
    ñ = 1:∞
    a,b = P.a,P.b
    A = Vcat((a+b+2)/2, (2ñ .+ (a+b+1)) .* (2ñ .+ (a+b+2)) ./ ((2*(ñ .+ 1)) .* (ñ .+ (a+b+1))))
    # n = 0 is special to avoid divide-by-zero
    B = Vcat((a-b)/2, (a^2 - b^2) * (2ñ .+ (a + b+1)) ./ ((2*(ñ .+ 1)) .* (ñ .+ (a+b+1)) .* (2ñ .+ (a+b))))
    C = ((n .+ a) .* (n .+ b) .* (2n .+ (a+b+2))) ./ (ñ .* (n .+ (a +b+1)) .* (2n .+ (a+b)))
    (A,B,C)
end

# explicit special case for normalized Legendre
# todo: do we want these explicit constructors for normalized Legendre?
# function jacobimatrix(::Normalized{<:Any,<:Legendre{T}}) where T
#     b = (one(T):∞) ./sqrt.(4 .*(one(T):∞).^2 .-1)
#     Symmetric(_BandedMatrix(Vcat(zeros(∞)', (b)'), ∞, 1, 0), :L)
# end
# function recurrencecoefficients(::Normalized{<:Any,<:Legendre{T}}) where T
#     n = zero(T):∞
#     nn = one(T):∞
#     ((2n .+ 1) ./ (n .+ 1) ./ sqrt.(1 .-2 ./(3 .+2n)), Zeros{T}(∞), Vcat(zero(T),nn ./ (nn .+ 1) ./ sqrt.(1 .-4 ./(3 .+2nn))))
# end

@simplify *(X::Identity, P::Legendre) = ApplyQuasiMatrix(*, P, P\(X*P))



##########
# Conversion
##########

\(A::Jacobi, B::Legendre) = A\Jacobi(B)
\(A::Legendre, B::Jacobi) = Jacobi(A)\B

function \(A::Jacobi, B::Jacobi)
    T = promote_type(eltype(A), eltype(B))
    a,b = B.a,B.b
    if A.a ≈ a && A.b ≈ b
        Eye{T}(∞)
    elseif isone(-a-b) && A.a == a && A.b == b+1
        Bidiagonal(Vcat(1, ((2:∞) .+ (a+b)) ./ ((3:2:∞) .+ (a+b))), ((1:∞) .+ a) ./ ((3:2:∞) .+ (a+b)), :U)
    elseif isone(-a-b) && A.a == a+1 && A.b == b
        Bidiagonal(Vcat(1, ((2:∞) .+ (a+b)) ./ ((3:2:∞) .+ (a+b))), -((1:∞) .+ b) ./ ((3:2:∞) .+ (a+b)), :U)
    elseif A.a == a && A.b == b+1
        Bidiagonal(((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)), ((1:∞) .+ a)./((3:2:∞) .+ (a+b)), :U)
    elseif A.a == a+1 && A.b == b
        Bidiagonal(((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)), -((1:∞) .+ b)./((3:2:∞) .+ (a+b)), :U)
    elseif A.a ≥ a+1
        J = Jacobi(a+1,b)
        (A \ J) * (J \ B)
    elseif A.b ≥ b+1
        J = Jacobi(a,b+1)
        (A \ J) * (J \ B)
    elseif isinteger(A.a-a) && isinteger(A.b-b)
        inv(B \ A)
    else
        error("not implemented for $A and $B")
    end
end

function \(A::Jacobi, w_B::WeightedJacobi)
    a,b = A.a,A.b
    (JacobiWeight(zero(a),zero(b)) .* A) \ w_B
end

function \(w_A::WeightedJacobi, B::Jacobi)
    a,b = B.a,B.b
    w_A \ (JacobiWeight(zero(a),zero(b)) .* B)
end

function \(A::AbstractJacobi, w_B::WeightedJacobi)
    Ã = Jacobi(A)
    (A \ Ã) * (Ã \ w_B)
end
function \(w_A::WeightedJacobi, B::AbstractJacobi)
    B̃ = Jacobi(B)
    (w_A \ B̃) * (B̃ \ B)
end


function broadcastbasis(::typeof(+), w_A::WeightedJacobi, w_B::WeightedJacobi)
    wA,A = w_A.args
    wB,B = w_B.args

    w = JacobiWeight(min(wA.a,wB.a), min(wA.b,wB.b))
    P = Jacobi(max(A.a,B.a + w.a - wB.a), max(A.b,B.b + w.b - wB.b))
    w .* P
end

function \(w_A::WeightedJacobi, w_B::WeightedJacobi)
    wA,A = w_A.args
    wB,B = w_B.args

    if wA == wB
        A \ B
    elseif B.a == A.a && B.b == A.b+1 && wB.b == wA.b+1 && wB.a == wA.a
        Bidiagonal(((2:2:∞) .+ 2A.b)./((2:2:∞) .+ (A.a+A.b)), (2:2:∞)./((2:2:∞) .+ (A.a+A.b)), :L)
    elseif B.a == A.a+1 && B.b == A.b && wB.b == wA.b && wB.a == wA.a+1
        Bidiagonal(((2:2:∞) .+ 2A.a)./((2:2:∞) .+ (A.a+A.b)), -(2:2:∞)./((2:2:∞) .+ (A.a+A.b)), :L)
    elseif wB.a ≥ wA.a+1
        J = JacobiWeight(wB.a-1,wB.b) .* Jacobi(B.a-1,B.b)
        (w_A\J) * (J\w_B)
    elseif wB.b ≥ wA.b+1
        J = JacobiWeight(wB.a,wB.b-1) .* Jacobi(B.a,B.b-1)
        (w_A\J) * (J\w_B)
    else
        error("not implemented for $A and $wB")
    end
end

\(A::Legendre, wB::WeightedJacobi) = Jacobi(A) \ wB

##########
# Derivatives
##########

# Jacobi(a+1,b+1)\(D*Jacobi(a,b))
@simplify *(D::Derivative{<:Any,<:AbstractInterval}, S::Jacobi) = Jacobi(S.a+1,S.b+1) * _BandedMatrix((((1:∞) .+ (S.a + S.b))/2)', ℵ₀, -1,1)


#L_6^t
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::HalfWeighted{:a,<:Any,<:Jacobi})
    S = WS.P
    a,b = S.a, S.b
    HalfWeighted{:a}(Jacobi(a-1,b+1)) * Diagonal(-(a:∞))
end

#L_6
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::HalfWeighted{:b,<:Any,<:Jacobi})
    S = WS.P
    a,b = S.a, S.b
    HalfWeighted{:b}(Jacobi(a+1,b-1)) * Diagonal(b:∞)
end

for ab in (:(:a), :(:b))
    @eval @simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::HalfWeighted{$ab,<:Any,<:Normalized})
        P,M = arguments(ApplyLayout{typeof(*)}(), WS.P)
        D * HalfWeighted{$ab}(P) * M
    end
end


@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::Weighted{<:Any,<:Jacobi})
    # L_1^t
    S = WS.P
    a,b = S.a, S.b
    if a == b == 0
        D*S
    elseif iszero(a)
        D * HalfWeighted{:b}(S)
    elseif iszero(b)
        D * HalfWeighted{:a}(S)
    else
        Weighted(Jacobi(a-1, b-1)) * _BandedMatrix((-2*(1:∞))', ℵ₀, 1,-1)
    end
end


# Jacobi(a-1,b-1)\ (D*w*Jacobi(a,b))
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::WeightedJacobi)
    w,S = WS.args
    a,b = S.a, S.b
    if isorthogonalityweighted(WS) # L_1^t
        D * Weighted(S)
    elseif w.a == w.b == 0
        D*S
    elseif iszero(w.a) && w.b == b #L_6
        D * HalfWeighted{:b}(S)
    elseif iszero(w.b) && w.a == a #L_6^t
        D * HalfWeighted{:a}(S)
    elseif iszero(w.a)
        W = (JacobiWeight(w.a, b-1) .* Jacobi(a+1, b-1)) \ (D * (JacobiWeight(w.a,b) .* S))
        J = Jacobi(a+1,b) # range Jacobi
        C1 = J \ Jacobi(a+1, b-1)
        C2 = J \ Jacobi(a,b)
        ApplyQuasiMatrix(*, JacobiWeight(w.a,w.b-1) .* J, (w.b-b) * C2 + C1 * W)
    elseif iszero(w.b)
        W = (JacobiWeight(a-1, w.b) .* Jacobi(a-1, b+1)) \ (D * (JacobiWeight(a,w.b) .* S))
        J = Jacobi(a,b+1) # range Jacobi
        C1 = J \ Jacobi(a-1, b+1)
        C2 = J \ Jacobi(a,b)
        ApplyQuasiMatrix(*, JacobiWeight(w.a-1,w.b) .* J, -(w.a-a) * C2 + C1 * W)
    else
        error("Not implemented")
    end
end


function \(L::Legendre, WS::WeightedBasis{Bool,JacobiWeight{Bool},Jacobi{Bool}})
    w,S = WS.args
    if w.b && w.a
        @assert S.b && S.a
        _BandedMatrix(Vcat(((2:2:∞)./(3:2:∞))', Zeros(1,∞), (-(2:2:∞)./(3:2:∞))'), ℵ₀, 2,0)
    elseif w.b && !w.a
        @assert S.b && !S.a
        Bidiagonal(Ones{eltype(L)}(∞), Ones{eltype(L)}(∞), :L)
    elseif !w.b && w.a
        @assert !S.b && S.a
        Bidiagonal(Ones{eltype(L)}(∞), -Ones{eltype(L)}(1,∞), :L)
    else
        error("Not implemented")
    end
end

@simplify function *(St::QuasiAdjoint{Bool,Jacobi{Bool}}, WS::WeightedBasis{Int,JacobiWeight{Int},Jacobi{Bool}})
    w = parent(W)
    (w.b == 2 && S.b && w.a == 2 && S.a && parent(St) == S) || throw(ArgumentError())
    W_sqrt = Diagonal(JacobiWeight(true,true))
    L = Legendre()
    A = PInv(L)*W_sqrt*S
    A'*(L'L)*A
end


###
# Splines
###

function \(A::Legendre, B::HeavisideSpline)
    @assert B.points == -1:2:1
    Vcat(1, Zeros(∞,1))
end

###
# sum
###

function _sum(P::Legendre{T}, dims) where T
    @assert dims == 1
    Hcat(convert(T, 2), Zeros{T}(1,∞))
end

_sum(p::SubQuasiArray{T,1,Legendre{T},<:Tuple{Inclusion,Int}}, ::Colon) where T = parentindices(p)[2] == 1 ? convert(T, 2) : zero(T)

