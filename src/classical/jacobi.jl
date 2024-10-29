abstract type AbstractJacobiWeight{T} <: Weight{T} end

axes(::AbstractJacobiWeight{T}) where T = (Inclusion(ChebyshevInterval{T}()),)

==(w::AbstractJacobiWeight, v::AbstractJacobiWeight) = w.a == v.a && w.b == v.b
function ==(a::AffineQuasiVector, w::AbstractJacobiWeight)
    axes(a,1) == axes(w,1) || return false
    iszero(w.a) && iszero(w.b) && return iszero(a.A) && return iszero(a.b)
    isone(w.a) && iszero(w.b) && return isone(-a.A) && return isone(a.b)
    iszero(w.a) && isone(w.b) && return isone(a.A) && return isone(a.b)
    return false
end
==(w::AbstractJacobiWeight, a::AffineQuasiVector) = a == w

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

AbstractQuasiArray{T}(w::JacobiWeight) where T = JacobiWeight{T}(w.a, w.b)
AbstractQuasiVector{T}(w::JacobiWeight) where T = JacobiWeight{T}(w.a, w.b)


==(A::JacobiWeight, B::JacobiWeight) = A.b == B.b && A.a == B.a

function getindex(w::JacobiWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x)^w.a * (1+x)^w.b
end

show(io::IO, P::JacobiWeight) = summary(io, P)
summary(io::IO, w::JacobiWeight) = print(io, "(1-x)^$(w.a) * (1+x)^$(w.b) on -1..1")

sum(P::JacobiWeight) = jacobimoment(P.a, P.b)

hasboundedendpoints(w::AbstractJacobiWeight) = w.a ≥ 0 && w.b ≥ 0


# support auto-basis determination

singularities(a::AbstractAffineQuasiVector) = singularities(a.x)


singularities(w::JacobiWeight) = w

## default is to just assume no singularities
singularitiesbroadcast(_...) = NoSingularities()

for op in (:+, :*)
    @eval singularitiesbroadcast(::typeof($op), A, B, C, D...) = singularitiesbroadcast(*, singularitiesbroadcast(*, A, B), C, D...)
    @eval singularitiesbroadcast(::typeof($op), ::NoSingularities, ::NoSingularities) = NoSingularities()
    @eval singularitiesbroadcast(::typeof($op), ::NoSingularities, ::NoSingularities, ::NoSingularities) = NoSingularities()
end

singularitiesbroadcast(::typeof(*), V::SubQuasiArray...) = singularitiesbroadcast(*, map(parent,V)...)[parentindices(V...)...]
singularitiesbroadcast(::typeof(*), ::NoSingularities, b) = b
singularitiesbroadcast(::typeof(*), a, ::NoSingularities) = a


# for singularitiesbroadcast(literal_pow), ^, ...)
singularitiesbroadcast(F::Function, G::Function, V::SubQuasiArray, K) = singularitiesbroadcast(F, G, parent(V), K)[parentindices(V)...]
singularitiesbroadcast(F, V::SubQuasiArray...) = singularitiesbroadcast(F, map(parent,V)...)[parentindices(V...)...]
singularitiesbroadcast(F, V::NoSingularities...) = NoSingularities() # default is to assume smooth


abstract type AbstractJacobi{T} <: OrthogonalPolynomial{T} end

struct JacobiTransformPlan{T, CHEB2JAC, DCT} <: Plan{T}
    cheb2jac::CHEB2JAC
    chebtransform::DCT
end

JacobiTransformPlan(c2l, ct) = JacobiTransformPlan{promote_type(eltype(c2l),eltype(ct)),typeof(c2l),typeof(ct)}(c2l, ct)
size(P::JacobiTransformPlan, k...) = size(P.chebtransform, k...)

*(P::JacobiTransformPlan, x::AbstractArray) = P.cheb2jac*(P.chebtransform*x)
\(P::JacobiTransformPlan, x::AbstractArray) = P.chebtransform\(P.cheb2jac\x)


include("legendre.jl")


struct Jacobi{T} <: AbstractJacobi{T}
    a::T
    b::T
    Jacobi{T}(a, b) where T = new{T}(convert(T,a), convert(T,b))
end

Jacobi(a::V, b::T) where {T,V} = Jacobi{float(promote_type(T,V))}(a, b)

AbstractQuasiArray{T}(w::Jacobi) where T = Jacobi{T}(w.a, w.b)
AbstractQuasiMatrix{T}(w::Jacobi) where T = Jacobi{T}(w.a, w.b)


jacobi(a,b) = Jacobi(a,b)
jacobi(a,b, d::AbstractInterval{T}) where T = Jacobi{float(promote_type(eltype(a),eltype(b),T))}(a,b)[affine(d,ChebyshevInterval{T}()), :]
jacobi(a,b, d::ChebyshevInterval{T}) where T = Jacobi{float(promote_type(eltype(a),eltype(b),T))}(a,b)

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

"""
    HalfWeighted{lr}(Jacobi(a,b))

is equivalent to `JacobiWeight(a,0) .* Jacobi(a,b)` (`lr = :a`) or
`JacobiWeight(0,b) .* Jacobi(a,b)` (`lr = :b`)
"""
struct HalfWeighted{lr, T, PP<:AbstractQuasiMatrix{T}} <: AbstractWeighted{T}
    P::PP
end

HalfWeighted{lr}(P) where lr = HalfWeighted{lr,eltype(P),typeof(P)}(P)

weight(wP::HalfWeighted{:a, T, <:Jacobi}) where T = JacobiWeight(wP.P.a,zero(T))
weight(wP::HalfWeighted{:b, T, <:Jacobi}) where T = JacobiWeight(zero(T),wP.P.b)

weight(wP::HalfWeighted{lr, T, <:Normalized}) where {lr,T} = weight(HalfWeighted{lr}(wP.P.P))

axes(Q::HalfWeighted) = axes(Q.P)
copy(Q::HalfWeighted) = Q

==(A::HalfWeighted{lr}, B::HalfWeighted{lr}) where lr = A.P == B.P
==(A::HalfWeighted, B::HalfWeighted) = false

function ==(A::HalfWeighted, wB::WeightedJacobi)
    w,B = arguments(wB)
    A.P == B && w == weight(A)
end
==(wB::WeightedJacobi, A::HalfWeighted) = A == wB
==(A::Jacobi, wB::HalfWeighted{:a}) = A == wB.P && iszero(A.a)
==(A::Jacobi, wB::HalfWeighted{:b}) = A == wB.P && iszero(A.b)

==(wB::HalfWeighted, A::Jacobi) = A == wB


function convert(::Type{WeightedBasis}, Q::HalfWeighted{lr,T,<:Normalized}) where {T,lr}
    w,_ = arguments(convert(WeightedBasis, HalfWeighted{lr}(Q.P.P)))
    w .* Q.P
end

# broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, Q::HalfWeighted) = Q * (Q.P \ (x .* Q.P))

\(w_A::HalfWeighted, w_B::HalfWeighted) = convert(WeightedBasis, w_A) \ convert(WeightedBasis, w_B)
\(w_A::HalfWeighted, B::AbstractQuasiArray) = convert(WeightedBasis, w_A) \ B
\(A::AbstractQuasiArray, w_B::HalfWeighted) = A \ convert(WeightedBasis, w_B)

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

show(io::IO, P::Jacobi) = summary(io, P)
summary(io::IO, P::Jacobi) = print(io, "Jacobi($(P.a), $(P.b))")

###
# transforms
###

grid(P::AbstractJacobi{T}, n::Integer) where T = ChebyshevGrid{1,T}(n)
plotgrid(P::AbstractJacobi{T}, n::Integer) where T = ChebyshevGrid{2,T}(min(40n, MAX_PLOT_POINTS))

plan_transform(::AbstractJacobi{T}, szs::NTuple{N,Int}, dims...) where {T,N} = error("Override")
plan_transform(P::Jacobi{T}, szs::NTuple{N,Int}, dims...) where {T,N} = JacobiTransformPlan(FastTransforms.plan_th_cheb2jac!(T, szs, P.a, P.b, dims), plan_chebyshevtransform(T, szs, dims...))

ldiv(P::Jacobi{V}, f::Inclusion{T}) where {T,V} = _op_ldiv(P, f)
ldiv(P::Jacobi{V}, f::AbstractQuasiFill{T,1}) where {T,V} = _op_ldiv(P, f)
function transform_ldiv(P::Jacobi{V}, f::AbstractQuasiArray) where V
    T = ChebyshevT{V}()
    pad(FastTransforms.th_cheb2jac(paddeddata(T \ f), P.a, P.b, 1), axes(P,2), tail(axes(f))...)
end


########
# Mass Matrix
#########


@simplify *(Ac::QuasiAdjoint{<:Any,<:AbstractJacobi}, B::AbstractJacobi) = legendre_grammatrix(parent(Ac),B)
@simplify *(Ac::QuasiAdjoint{<:Any,<:AbstractJacobi}, B::Weighted{<:Any,<:AbstractJacobi}) = legendre_grammatrix(parent(Ac),B)
grammatrix(A::AbstractJacobi) = legendre_grammatrix(A)
grammatrix(A::Weighted{<:Any,<:AbstractJacobi}) = legendre_grammatrix(A)

@simplify function *(Ac::QuasiAdjoint{<:Any,<:AbstractJacobi}, B::AbstractQuasiVector)
    P = Legendre{eltype(Ac)}()
    (Ac * P) * (P \ B)
end

# 2^{a + b + 1} {\Gamma(n+a+1) \Gamma(n+b+1) \over (2n+a+b+1) \Gamma(n+a+b+1) n!}.
function weightedgrammatrix(P::Jacobi)
    a,b = P.a,P.b
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
    (P\A)' * weightedgrammatrix(P) * (P \ B)
end

########
# Jacobi Matrix
########

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




##########
# Conversion
##########

\(A::Jacobi, B::Legendre) = A\Jacobi(B)
\(A::Legendre, B::Jacobi) = Jacobi(A)\B
\(A::Legendre, B::Weighted{<:Any,<:Jacobi}) = Jacobi(A)\B

function _jacobi_convert_a(a, b) # Jacobi(a+1, b) \ Jacobi(a, b)
    if isone(-a-b)
        Bidiagonal(Vcat(1, ((2:∞) .+ (a+b)) ./ ((3:2:∞) .+ (a+b))), -((1:∞) .+ b) ./ ((3:2:∞) .+ (a+b)), :U)
    else
        Bidiagonal(((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)), -((1:∞) .+ b)./((3:2:∞) .+ (a+b)), :U)
    end
end
function _jacobi_convert_b(a, b) # Jacobi(a, b+1) \ Jacobi(a, b)
    if isone(-a-b)
        Bidiagonal(Vcat(1, ((2:∞) .+ (a+b)) ./ ((3:2:∞) .+ (a+b))), ((1:∞) .+ a) ./ ((3:2:∞) .+ (a+b)), :U)
    else
        Bidiagonal(((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)), ((1:∞) .+ a)./((3:2:∞) .+ (a+b)), :U)
    end
end

# TODO: implement ApplyArray(*, A) in LazyArrays.jl, then these can be simplified
# the type specification is also because LazyArrays.jl is slow otherwise
# getindex and print could be very slow. This needs to be fixed by LazyArrays.jl
function _jacobi_convert_a(a, b, k, T) # Jacobi(a+k, b) \ Jacobi(a, b)
    if iszero(k)
        Eye{T}(∞)
    elseif isone(k)
        _jacobi_convert_a(a, b)
    else
        list = tuple([_jacobi_convert_a(a+j, b) for j in k-1:-1:0]...)
        ApplyArray{T,2,typeof(*),typeof(list)}(*, list)
    end
end
function _jacobi_convert_b(a, b, k, T) # Jacobi(a, b+k) \ Jacobi(a, b)
    if iszero(k)
        Eye{T}(∞)
    elseif isone(k)
        _jacobi_convert_b(a, b)
    else
        list = tuple([_jacobi_convert_b(a, b+j) for j in k-1:-1:0]...)
        ApplyArray{T,2,typeof(*),typeof(list)}(*, list)
    end
end

function \(A::Jacobi, B::Jacobi)
    T = promote_type(eltype(A), eltype(B))
    aa, ab = A.a, A.b
    ba, bb = B.a, B.b
    ka = Integer(aa-ba)
    kb = Integer(ab-bb)
    if ka >= 0
        C1 = _jacobi_convert_a(ba, ab, ka, T)
        if kb >= 0
            C2 = _jacobi_convert_b(ba, bb, kb, T)
        else
            C2 = inv(_jacobi_convert_b(ba, ab, -kb, T))
        end
        if iszero(ka)
            C2
        elseif iszero(kb)
            C1
        else
            list = (C1, C2)
            ApplyArray{T,2,typeof(*),typeof(list)}(*, list)
        end
    else
        inv(B \ A)
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

broadcastbasis(::typeof(+), w_A::Weighted{<:Any,<:Jacobi}, w_B::Weighted{<:Any,<:Jacobi}) = broadcastbasis(+, convert(WeightedBasis,w_A), convert(WeightedBasis,w_B))
broadcastbasis(::typeof(+), w_A::Weighted{<:Any,<:Jacobi}, w_B::WeightedJacobi) = broadcastbasis(+, convert(WeightedBasis,w_A), w_B)
broadcastbasis(::typeof(+), w_A::WeightedJacobi, w_B::Weighted{<:Any,<:Jacobi}) = broadcastbasis(+, w_A, convert(WeightedBasis,w_B))

function \(w_A::WeightedJacobi, w_B::WeightedJacobi)
    wA,A = w_A.args
    wB,B = w_B.args

    if wA == wB
        A \ B
    elseif B.a ≈ A.a && B.b ≈ A.b+1 && wB.b ≈ wA.b+1 && wB.a ≈ wA.a
        Bidiagonal(((2:2:∞) .+ 2A.b)./((2:2:∞) .+ (A.a+A.b)), (2:2:∞)./((2:2:∞) .+ (A.a+A.b)), :L)
    elseif B.a ≈ A.a+1 && B.b ≈ A.b && wB.b ≈ wA.b && wB.a ≈ wA.a+1
        Bidiagonal(((2:2:∞) .+ 2A.a)./((2:2:∞) .+ (A.a+A.b)), -(2:2:∞)./((2:2:∞) .+ (A.a+A.b)), :L)
    elseif wB.a ≥ wA.a+1 && B.a > 0
        J = JacobiWeight(wB.a-1,wB.b) .* Jacobi(B.a-1,B.b)
        (w_A\J) * (J\w_B)
    elseif wB.b ≥ wA.b+1 && B.b > 0
        J = JacobiWeight(wB.a,wB.b-1) .* Jacobi(B.a,B.b-1)
        (w_A\J) * (J\w_B)
    elseif wB.a ≥ wA.a+1
        X = jacobimatrix(B)
        J = JacobiWeight(wB.a-1,wB.b) .* Jacobi(B.a,B.b)
        (w_A\J) * (I-X)
    elseif wB.b ≥ wA.b+1
        X = jacobimatrix(B)
        J = JacobiWeight(wB.a,wB.b-1) .* Jacobi(B.a,B.b)
        (w_A\J) * (I+X)
    else
        error("not implemented for $w_A and $w_B")
    end
end

\(w_A::WeightedJacobi, w_B::Weighted{<:Any,<:Jacobi}) = w_A \ convert(WeightedBasis,w_B)
\(w_A::Weighted{<:Any,<:Jacobi}, w_B::Weighted{<:Any,<:Jacobi}) = convert(WeightedBasis,w_A) \ convert(WeightedBasis,w_B)
\(w_A::Weighted{<:Any,<:Jacobi}, w_B::WeightedJacobi) = convert(WeightedBasis,w_A) \ w_B
\(A::Jacobi, w_B::Weighted{<:Any,<:Jacobi}) = A \ convert(WeightedBasis,w_B)

\(w_A::Weighted{<:Any,<:Jacobi}, B::Legendre) = w_A \ Weighted(Jacobi(B))
\(A::Legendre, wB::WeightedJacobi) = Jacobi(A) \ wB

##########
# Derivatives
##########

# Jacobi(a+1,b+1)\(D*Jacobi(a,b))
diff(S::Jacobi; dims=1) = ApplyQuasiMatrix(*, Jacobi(S.a+1,S.b+1), _BandedMatrix((((1:∞) .+ (S.a + S.b))/2)', ℵ₀, -1,1))


#L_6^t
function diff(WS::HalfWeighted{:a,<:Any,<:Jacobi}; dims=1)
    S = WS.P
    a,b = S.a, S.b
    ApplyQuasiMatrix(*, HalfWeighted{:a}(Jacobi(a-1,b+1)), Diagonal(-(a:∞)))
end

#L_6
function diff(WS::HalfWeighted{:b,<:Any,<:Jacobi}; dims=1)
    S = WS.P
    a,b = S.a, S.b
    ApplyQuasiMatrix(*, HalfWeighted{:b}(Jacobi(a+1,b-1)), Diagonal(b:∞))
end

for ab in (:(:a), :(:b))
    @eval function diff(WS::HalfWeighted{$ab,<:Any,<:Normalized}; dims=1)
        P,M = arguments(ApplyLayout{typeof(*)}(), WS.P)
        ApplyQuasiMatrix(*, diff(HalfWeighted{$ab}(P);dims=dims), M)
    end
end


function diff(WS::Weighted{<:Any,<:Jacobi}; dims=1)
    # L_1^t
    S = WS.P
    a,b = S.a, S.b
    if a == b == 0
        diff(S)
    elseif iszero(a)
        diff(HalfWeighted{:b}(S))
    elseif iszero(b)
        diff(HalfWeighted{:a}(S))
    else
        ApplyQuasiMatrix(*, Weighted(Jacobi(a-1, b-1)), _BandedMatrix((-2*(1:∞))', ℵ₀, 1,-1))
    end
end


# Jacobi(a-1,b-1)\ (D*w*Jacobi(a,b))
function diff(WS::WeightedJacobi; dims=1)
    w,S = WS.args
    a,b = S.a, S.b
    if isorthogonalityweighted(WS) # L_1^t
        diff(Weighted(S))
    elseif w.a == w.b == 0
        diff(S)
    elseif iszero(w.a) && w.b == b #L_6
        diff(HalfWeighted{:b}(S))
    elseif iszero(w.b) && w.a == a #L_6^t
        diff(HalfWeighted{:a}(S))
    elseif iszero(w.a)
        # We differentiate
        # D * ((1+x)^w.b * P^(a,b)) == D * ((1+x)^(w.b-b) * (1+x)^b * P^(a,b))
        #    == (1+x)^(w.b-1) * (w.b-b) * P^(a,b) + (1+x)^(w.b-b) * D*((1+x)^b*P^(a,b))
        #    == (1+x)^(w.b-1) * P^(a+1,b) ((w.b-b) * C2 + C1 * W)
        W = HalfWeighted{:b}(Jacobi(a+1, b-1)) \ diff(HalfWeighted{:b}(S))
        J = Jacobi(a+1,b) # range Jacobi
        C1 = J \ Jacobi(a+1, b-1)
        C2 = J \ Jacobi(a,b)
        ApplyQuasiMatrix(*, JacobiWeight(w.a,w.b-1) .* J, (w.b-b) * C2 + C1 * W)
    elseif iszero(w.b)
        W = HalfWeighted{:a}(Jacobi(a-1, b+1)) \ diff(HalfWeighted{:a}(S))
        J = Jacobi(a,b+1) # range Jacobi
        C1 = J \ Jacobi(a-1, b+1)
        C2 = J \ Jacobi(a,b)
        ApplyQuasiMatrix(*, JacobiWeight(w.a-1,w.b) .* J, -(w.a-a) * C2 + C1 * W)
    elseif iszero(a) && iszero(b) # Legendre
        # D * ((1+x)^w.b * (1-x)^w.a * P))
        #    == (1+x)^(w.b-1) * (1-x)^(w.a-1) * ((1-x) * (w.b) * P - (1+x) * w.a * P + (1-x^2) * D * P)
        #    == (1+x)^(w.b-1) * (1-x)^(w.a-1) * ((1-x) * (w.b) * P - (1+x) * w.a * P + P * L * W)
        J = Jacobi(a+1,b+1) # range space
        W = J \ diff(S)
        X = jacobimatrix(S)
        L = S \ Weighted(J)
        (JacobiWeight(w.a-1,w.b-1) .* S) *  (((w.b-w.a)*I-(w.a+w.b) * X) + L*W)
    else
        # We differentiate
        # D * ((1+x)^w.b * (1-x)^w.a * P^(a,b)) == D * ((1+x)^(w.b-b) * (1-x)^(w.a-a)  * (1+x)^b * (1-x)^a * P^(a,b))
        #    == (1+x)^(w.b-1) * (1-x)^(w.a-1) * ((1-x) * (w.b-b) * P^(a,b) + (1+x) * (a-w.a) * P^(a,b))
        #        + (1+x)^(w.b-b) * (1-x)^(w.a-a) * D * ((1+x)^b * (1-x)^a * P^(a,b)))
        
        W = Weighted(Jacobi(a-1,b-1)) \ diff(Weighted(S))
        X = jacobimatrix(S)
        C = S \ Jacobi(a-1,b-1)
        (JacobiWeight(w.a-1,w.b-1) .* S) *  (((w.b-b+a-w.a)*I+(a-w.a-w.b+b) * X) + C*W)
    end
end

function diff(WS::WeightedBasis{<:Any,<:JacobiWeight,<:Legendre}; dims=1)
    w,S = WS.args
    diff(w .* Jacobi(S))
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


###
# sum
###

_sum(P::AbstractJacobi{T}, dims) where T = 2 * (Legendre{T}() \ P)[1:1,:]


