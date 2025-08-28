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

"""
    JacobiWeight{T}(a,b)
    JacobiWeight(a,b)

The quasi-vector representing the Jacobi weight function ``(1-x)^a (1+x)^b`` on ``[-1,1]``. See also [`jacobiweight`](@ref) and [`Jacobi`](@ref).
# Examples
```jldoctest
julia> J=JacobiWeight(1.0,1.0)
(1-x)^1.0 * (1+x)^1.0 on -1..1

julia> J[0.5]
0.75

julia> axes(J)
(Inclusion(-1.0 .. 1.0 (Chebyshev)),)
```
"""
struct JacobiWeight{T,V} <: AbstractJacobiWeight{T}
    a::V
    b::V
    JacobiWeight{T,V}(a, b) where {T,V} = new{T,V}(convert(V,a), convert(V,b))
end

JacobiWeight{T}(a::V, b::V) where {T,V} = JacobiWeight{T,V}(a, b)
JacobiWeight{T}(a, b) where T = JacobiWeight{T}(promote(a,b)...)
JacobiWeight(a::V, b::T) where {T,V} = JacobiWeight{float(promote_type(T,V))}(a, b)


"""
    jacobiweight(a,b, d::AbstractInterval)

The [`JacobiWeight`](@ref) affine-mapped to interval `d`.

# Examples
```jldoctest
julia> J = jacobiweight(1, 1, 0..1)
(1-x)^1 * (1+x)^1 on -1..1 affine mapped to 0 .. 1

julia> axes(J)
(Inclusion(0 .. 1),)

julia> J[0.5]
1.0
```
"""
jacobiweight(a,b, d::AbstractInterval{T}) where T = JacobiWeight(a,b)[affine(d,ChebyshevInterval{T}())]

AbstractQuasiArray{T}(w::JacobiWeight) where T = JacobiWeight{T}(w.a, w.b)
AbstractQuasiVector{T}(w::JacobiWeight) where T = JacobiWeight{T}(w.a, w.b)


==(A::JacobiWeight, B::JacobiWeight) = A.b == B.b && A.a == B.a

function getindex(w::JacobiWeight{T}, x::Number) where T
    x ∈ axes(w,1) || throw(BoundsError())
    convert(T, (1-x)^w.a * (1+x)^w.b)
end

show(io::IO, P::JacobiWeight) = summary(io, P)
summary(io::IO, w::JacobiWeight) = print(io, "(1-x)^$(w.a) * (1+x)^$(w.b) on -1..1")

sum(P::JacobiWeight) = jacobimoment(P.a, P.b)

hasboundedendpoints(w::AbstractJacobiWeight) = w.a ≥ 0 && w.b ≥ 0


# support auto-basis determination

singularities(a::AbstractAffineQuasiVector) = singularities(a.x)


singularities(w::AbstractJacobiWeight) = w


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

singularitiesbroadcast(::typeof(*), ::LegendreWeight, b::AbstractJacobiWeight) = b
singularitiesbroadcast(::typeof(*), a::AbstractJacobiWeight, ::LegendreWeight) = a

"""
    Jacobi{T}(a,b)
    Jacobi(a,b)

The quasi-matrix representing Jacobi polynomials, where the first axes represents the interval and the second axes represents the polynomial index (starting from 1). See also [`jacobi`](@ref), [`jacobip`](@ref) and [`JacobiWeight`](@ref).

The eltype, when not specified, will be converted to a floating point data type.
# Examples
```jldoctest
julia> J=Jacobi(0, 0) # The eltype will be converted to float
Jacobi(0, 0)

julia> axes(J)
(Inclusion(-1.0 .. 1.0 (Chebyshev)), OneToInf())

julia> J[0,:] # Values of polynomials at x=0
ℵ₀-element view(::Jacobi{Float64, $Int}, 0.0, :) with eltype Float64 with indices OneToInf():
  1.0
  0.0
 -0.5
 -0.0
  0.375
  0.0
 -0.3125
 -0.0
  0.2734375
  0.0
  ⋮

julia> J0=J[:,1]; # J0 is the first Jacobi polynomial which is constant.

julia> J0[0],J0[0.5]
(1.0, 1.0)
```
"""
struct Jacobi{T,V} <: AbstractJacobi{T}
    a::V
    b::V
    Jacobi{T,V}(a, b) where {T,V} = new{T,V}(convert(V,a), convert(V,b))
end

Jacobi{T}(a::V, b::V) where {T,V} = Jacobi{T,V}(a, b)
Jacobi{T}(a, b) where T = Jacobi{T}(promote(a,b)...)
Jacobi(a::V, b::T) where {T,V} = Jacobi{float(promote_type(T,V))}(a, b)

AbstractQuasiArray{T}(w::Jacobi) where T = Jacobi{T}(w.a, w.b)
AbstractQuasiMatrix{T}(w::Jacobi) where T = Jacobi{T}(w.a, w.b)

"""
    jacobi(a,b, d::AbstractInterval)

The [`Jacobi`](@ref) polynomials affine-mapped to interval `d`.

# Examples
```jldoctest
julia> J = jacobi(1, 1, 0..1)
Jacobi(1, 1) affine mapped to 0 .. 1

julia> axes(J)
(Inclusion(0 .. 1), OneToInf())

julia> J[0,:]
ℵ₀-element view(::Jacobi{Float64, $Int}, -1.0, :) with eltype Float64 with indices OneToInf():
   1.0
  -2.0
   3.0
  -4.0
   5.0
  -6.0
   7.0
  -8.0
   9.0
 -10.0
   ⋮
```
"""
jacobi(a,b) = Jacobi(a,b)
jacobi(a,b, d::AbstractInterval{T}) where T = Jacobi{float(promote_type(eltype(a),eltype(b),T))}(a,b)[affine(d,ChebyshevInterval{T}()), :]
jacobi(a,b, d::ChebyshevInterval{T}) where T = Jacobi{float(promote_type(eltype(a),eltype(b),T))}(a,b)

Jacobi(P::Legendre{T}) where T = Jacobi(zero(T), zero(T))




"""
     jacobip(n, a, b, z)

computes the `n`-th Jacobi polynomial, orthogonal with
respec to `(1-x)^a*(1+x)^b`, at `z`.
"""
jacobip(n::Integer, a, b, z) = Base.unsafe_getindex(Jacobi{polynomialtype(promote_type(typeof(a), typeof(b)), typeof(z))}(a,b), z, n+1)
normalizedjacobip(n::Integer, a, b, z) = Base.unsafe_getindex(Normalized(Jacobi{polynomialtype(promote_type(typeof(a), typeof(b)), typeof(z))}(a,b)), z, n+1)

OrthogonalPolynomial(w::JacobiWeight) = Jacobi(w.a, w.b)
orthogonalityweight(P::Jacobi) = JacobiWeight(P.a, P.b)

const WeightedJacobi{T} = WeightedBasis{T,<:JacobiWeight,<:Jacobi}
WeightedJacobi(P::Jacobi{T}) where T = JacobiWeight(zero(T),zero(T)) .* P

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
==(A::WeightedJacobi, B::Jacobi{T,V}) where {T,V} = A == JacobiWeight(zero(V),zero(V)).*B
==(A::WeightedJacobi, B::Legendre) = A == Jacobi(B)
==(A::Legendre, B::WeightedJacobi) = Jacobi(A) == B
==(A::Jacobi{T,V}, B::WeightedJacobi) where {T,V} = JacobiWeight(zero(V),zero(V)).*A == B
==(A::Legendre, B::Weighted{<:Any,<:AbstractJacobi}) = A == B.P
==(A::Weighted{<:Any,<:AbstractJacobi}, B::Legendre) = A.P == B

show(io::IO, P::Jacobi) = summary(io, P)
summary(io::IO, P::Jacobi{Float64}) = print(io, "Jacobi($(P.a), $(P.b))")
summary(io::IO, P::Jacobi{T}) where T = print(io, "Jacobi{$T}($(P.a), $(P.b))")

###
# transforms
###

grid(P::AbstractJacobi{T}, n::Integer) where T = ChebyshevGrid{1,T}(n)
plotgrid(P::AbstractJacobi{T}, n::Integer) where T = ChebyshevGrid{2,T}(min(40n, MAX_PLOT_POINTS))

plan_transform(::AbstractJacobi{T}, szs::NTuple{N,Int}, dims...) where {T,N} = error("Override")
plan_transform(P::Jacobi{T}, szs::NTuple{N,Int}, dims...) where {T,N} = JacobiTransformPlan(FastTransforms.plan_th_cheb2jac!(T, szs, P.a, P.b, dims...), plan_chebyshevtransform(T, szs, dims...))

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

function _jacobi_convert_a(a, b, k, T) # Jacobi(a+k, b) \ Jacobi(a, b)
    j = round(k)
    if j ≉ k
        throw(ArgumentError("non-integer conversions not supported"))
    end
    k = Integer(j)
    reduce(*, [_jacobi_convert_a(a+j, b) for j in k-1:-1:0], init=Eye{T}(∞))
end
function _jacobi_convert_b(a, b, k, T) # Jacobi(a, b+k) \ Jacobi(a, b)
    j = round(k)
    if j ≉ k
        throw(ArgumentError("non-integer conversions not supported"))
    end
    k = Integer(j)
    reduce(*, [_jacobi_convert_b(a, b+j) for j in k-1:-1:0], init=Eye{T}(∞))
end

function \(A::Jacobi, B::Jacobi)
    T = promote_type(eltype(A), eltype(B))
    aa, ab = A.a, A.b
    ba, bb = B.a, B.b
    ka = aa - ba
    kb = ab - bb
    if ka >= 0
        C1 = _jacobi_convert_a(ba, ab, ka, T)
        if kb >= 0
            C2 = _jacobi_convert_b(ba, bb, kb, T)
        else
            C2 = inv(_jacobi_convert_b(ba, ab, -kb, T))
        end
        C1 * C2
    else
        inv(B \ A)
    end
end

\(A::Jacobi, w_B::WeightedJacobi) = WeightedJacobi(A) \ w_B
\(w_A::WeightedJacobi, B::Jacobi) = w_A \ WeightedJacobi(B)
\(A::AbstractJacobi, w_B::WeightedJacobi) = Jacobi(A) \ w_B
\(w_A::WeightedJacobi, B::AbstractJacobi) = w_A \ Jacobi(B)


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
broadcastbasis(::typeof(+), A::Jacobi, B::Weighted{<:Any,<:Jacobi{<:Any,<:Integer}}) = A
broadcastbasis(::typeof(+), A::Weighted{<:Any,<:Jacobi{<:Any,<:Integer}}, B::Jacobi) = B

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

function diff(S::Jacobi{T}, m::Integer; dims=1) where T
    D = _BandedMatrix((pochhammer.((S.a + S.b+1):∞, m)/convert(T, 2)^m)', ℵ₀, -m, m)
    ApplyQuasiMatrix(*, Jacobi{T}(S.a+m,S.b+m), D)
end


#L_6^t
function diff(WS::HalfWeighted{:a,T,<:Jacobi}; dims=1) where T
    S = WS.P
    a,b = S.a, S.b
    ApplyQuasiMatrix(*, HalfWeighted{:a}(Jacobi{T}(a-1,b+1)), Diagonal(-(a:∞)))
end

#L_6
function diff(WS::HalfWeighted{:b,T,<:Jacobi}; dims=1) where T
    S = WS.P
    a,b = S.a, S.b
    ApplyQuasiMatrix(*, HalfWeighted{:b}(Jacobi{T}(a+1,b-1)), Diagonal(b:∞))
end

for ab in (:(:a), :(:b))
    @eval function diff(WS::HalfWeighted{$ab,<:Any,<:Normalized}; dims=1)
        P,M = arguments(ApplyLayout{typeof(*)}(), WS.P)
        ApplyQuasiMatrix(*, diff(HalfWeighted{$ab}(P);dims=dims), M)
    end
end


function diff(WS::Weighted{T,<:Jacobi}; dims=1) where T
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
        ApplyQuasiMatrix(*, Weighted(Jacobi{T}(a-1, b-1)), _BandedMatrix((-2*(1:∞))', ℵ₀, 1,-1))
    end
end


# Jacobi(a-1,b-1)\ (D*w*Jacobi(a,b))
function diff(WS::WeightedJacobi{T}; dims=1) where T
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
        W = HalfWeighted{:b}(Jacobi{T}(a+1, b-1)) \ diff(HalfWeighted{:b}(S))
        J = Jacobi{T}(a+1,b) # range Jacobi
        C1 = J \ Jacobi{T}(a+1, b-1)
        C2 = J \ Jacobi{T}(a,b)
        ApplyQuasiMatrix(*, JacobiWeight(w.a,w.b-1) .* J, (w.b-b) * C2 + C1 * W)
    elseif iszero(w.b)
        W = HalfWeighted{:a}(Jacobi{T}(a-1, b+1)) \ diff(HalfWeighted{:a}(S))
        J = Jacobi{T}(a,b+1) # range Jacobi
        C1 = J \ Jacobi{T}(a-1, b+1)
        C2 = J \ Jacobi{T}(a,b)
        ApplyQuasiMatrix(*, JacobiWeight(w.a-1,w.b) .* J, -(w.a-a) * C2 + C1 * W)
    elseif iszero(a) && iszero(b) # Legendre
        # D * ((1+x)^w.b * (1-x)^w.a * P))
        #    == (1+x)^(w.b-1) * (1-x)^(w.a-1) * ((1-x) * (w.b) * P - (1+x) * w.a * P + (1-x^2) * D * P)
        #    == (1+x)^(w.b-1) * (1-x)^(w.a-1) * ((1-x) * (w.b) * P - (1+x) * w.a * P + P * L * W)
        J = Jacobi{T}(a+1,b+1) # range space
        W = J \ diff(S)
        X = jacobimatrix(S)
        L = S \ Weighted(J)
        (JacobiWeight(w.a-1,w.b-1) .* S) *  (((w.b-w.a)*I-(w.a+w.b) * X) + L*W)
    else
        # We differentiate
        # D * ((1+x)^w.b * (1-x)^w.a * P^(a,b)) == D * ((1+x)^(w.b-b) * (1-x)^(w.a-a)  * (1+x)^b * (1-x)^a * P^(a,b))
        #    == (1+x)^(w.b-1) * (1-x)^(w.a-1) * ((1-x) * (w.b-b) * P^(a,b) + (1+x) * (a-w.a) * P^(a,b))
        #        + (1+x)^(w.b-b) * (1-x)^(w.a-a) * D * ((1+x)^b * (1-x)^a * P^(a,b)))
        
        W = Weighted(Jacobi{T}(a-1,b-1)) \ diff(Weighted(S))
        X = jacobimatrix(S)
        C = S \ Jacobi{T}(a-1,b-1)
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


