struct LegendreWeight{T} <: AbstractJacobiWeight{T} end
LegendreWeight() = LegendreWeight{Float64}()
legendreweight(d::AbstractInterval{T}) where T = LegendreWeight{float(T)}()[affine(d,ChebyshevInterval{T}())]

function getindex(w::LegendreWeight{T}, x::Number) where T
    x ∈ axes(w,1) || throw(BoundsError())
    one(T)
end

getproperty(w::LegendreWeight{T}, ::Symbol) where T = zero(T)

sum(::LegendreWeight{T}) where T = 2one(T)



broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(*), ::LegendreWeight{T}, ::LegendreWeight{V}) where {T,V} =
    LegendreWeight{promote_type(T,V)}()

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(sqrt), w::LegendreWeight{T}) where T = w

broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, w::LegendreWeight, ::Base.RefValue{Val{k}}) where k = w

singularitiesbroadcast(_, L::LegendreWeight) = L # Assume we stay smooth
singularitiesbroadcast(::typeof(exp), L::LegendreWeight) = L
singularitiesbroadcast(::typeof(Base.literal_pow), ::typeof(^), L::LegendreWeight, ::Val) = L

for op in (:+, :-, :*)
    @eval begin
        singularitiesbroadcast(::typeof($op), A::LegendreWeight, B::LegendreWeight) = LegendreWeight{promote_type(eltype(A), eltype(B))}()
        singularitiesbroadcast(::typeof($op), L::LegendreWeight, ::NoSingularities) = L
        singularitiesbroadcast(::typeof($op), ::NoSingularities, L::LegendreWeight) = L
    end
end
singularitiesbroadcast(::typeof(^), L::LegendreWeight, ::NoSingularities) = L
singularitiesbroadcast(::typeof(/), ::NoSingularities, L::LegendreWeight) = L # can't find roots

singularities(::AbstractJacobi{T}) where T = LegendreWeight{T}()
singularities(::Inclusion{T,<:AbstractInterval}) where T = LegendreWeight{T}()
singularities(d::Inclusion{T,<:Interval}) where T = LegendreWeight{T}()[affine(d,ChebyshevInterval{T}())]
singularities(::AbstractFillLayout, P) = LegendreWeight{eltype(P)}()

struct Legendre{T} <: AbstractJacobi{T} end
Legendre() = Legendre{Float64}()

weighted(P::Legendre) = P
weighted(P::Normalized{<:Any,<:Legendre}) = P
weighted(P::SubQuasiArray{<:Any,2,<:Legendre}) = P
weighted(P::SubQuasiArray{<:Any,2,<:Normalized{<:Any,<:Legendre}}) = P

legendre() = Legendre()
legendre(d::AbstractInterval{T}) where T = Legendre{float(T)}()[affine(d,ChebyshevInterval{T}()), :]

"""
     legendrep(n, z)

computes the `n`-th Legendre polynomial at `z`.
"""
legendrep(n::Integer, z::Number) = Base.unsafe_getindex(Legendre{typeof(z)}(), z, n+1)


summary(io::IO, ::Legendre) = print(io, "Legendre()")

==(::Legendre, ::Legendre) = true

OrthogonalPolynomial(w::LegendreWeight{T}) where {T} = Legendre{T}()
orthogonalityweight(::Legendre{T}) where T = LegendreWeight{T}()

function qr(P::Legendre)
    Q = Normalized(P)
    QuasiQR(Q, Diagonal(Q.scaling))
end


function ldiv(P::Legendre{V}, f::Inclusion{T}) where {T,V}
    axes(P,1) == f || throw(DimensionMismatch())
    TV = promote_type(T,V)
    Vcat(zero(TV), one(TV), Zeros{TV}(∞))
end


ldiv(P::Legendre{V}, f::AbstractQuasiFill{T,1}) where {T,V} = _op_ldiv(P, f)
function transform_ldiv(::Legendre{V}, f::Union{AbstractQuasiVector,AbstractQuasiMatrix}) where V
    T = ChebyshevT{V}()
    dat = T \ f
    pad(cheb2leg(paddeddata(dat)), axes(dat)...)
end

struct LegendreTransformPlan{T, CHEB2LEG, DCT} <: Plan{T}
    cheb2leg::CHEB2LEG
    chebtransform::DCT
end

LegendreTransformPlan(c2l, ct) = LegendreTransformPlan{promote_type(eltype(c2l),eltype(ct)),typeof(c2l),typeof(ct)}(c2l, ct)

*(P::LegendreTransformPlan, x::AbstractArray) = P.cheb2leg*(P.chebtransform*x)

function plan_grid_transform(P::Legendre{T}, szs::NTuple{N,Int}, dims=1:N) where {T,N}
    arr = Array{T}(undef, szs...)
    x = grid(P, size(arr,1))
    x, LegendreTransformPlan(FastTransforms.plan_th_cheb2leg!(arr, dims), plan_chebyshevtransform(arr, dims))
end


"""
    legendre_massmatrix

computes the massmatrix by first re-expanding in Legendre
"""
function legendre_massmatrix(Ac, B)
    A = parent(Ac)
    P = Legendre{eltype(B)}()
    (P\A)'*massmatrix(P)*(P\B)
end

@simplify *(Ac::QuasiAdjoint{<:Any,<:Legendre}, B::Legendre) = massmatrix(Legendre{promote_type(eltype(Ac), eltype(B))}())

# massmatrix(P) = Weighted(P)'P
massmatrix(P::Legendre{T}) where T = Diagonal(convert(T,2) ./ (2(0:∞) .+ 1))

########
# Jacobi Matrix
########

jacobimatrix(::Legendre{T}) where T =  Tridiagonal((one(T):∞)./(1:2:∞), Zeros{T}(∞), (one(T):∞)./(3:2:∞))

# These return vectors A[k], B[k], C[k] are from DLMF. Cause of MikaelSlevinsky we need an extra entry in C ... for now.
function recurrencecoefficients(::Legendre{T}) where T
    n = zero(T):∞
    ((2n .+ 1) ./ (n .+ 1), Zeros{T}(∞), n ./ (n .+ 1))
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


###
# dot
###

_dot(::Inclusion{<:Any,<:ChebyshevInterval}, a, b) = _dot(singularities(a), singularities(b), a, b)
function _dot(::LegendreWeight, ::LegendreWeight, a, b)
    P = Legendre{promote_type(eltype(a),eltype(b))}()
    dot(P\a, (massmatrix(P) * (P\b)))
end