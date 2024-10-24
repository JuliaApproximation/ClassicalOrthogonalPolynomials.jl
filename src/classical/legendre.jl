struct LegendreWeight{T} <: AbstractJacobiWeight{T} end
LegendreWeight() = LegendreWeight{Float64}()
legendreweight(d::AbstractInterval{T}) where T = LegendreWeight{float(T)}()[affine(d,ChebyshevInterval{T}())]

AbstractQuasiArray{T}(::LegendreWeight) where T = LegendreWeight{T}()
AbstractQuasiVector{T}(::LegendreWeight) where T = LegendreWeight{T}()


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


singularities(::AbstractFillLayout, P) = LegendreWeight{eltype(P)}()
singularities(::Legendre) = NoSingularities()
basis_singularities(ax::Inclusion, ::NoSingularities) = legendre(ax)

struct Legendre{T} <: AbstractJacobi{T} end
Legendre() = Legendre{Float64}()

AbstractQuasiArray{T}(::Legendre) where T = Legendre{T}()
AbstractQuasiMatrix{T}(::Legendre) where T = Legendre{T}()

weighted(P::Legendre) = P
weighted(P::Normalized{<:Any,<:Legendre}) = P
weighted(P::SubQuasiArray{<:Any,2,<:Legendre}) = P
weighted(P::SubQuasiArray{<:Any,2,<:Normalized{<:Any,<:Legendre}}) = P

legendre() = Legendre()
legendre(d::AbstractInterval{T}) where T = Legendre{float(T)}()[affine(d,ChebyshevInterval{T}()), :]
legendre(d::ChebyshevInterval{T}) where T = Legendre{float(T)}()
legendre(d::Inclusion) = legendre(d.domain)

"""
     legendrep(n, z)

computes the `n`-th Legendre polynomial at `z`.
"""
legendrep(n::Integer, z::Number) = Base.unsafe_getindex(Legendre{typeof(z)}(), z, n+1)


show(io::IO, w::Legendre{Float64}) = summary(io, w)
summary(io::IO, ::Legendre{Float64}) = print(io, "Legendre()")

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
    dat = transform_ldiv(T, f)
    pad(th_cheb2leg(paddeddata(dat)), axes(dat)...)
end

plan_transform(::Legendre{T}, szs::NTuple{N,Int}, dims...) where {T,N} = JacobiTransformPlan(FastTransforms.plan_th_cheb2leg!(T, szs, dims...), plan_chebyshevtransform(T, szs, dims...))


"""
    legendre_grammatrix

computes the grammatrix by first re-expanding in Legendre
"""
function legendre_grammatrix(A, B)
    P = Legendre{eltype(B)}()
    (P\A)'*grammatrix(P)*(P\B)
end

function legendre_grammatrix(A)
    P = Legendre{eltype(A)}()
    R = P\A
    R' * grammatrix(P) * R
end

grammatrix(P::Legendre{T}) where T = Diagonal(convert(T,2) ./ (2(0:∞) .+ 1))
grammatrix(P::Normalized{T,<:Legendre}) where T = Eye{T}(∞)
@simplify *(P::QuasiAdjoint{<:Any,<:Normalized{<:Any,<:Legendre}}, Q::Normalized{<:Any,<:Legendre}) =
    grammatrix(Normalized(Legendre{promote_type(eltype(P), eltype(Q))}()))

########
# Jacobi Matrix
########

jacobimatrix(::Legendre{T}) where T =  Tridiagonal((one(T):∞)./(1:2:∞), Zeros{T}(∞), (one(T):∞)./(3:2:∞))

# These return vectors A[k], B[k], C[k] are from DLMF. Cause of MikaelSlevinsky we need an extra entry in C ... for now.
function recurrencecoefficients(::Legendre{T}) where T
    n = zero(real(T)):∞
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
