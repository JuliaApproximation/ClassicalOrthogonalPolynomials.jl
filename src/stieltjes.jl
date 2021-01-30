const StieltjesPoint{T,V,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{BroadcastQuasiMatrix{T,typeof(-),Tuple{T,QuasiAdjoint{V,Inclusion{V,D}}}}}}
const ConvKernel{T,D} = BroadcastQuasiMatrix{T,typeof(-),Tuple{D,QuasiAdjoint{T,D}}}
const Hilbert{T,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{ConvKernel{T,Inclusion{T,D}}}}
const LogKernel{T,D} = BroadcastQuasiMatrix{T,typeof(log),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D}}}}}}
const PowKernel{T,D,F<:Real} = BroadcastQuasiMatrix{T,typeof(^),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D}}}},F}}


@simplify function *(S::StieltjesPoint{<:Any,<:Any,<:ChebyshevInterval}, wT::WeightedBasis{<:Any,<:ChebyshevTWeight,<:ChebyshevT})
    w,T = wT.args
    J = jacobimatrix(T)
    z, x = parent(S).args[1].args
    transpose((J'-z*I) \ [-π; zeros(∞)])
end

@simplify function *(H::Hilbert{<:Any,<:ChebyshevInterval}, wT::WeightedBasis{<:Any,<:ChebyshevTWeight,<:ChebyshevT}) 
    T = promote_type(eltype(H), eltype(wT))
    ApplyQuasiArray(*, ChebyshevU{T}(), _BandedMatrix(Fill(-convert(T,π),1,∞), ∞, -1, 1))
end

@simplify function *(H::Hilbert{<:Any,<:ChebyshevInterval}, wU::WeightedBasis{<:Any,<:ChebyshevUWeight,<:ChebyshevU}) 
    T = promote_type(eltype(H), eltype(wU))
    ApplyQuasiArray(*, ChebyshevT{T}(), _BandedMatrix(Fill(convert(T,π),1,∞), ∞, 1, -1))
end

### 
# LogKernel
###

@simplify function *(L::LogKernel{<:Any,<:ChebyshevInterval}, wT::WeightedBasis{<:Any,<:ChebyshevTWeight,<:ChebyshevT}) 
    T = promote_type(eltype(L), eltype(wT))
    ApplyQuasiArray(*, ChebyshevT{T}(), Diagonal(Vcat(-π*log(2*one(T)),-convert(T,π)./(1:∞))))
end


### 
# PowKernel
###

@simplify function *(K::PowKernel{<:Any,<:ChebyshevInterval}, wT::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}) 
    T = promote_type(eltype(K), eltype(wT))
    cnv,α = K.args
    x,y = K.args[1].args[1].args
    @assert x' == y
    β = (-α-1)/2
    ApplyQuasiArray(*, ChebyshevT{T}(), Diagonal(1:∞))
end


####
# StieltjesPoint
####

@simplify function *(S::StieltjesPoint, wT::SubQuasiArray{<:Any,2,<:WeightedBasis,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    P = parent(wT)
    z, x = parent(S).args[1].args
    z̃ = inbounds_getindex(parentindices(wT)[1], z)
    x̃ = axes(P,1)
    (inv.(z̃ .- x̃') * P)[:,parentindices(wT)[2]]
end

@simplify function *(H::Hilbert, wT::SubQuasiArray{<:Any,2,<:WeightedBasis,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}) 
    P = parent(wT)
    x = axes(P,1)
    apply(*, inv.(x .- x'), P)[parentindices(wT)...]
end


@simplify function *(L::LogKernel, wT::SubQuasiArray{<:Any,2,<:WeightedBasis,<:Tuple{<:AbstractAffineQuasiVector,<:Slice}}) 
    V = promote_type(eltype(L), eltype(wT))
    P = parent(wT)
    kr, jr = parentindices(wT)
    @assert P isa WeightedBasis{<:Any,<:ChebyshevWeight,<:Chebyshev}
    x = axes(P,1)
    w,T = P.args
    D = T \ apply(*, log.(abs.(x .- x')), P)
    c = inv(2*kr.A)
    T[kr,:] * Diagonal(Vcat(2*convert(V,π)*c*log(c), 2c*D.diag.args[2]))
end

#################################################
# ∫f(x)g(x)(t-x)^a dx evaluation where f and g in Legendre
#################################################
const PowKernelPoint{T, V, D, F} = BroadcastQuasiVector{T, typeof(^), Tuple{ContinuumArrays.AffineQuasiVector{T, V, Inclusion{V, D}, T}, F}}

############
# METHODS
############

# This will need re-adjustments now that the cached version exists, so I'm commenting it out for now

# function dot(f::AbstractVector{T}, K::PowKernelPoint, g::AbstractVector{T}) where T
#     (lf, lg) = (length(f),length(g))
#     a = K.args[2]
#     t = (K.args[1])[0.] # there must be something better than this? 
#                         # maybe something in the spirit of (K.args[1]).args[1]?
#     t<1 && error("t must be greater than 1.")
#     (lf<∞) && (lg<∞) && return pointwisedot(f,g,a,t)
#     ((lf<∞) || (lg<∞)) && error("TODO: Currently only both finite or both infinite.")
#     # for now, seek naive convergence for infinite input.
#     (i,conv1,conv2) = (0,1,2)
#     while abs(conv1-conv2)>1e-15
#         i = i+1
#         conv1 = pointwisedot(f[1:i*20],g[1:i*20],a,t)
#         conv2 = pointwisedot(f[1:2*i*20],g[1:2*i*20],a,t)
#     end
#     return conv2
# end
# function *(g::Adjoint, K::PowKernelPoint{<:Any,<:Any,<:ChebyshevInterval}, f::AbstractVector)
#     return dot(g',K,f)
# end


############
# IMPLEMENT CACHED VERSION
############
# Constructors support BigFloat and it's recommended to use them for high orders.
mutable struct PowerLawIntegral{T, PP<:AbstractQuasiMatrix} <: AbstractCachedMatrix{T}
    P::PP # OPs - only Legendre supported for now
    a::T  # naming scheme follows (t-x)^a
    t::T
    data::Matrix{T}
    datasize::Tuple{Int,Int}
    array

    function PowerLawIntegral{T, PP}(P::PP, a::T, t::T) where {T, PP<:AbstractQuasiMatrix}
        new{T, PP}(P,a,t, pointwisecoeffmatrixdense(a,t,10),(10,10))
    end
end
PowerLawIntegral(P::AbstractQuasiMatrix, a::T, t::T) where T = PowerLawIntegral{T,typeof(P)}(P,a,t)
size(K::PowerLawIntegral) = (∞,∞) # potential to add maximum size of operator

# data filling
function _legendrepowerlaw_fill_data!(K::PowerLawIntegral, inds)
    fillcoeffmatrix!(K, inds)
end
cache_filldata!(K::PowerLawIntegral, inds) = _legendrepowerlaw_fill_data!(K, inds)

# because it really only makes sense to compute this symmetric operator in square blocks, we have to slightly rework some of LazyArrays caching and resizing
function Base.getindex(A::PowerLawIntegral{T, PP}, I::CartesianIndex) where {T,PP<:AbstractQuasiMatrix}
    resizedata!(A, Tuple(I))
    A.data[I]
end
function getindex(A::PowerLawIntegral{T,PP}, I::Vararg{Int,2}) where {T,PP<:AbstractQuasiMatrix}
    # @boundscheck checkbounds(A, I...)
    resizedata!(A, Tuple([I...]))
    A.data[I...]
end
function resizedata!(A::PowerLawIntegral, nm) 
    olddata = A.data
    νμ = size(olddata)
    nm = (maximum(nm),maximum(nm))
    nm = max.(νμ,nm)
    nm = (maximum(nm),maximum(nm))
    if νμ ≠ nm
        A.data = similar(A.data, nm...)
        A.data[axes(olddata)...] = olddata
    end
    inds = Array(maximum(νμ):maximum(nm))
    cache_filldata!(A, inds)
    A.datasize = nm
    A
end

############
# RECURRENCE EVALUATION
############
# this function actually evaluates the recurrence and returns the full operator. 
# We don't use this outside of the initial block.
function pointwisecoeffmatrixdense(a::Real, t::Real, ℓ::Integer)
    # initialization
    ℓ = ℓ+1
    coeff = convert.(typeof(a),zeros(ℓ,ℓ))

    # load in explicit initial cases
    coeff[1,1] = PLinitial00(t,a)
    coeff[1,2] = PLinitial01(t,a)
    coeff[2,2] = PLinitial11(t,a)
    coeff[2,3] = PLinitial12(t,a)

    # we have to build these two cases with some care
    coeff[1,3] = t/((a+3)/3)*coeff[1,2]+(a/3)/((a+3)/3)*coeff[1,1]
    m=1
    coeff[3,m+2] = t/((m+1)*(a+m+4)/((2*m+1)*(m+3)))*coeff[m+1,3]+((a+1)*2/(6-m*(m+1)))/((m+1)*(a+m+4)/((2*m+1)*(m+3)))*coeff[2,m+1]-(m*(a+3-m)/((2*m+1)*(2-m)))*1/((m+1)*(a+m+4)/((2*m+1)*(m+3)))*coeff[m,3]

    # the remaining cases can be constructed iteratively
    @inbounds for m = 2:ℓ-2
        # first row
        coeff[1,m+2] = (t/((a+m+2)/(2*m+1))*coeff[1,m+1]+((a-m+1)/(2*m+1))/((a+m+2)/(2*m+1))*coeff[1,m])
        # second row
        coeff[2,m+2] = (t/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*coeff[2,m+1]+((a+1)/(2-m*(m+1)))/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*coeff[1,m+1]-(m*(a+2-m)/((2*m+1)*(1-m)))*1/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*coeff[2,m])
        # build remaining row elements
        @inbounds for j=1:m-1
            n = j
            coeff[j+2,m+1] = (t/((n+1)*(a+m+n+2)/((2*n+1)*(m+n+1)))*coeff[n+1,m+1]+((a+1)*m/(m*(m+1)-n*(n+1)))/((n+1)*(a+m+n+2)/((2*n+1)*(m+n+1)))*coeff[n+1,m]-(n*(a+m-n+1)/((2*n+1)*(m-n)))*1/((n+1)*(a+m+n+2)/((2*n+1)*(m+n+1)))*coeff[n,m+1])
        end
    end
    @inbounds for m=1:ℓ
        @inbounds for n=m+1:ℓ
            coeff[n,m] = coeff[m,n]
        end
    end
    return coeff[1:ℓ-1,1:ℓ-1]
end
# these explicit initial cases are needed to kick off the recurrence
function PLinitial00(t, a)
    return ((t+1)^(a+1)-(t-1)^(a+1))/(a+1)
end
function PLinitial01(t, a)
    return ((t+1)^(a+1)*(-a+t-1)-(a+t+1)*(t-1)^(a+1))/((a+1)*(a+2))
end
function PLinitial11(t, a)
    return ((t+1)^(a+1)*(a^2+a*(3-2*t)+2*(t-1)*t+2)-(t-1)^(a+1)*(a^2+a*(2*t+3)+2*(t^2+t+1)))/((a+1)*(a+2)*(a+3))
end
function PLinitial12(t, a)
    return -(((1+t)^(1+a)*((1+a)^2*(3+a)-(3+2*a*(5+2*a))*t+9*(1+a)*t^2-9*t^3)+(-1+t)^(1+a)*((1+a)^2*(3+a)+(3+2*a*(5+2*a))*t+9*(1+a)*t^2+9*t^3))/((1+a)*(2+a)*(3+a)*(4+a)))
end
# the following version takes a previously computed block that's been resized and fills in the missing data guided by indices in inds
function fillcoeffmatrix!(K, inds)
    # the remaining cases can be constructed iteratively
    a = K.a
    t = K.t
    @inbounds for m in inds
        m=m-2
        # first row
        K.data[1,m+2] = (t/((a+m+2)/(2*m+1))*K.data[1,m+1]+((a-m+1)/(2*m+1))/((a+m+2)/(2*m+1))*K.data[1,m])
        # second row
        K.data[2,m+2] = (t/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*K.data[2,m+1]+((a+1)/(2-m*(m+1)))/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*K.data[1,m+1]-(m*(a+2-m)/((2*m+1)*(1-m)))*1/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*K.data[2,m])
        # build remaining row elements
        @inbounds for j=1:m
            n = j
            K.data[j+2,m+2] = (t/((n+1)*(a+m+1+n+2)/((2*n+1)*(m+1+n+1)))*K.data[n+1,m+2]+((a+1)*(m+1)/((m+1)*(m+2)-n*(n+1)))/((n+1)*(a+m+1+n+2)/((2*n+1)*(m+1+n+1)))*K.data[n+1,m+1]-(n*(a+m+1-n+1)/((2*n+1)*(m+1-n)))*1/((n+1)*(a+m+1+n+2)/((2*n+1)*(m+1+n+1)))*K.data[n,m+2])
        end
    end
    # matrix is symmetric
    @inbounds for m in reverse(inds)
        K.data[m,1:end] = K.data[1:end,m]
    end
end