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
# ∫f(x)g(x)(t-x)^a dx evaluation where f and g given in coefficients
#################################################
# recognize structure of W = ((t .- x).^a
const PowKernelPoint{T,V,D,F} =  BroadcastQuasiVector{T, typeof(^), Tuple{ContinuumArrays.AffineQuasiVector{T, V, Inclusion{V, D}, T}, F}}

####
# cached operator implementation
####
# Constructors support BigFloat and it's recommended to use them for high orders.
mutable struct PowerLawMatrix{T, PP<:Normalized{<:Any,<:Legendre{<:Any}}} <: AbstractCachedMatrix{T}
    P::PP # OPs - only normalized Legendre supported for now
    a::T  # naming scheme follows (t-x)^a
    t::T
    data::Matrix{T}
    datasize::Tuple{Int,Int}
    array
    function PowerLawMatrix{T, PP}(P::PP, a::T, t::T) where {T, PP<:AbstractQuasiMatrix}
        new{T, PP}(P,a,t, gennormalizedpower(a,t,10),(10,10))
    end
end
PowerLawMatrix(P::AbstractQuasiMatrix, a::T, t::T) where T = PowerLawMatrix{T,typeof(P)}(P,a,t)
size(K::PowerLawMatrix) = (∞,∞) # potential to add maximum size of operator

# data filling
cache_filldata!(K::PowerLawMatrix, inds) = fillcoeffmatrix!(K, inds)

# because it really only makes sense to compute this symmetric operator in square blocks, we have to slightly rework some of LazyArrays caching and resizing
function getindex(K::PowerLawMatrix{T, PP}, I::CartesianIndex) where {T,PP<:AbstractQuasiMatrix}
    resizedata!(K, Tuple(I))
    K.data[I]
end
function getindex(K::PowerLawMatrix{T,PP}, I::Vararg{Int,2}) where {T,PP<:AbstractQuasiMatrix}
    resizedata!(K, Tuple([I...]))
    K.data[I...]
end
function resizedata!(K::PowerLawMatrix, nm) 
    olddata = K.data
    νμ = size(olddata)
    nm = (maximum(nm),maximum(nm))
    nm = max.(νμ,nm)
    nm = (maximum(nm),maximum(nm))
    if νμ ≠ nm
        K.data = similar(K.data, nm...)
        K.data[axes(olddata)...] = olddata
    end
    if maximum(nm) > maximum(νμ)
        inds = Array(maximum(νμ):maximum(nm))
        cache_filldata!(K, inds)
        K.datasize = nm
    end
    K
end

####
# methods
####
function *(K::PowKernelPoint,Q::Normalized{<:Any,<:Legendre{<:Any}})
    t = K.args[1][0.]
    a = K.args[2]
    return Q*PowerLawMatrix(Q,a,t)
end

####
# recurrence evaluation
####
# this function evaluates the recurrence and returns the full operator. 
# We don't use this outside of the initial block.
function gennormalizedpower(a::Real, t::Real, ℓ::Integer)
    # initialization
    ℓ = ℓ+3
    coeff = convert.(typeof(a),zeros(ℓ,ℓ))
    # load in explicit initial cases
    coeff[1,1] = PLnorminitial00(t,a)
    coeff[1,2] = PLnorminitial01(t,a)
    coeff[2,2] = PLnorminitial11(t,a)
    coeff[2,3] = PLnorminitial12(t,a)
    # we have to build these two cases with some care
    coeff[1,3] = t/((a+3)/3)*normconst_Pnadd1(1)*coeff[1,2]+(a/3)/((a+3)/3)*normconst_Pnsub1(1)*coeff[1,1]
    #the remaining cases can be constructed iteratively
    @inbounds for m = 2:ℓ-2
        # first row
        coeff[1,m+2] = (t/((a+m+2)/(2*m+1))*normconst_Pnadd1(m)*coeff[1,m+1]+((a-m+1)/(2*m+1))/((a+m+2)/(2*m+1))*normconst_Pnsub1(m)*coeff[1,m])
        # second row
        coeff[2,m+2] = (t/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pnadd1(m)*coeff[2,m+1]+((a+1)/(2-m*(m+1)))/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pmnmix(1,m)*coeff[1,m+1]-(m*(a+2-m)/((2*m+1)*(1-m)))*1/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pnsub1(m)*coeff[2,m])
        # build remaining row elements
        @inbounds for j=1:m-1
            coeff[j+2,m+1] = (t/((j+1)*(a+m+j+2)/((2*j+1)*(m+j+1)))*normconst_Pnadd1(j)*coeff[j+1,m+1]+((a+1)*m/(m*(m+1)-j*(j+1)))/((j+1)*(a+m+j+2)/((2*j+1)*(m+j+1)))*normconst_Pmnmix(m,j)*coeff[j+1,m]-(j*(a+m-j+1)/((2*j+1)*(m-j)))*1/((j+1)*(a+m+j+2)/((2*j+1)*(m+j+1)))*normconst_Pnsub1(j)*coeff[j,m+1])
        end
    end
    #matrix is symmetric
    @inbounds for m=1:ℓ
        @inbounds for n=m+1:ℓ
            coeff[n,m] = coeff[m,n]
        end
    end
    return coeff[1:ℓ-3,1:ℓ-3]
end

# modify recurrence coefficients to work for normalized Legendre
# TODO: should probably use whatever type is being asked for instead of defaulting to BigFloat
normconst_Pnadd1(m) = sqrt(2*BigInt(m)+3)/sqrt(2*BigInt(m)+1)
normconst_Pnsub1(m) = sqrt(2*BigInt(m)+3)/sqrt(2*BigInt(m)-1)
normconst_Pmnmix(n,m) = sqrt(2*BigInt(m)+3)*sqrt(2*BigInt(n)+1)/(sqrt(2*BigInt(m)+1)*sqrt(2*BigInt(n)-1))
# these explicit initial cases are needed to kick off the recurrence
function PLnorminitial00(t, a)
    return ((t+1)^(a+1)-(t-1)^(a+1))/(2*(a+1))
end
function PLnorminitial01(t, a)
    return sqrt(BigInt(3))*((t+1)^(a+1)*(-a+t-1)-(a+t+1)*(t-1)^(a+1))/(2*(a+1)*(a+2))
end
function PLnorminitial11(t, a)
    return BigInt(3)*((t+1)^(a+1)*(a^2+a*(3-2*t)+2*(t-1)*t+2)-(t-1)^(a+1)*(a^2+a*(2*t+3)+2*(t^2+t+1)))/(2*(a+1)*(a+2)*(a+3))
end
function PLnorminitial12(t, a)
    return -sqrt(BigInt(15))*(((1+t)^(1+a)*((1+a)^2*(3+a)-(3+2*a*(5+2*a))*t+9*(1+a)*t^2-9*t^3)+(-1+t)^(1+a)*((1+a)^2*(3+a)+(3+2*a*(5+2*a))*t+9*(1+a)*t^2+9*t^3))/(2*(1+a)*(2+a)*(3+a)*(4+a)))
end

# the following version takes a previously computed block that has been resized and fills in the missing data guided by indices in inds
function fillcoeffmatrix!(K, inds)
    # the remaining cases can be constructed iteratively
    a = K.a
    t = K.t
    @inbounds for m in inds
        m=m-2
        # first row
        K.data[1,m+2] = (t/((a+m+2)/(2*m+1))*normconst_Pnadd1(m)*K.data[1,m+1]+((a-m+1)/(2*m+1))/((a+m+2)/(2*m+1))*normconst_Pnsub1(m)*K.data[1,m])
        # second row
        K.data[2,m+2] = (t/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pnadd1(m)*K.data[2,m+1]+((a+1)/(2-m*(m+1)))/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pmnmix(1,m)*K.data[1,m+1]-(m*(a+2-m)/((2*m+1)*(1-m)))*1/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pnsub1(m)*K.data[2,m])
        # build remaining row elements
        @inbounds for j=1:m
            K.data[j+2,m+2] = (t/((j+1)*(a+m+1+j+2)/((2*j+1)*(m+1+j+1)))*normconst_Pnadd1(j)*K.data[j+1,m+2]+((a+1)*(m+1)/((m+1)*(m+2)-j*(j+1)))/((j+1)*(a+m+1+j+2)/((2*j+1)*(m+j+2)))*normconst_Pmnmix(m+1,j)*K.data[j+1,m+1]-(j*(a+m+1-j+1)/((2*j+1)*(m+1-j)))*1/((j+1)*(a+m+1+j+2)/((2*j+1)*(m+1+j+1)))*normconst_Pnsub1(j)*K.data[j,m+2])
        end
    end
    # matrix is symmetric
    @inbounds for m in reverse(inds)
        K.data[m,1:end] = K.data[1:end,m]
    end
end