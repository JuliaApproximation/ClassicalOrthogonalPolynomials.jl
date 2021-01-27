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
const PowKernelPoint{T,V,D,F} =  BroadcastQuasiVector{T, typeof(^), Tuple{ContinuumArrays.AffineQuasiVector{T, V, Inclusion{V, D}, T}, F}}

struct PowerLawMatrix{T} #<: AbstractMatrix{T} # TODO
    a::T
    t::T
end

############
# METHODS
############
function dot(f::AbstractVector{T},K::PowKernelPoint{Float64,Float64,ChebyshevInterval{Float64}},g::AbstractVector{T}) where T
    (lf, lg) = (length(f),length(g))
    a = K.args[2]
    t = (K.args[1])[0.] # there must be something better than this? 
                        # maybe something in the spirit of (K.args[1]).args[1]?
    t<1 && error("t must be greater than 1.")
    (lf<∞) && (lg<∞) && return pointwisedot(f,g,a,t)
    ((lf<∞) || (lg<∞)) && error("TODO: Currently only both finite or both infinite.")
    # seek naive convergence for infinite input.
    (i,conv1,conv2) = (0,1,2)
    while abs(conv1-conv2)>1e-12
        i = i+1
        conv1 = pointwisedot(f[1:i*20],g[1:i*20],a,t)
        conv2 = pointwisedot(f[1:2*i*20],g[1:2*i*20],a,t)
    end
    return conv2
end
function dot(f::AbstractVector,M::PowerLawMatrix,g::AbstractVector)
    return dot(f,(M.t .- axes(Legendre(),1)).^M.a,g)
end
function *(P::Legendre,M::PowerLawMatrix)
    return (M.t .- axes(P,1)).^M.a.*P
end
function *(g::Adjoint,K::PowKernelPoint{<:Any,<:Any,<:ChebyshevInterval},f::AbstractVector)
    return dot(g',K,f)
end

############
# EVALUATE RECURRENCE
############
# experimental pointwise power law integral of Legendre product by recurrence
function pointwisedot(f::AbstractVector{T},g::AbstractVector{T},a::Real,t::Real) where T
    # initialization
    ℓ = max(length(f),length(g))
    f = pad(f,ℓ)
    g = pad(g,ℓ)
    coeff = zeros(ℓ,ℓ)

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
    for m = 2:ℓ-2
        # first row
        coeff[1,m+2] = (t/((a+m+2)/(2*m+1))*coeff[1,m+1]+((a-m+1)/(2*m+1))/((a+m+2)/(2*m+1))*coeff[1,m])
        # second row
        coeff[2,m+2] = (t/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*coeff[2,m+1]+((a+1)/(2-m*(m+1)))/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*coeff[1,m+1]-(m*(a+2-m)/((2*m+1)*(1-m)))*1/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*coeff[2,m])
        # build remaining row elements
        for j=1:m-1
            n = j
            coeff[j+2,m+1] = (t/((n+1)*(a+m+n+2)/((2*n+1)*(m+n+1)))*coeff[n+1,m+1]+((a+1)*m/(m*(m+1)-n*(n+1)))/((n+1)*(a+m+n+2)/((2*n+1)*(m+n+1)))*coeff[n+1,m]-(n*(a+m-n+1)/((2*n+1)*(m-n)))*1/((n+1)*(a+m+n+2)/((2*n+1)*(m+n+1)))*coeff[n,m+1])
        end
    end
    # apply the coefficients
    for n = 1:ℓ
        coeff[n,n] = f[n]*g[n]*coeff[n,n]
        for m=1:n-1
            coeff[m,n] = (f[m]*g[n]+f[n]*g[m])*coeff[m,n]
        end
    end
    return sum(coeff)
end

############
# HELPERS
############
# these explicit initial cases are needed to kick off the recurrence
function PLinitial00(t::Real,a::Real)
    return ((t+1)^(a+1)-(t-1)^(a+1))/(a+1)
end
function PLinitial01(t::Real,a::Real)
    return ((t+1)^(a+1)*(-a+t-1)-(a+t+1)*(t-1)^(a+1))/((a+1)*(a+2))
end
function PLinitial11(t::Real,a::Real)
    return ((t+1)^(a+1)*(a^2+a*(3-2*t)+2*(t-1)*t+2)-(t-1)^(a+1)*(a^2+a*(2*t+3)+2*(t^2+t+1)))/((a+1)*(a+2)*(a+3))
end
function PLinitial12(t::Real,a::Real)
    return -(((1+t)^(1+a)*((1+a)^2*(3+a)-(3+2*a*(5+2*a))*t+9*(1+a)*t^2-9*t^3)+(-1+t)^(1+a)*((1+a)^2*(3+a)+(3+2*a*(5+2*a))*t+9*(1+a)*t^2+9*t^3))/((1+a)*(2+a)*(3+a)*(4+a)))
end

# pad helper function from ApproxFun
function pad(f::AbstractVector{T},n::Integer) where T
	if n > length(f)
	   ret=Vector{T}(undef, n)
	   ret[1:length(f)]=f
	   for j=length(f)+1:n
	       ret[j]=zero(T)
	   end
       ret
	else
        f[1:n]
	end
end