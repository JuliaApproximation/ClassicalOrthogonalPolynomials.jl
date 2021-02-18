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
    ApplyQuasiArray(*, ChebyshevU{T}(), _BandedMatrix(Fill(-convert(T,π),1,∞), ℵ₀, -1, 1))
end

@simplify function *(H::Hilbert{<:Any,<:ChebyshevInterval}, wU::WeightedBasis{<:Any,<:ChebyshevUWeight,<:ChebyshevU}) 
    T = promote_type(eltype(H), eltype(wU))
    ApplyQuasiArray(*, ChebyshevT{T}(), _BandedMatrix(Fill(convert(T,π),1,∞), ℵ₀, 1, -1))
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