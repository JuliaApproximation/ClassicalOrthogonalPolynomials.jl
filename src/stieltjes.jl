####
# Associated
####


"""
    AssociatedWeighted(P)

We normalise so that `orthogonalityweight(::Associated)` is a probability measure.
"""
struct AssociatedWeight{T,OPs<:AbstractQuasiMatrix{T}} <: Weight{T}
    P::OPs
end
axes(w::AssociatedWeight) = (axes(w.P,1),)

sum(::AssociatedWeight{T}) where T = one(T)

"""
    Associated(P)

constructs the associated orthogonal polynomials for P, which have the Jacobi matrix

    jacobimatrix(P)[2:end,2:end]

and constant first term. Or alternatively

    w = orthogonalityweight(P)
    A = recurrencecoefficients(P)[1]
    Associated(P) == (w/(sum(w)*A[1]))'*((P[:,2:end]' - P[:,2:end]) ./ (x' - x))

where `x = axes(P,1)`.
"""

struct Associated{T, OPs<:AbstractQuasiMatrix{T}} <: OrthogonalPolynomial{T}
    P::OPs
end

associated(P) = Associated(P)

axes(Q::Associated) = axes(Q.P)
==(A::Associated, B::Associated) = A.P == B.P

orthogonalityweight(Q::Associated) = AssociatedWeight(Q.P)

function associated_jacobimatrix(X::Tridiagonal)
    c,a,b = subdiagonaldata(X),diagonaldata(X),supdiagonaldata(X)
    Tridiagonal(c[2:end], a[2:end], b[2:end])
end

function associated_jacobimatrix(X::SymTridiagonal)
    a,b = diagonaldata(X),supdiagonaldata(X)
    SymTridiagonal(a[2:end], b[2:end])
end
jacobimatrix(a::Associated) = associated_jacobimatrix(jacobimatrix(a.P))

associated(::ChebyshevT{T}) where T = ChebyshevU{T}()
associated(::ChebyshevU{T}) where T = ChebyshevU{T}()


const StieltjesPoint{T,V,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{BroadcastQuasiMatrix{T,typeof(-),Tuple{T,QuasiAdjoint{V,Inclusion{V,D}}}}}}
const ConvKernel{T,D} = BroadcastQuasiMatrix{T,typeof(-),Tuple{D,QuasiAdjoint{T,D}}}
const Hilbert{T,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{ConvKernel{T,Inclusion{T,D}}}}
const LogKernel{T,D} = BroadcastQuasiMatrix{T,typeof(log),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D}}}}}}
const PowKernel{T,D,F<:Real} = BroadcastQuasiMatrix{T,typeof(^),Tuple{BroadcastQuasiMatrix{T,typeof(abs),Tuple{ConvKernel{T,Inclusion{T,D}}}},F}}


@simplify function *(H::Hilbert, w::ChebyshevTWeight)
    T = promote_type(eltype(H), eltype(w))
    zeros(T, axes(w,1))
end

@simplify function *(H::Hilbert, w::ChebyshevUWeight)
    T = promote_type(eltype(H), eltype(w))
    fill(convert(T,π), axes(w,1))
end

@simplify function *(H::Hilbert, w::LegendreWeight)
    T = promote_type(eltype(H), eltype(w))
    x = axes(w,1)
    log.(x .+ 1) .- log.(1 .- x)
end

@simplify function *(H::Hilbert, wT::Weighted{<:Any,<:ChebyshevT}) 
    T = promote_type(eltype(H), eltype(wT))
    ChebyshevU{T}() * _BandedMatrix(Fill(-convert(T,π),1,∞), ℵ₀, -1, 1)
end

@simplify function *(H::Hilbert, wU::Weighted{<:Any,<:ChebyshevU}) 
    T = promote_type(eltype(H), eltype(wU))
    ChebyshevT{T}() * _BandedMatrix(Fill(convert(T,π),1,∞), ℵ₀, 1, -1)
end


@simplify function *(H::Hilbert, wP::Weighted{<:Any,<:OrthogonalPolynomial}) 
    P = wP.P
    w = orthogonalityweight(P)
    A = recurrencecoefficients(P)[1]
    (-A[1]*sum(w))*[zero(axes(P,1)) associated(P)] + (H*w) .* P
end

@simplify *(H::Hilbert, P::Legendre) = H * Weighted(P)

### 
# LogKernel
###

@simplify function *(L::LogKernel, wT::Weighted{<:Any,<:ChebyshevT}) 
    T = promote_type(eltype(L), eltype(wT))
    ChebyshevT{T}() * Diagonal(Vcat(-π*log(2*one(T)),-convert(T,π)./(1:∞)))
end



### 
# PowKernel
###

@simplify function *(K::PowKernel, wT::Weighted{<:Any,<:Jacobi}) 
    T = promote_type(eltype(K), eltype(wT))
    cnv,α = K.args
    x,y = K.args[1].args[1].args
    @assert x' == y
    β = (-α-1)/2
    error("Not implemented")
    # ChebyshevT{T}() * Diagonal(1:∞)
end


####
# StieltjesPoint
####

stieltjesmoment_jacobi_normalization(n::Int,α::Real,β::Real) = 2^(α+β)*gamma(n+α+1)*gamma(n+β+1)/gamma(2n+α+β+2)

@simplify function *(S::StieltjesPoint, w::AbstractJacobiWeight)
    α,β = w.a,w.b
    z,_ = parent(S).args[1].args
    (x = 2/(1-z);stieltjesmoment_jacobi_normalization(0,α,β)*HypergeometricFunctions.mxa_₂F₁(1,α+1,α+β+2,x))
end

@simplify function *(S::StieltjesPoint, wP::Weighted)
    P = wP.P
    w = orthogonalityweight(P)
    X = jacobimatrix(P)
    z, x = parent(S).args[1].args
    transpose((X'-z*I) \ [-sum(w)*_p0(P); zeros(∞)])
end


@simplify function *(S::StieltjesPoint, wT::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    P = parent(wT)
    z, x = parent(S).args[1].args
    z̃ = inbounds_getindex(parentindices(wT)[1], z)
    x̃ = axes(P,1)
    (inv.(z̃ .- x̃') * P)[:,parentindices(wT)[2]]
end

@simplify function *(H::Hilbert, wT::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Any}}) 
    P = parent(wT)
    x = axes(P,1)
    apply(*, inv.(x .- x'), P)[parentindices(wT)...]
end


@simplify function *(L::LogKernel, wT::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Slice}}) 
    V = promote_type(eltype(L), eltype(wT))
    wP = parent(wT)
    kr, jr = parentindices(wT)
    x = axes(wP,1)
    w,T = arguments(ApplyLayout{typeof(*)}(), wP)
    @assert w isa ChebyshevTWeight
    @assert T isa ChebyshevT
    D = T \ (log.(abs.(x .- x')) * wP)
    c = inv(2*kr.A)
    T[kr,:] * Diagonal(Vcat(2*convert(V,π)*c*log(c), 2c*D.diag.args[2]))
end

### generic fallback
for Op in (:Hilbert, :StieltjesPoint, :LogKernel, :PowKernel)
    @eval @simplify function *(H::$Op, wP::WeightedBasis{<:Any,<:Weight,<:Any}) 
        w,P = wP.args
        Q = OrthogonalPolynomial(w)
        (H * Weighted(Q)) * (Q \ P)
    end
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
function getindex(K::PowerLawMatrix{T,PP}, I::Vararg{Integer,2}) where {T,PP<:AbstractQuasiMatrix}
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
    a = K.args[2]
    t = K.args[1][zero(typeof(a))]
    return Q*PowerLawMatrix(Q,a,t)
end

####
# recurrence evaluation
####
# this function evaluates the recurrence and returns the full operator. 
# We don't use this outside of the initial block.
function gennormalizedpower(a::T, t::T, ℓ::Integer) where T<:Real
    # initialization
    ℓ = ℓ+3
    coeff = zeros(T,ℓ,ℓ)
    # load in explicit initial cases
    coeff[1,1] = PLnorminitial00(t,a)
    coeff[1,2] = PLnorminitial01(t,a)
    coeff[2,2] = PLnorminitial11(t,a)
    coeff[2,3] = PLnorminitial12(t,a)
    # we have to build these two cases with some care
    coeff[1,3] = t/((a+3)/3)*normconst_Pnadd1(1,t)*coeff[1,2]+(a/3)/((a+3)/3)*normconst_Pnsub1(1,t)*coeff[1,1]
    #the remaining cases can be constructed iteratively
    @inbounds for m = 2:ℓ-2
        # first row
        coeff[1,m+2] = (t/((a+m+2)/(2*m+1))*normconst_Pnadd1(m,t)*coeff[1,m+1]+((a-m+1)/(2*m+1))/((a+m+2)/(2*m+1))*normconst_Pnsub1(m,t)*coeff[1,m])
        # second row
        coeff[2,m+2] = (t/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pnadd1(m,t)*coeff[2,m+1]+((a+1)/(2-m*(m+1)))/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pmnmix(1,m,t)*coeff[1,m+1]-(m*(a+2-m)/((2*m+1)*(1-m)))*1/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pnsub1(m,t)*coeff[2,m])
        # build remaining row elements
        @inbounds for j=1:m-1
            coeff[j+2,m+1] = (t/((j+1)*(a+m+j+2)/((2*j+1)*(m+j+1)))*normconst_Pnadd1(j,t)*coeff[j+1,m+1]+((a+1)*m/(m*(m+1)-j*(j+1)))/((j+1)*(a+m+j+2)/((2*j+1)*(m+j+1)))*normconst_Pmnmix(m,j,t)*coeff[j+1,m]-(j*(a+m-j+1)/((2*j+1)*(m-j)))*1/((j+1)*(a+m+j+2)/((2*j+1)*(m+j+1)))*normconst_Pnsub1(j,t)*coeff[j,m+1])
        end
    end
    #matrix is symmetric
    @inbounds for m = 1:ℓ
        @inbounds for n = m+1:ℓ
            coeff[n,m] = coeff[m,n]
        end
    end
    return coeff[1:ℓ-3,1:ℓ-3]
end

# modify recurrence coefficients to work for normalized Legendre
normconst_Pnadd1(m::Integer, settype::T) where T<:Real = sqrt(2*m+3*one(T))/sqrt(2*m+one(T))
normconst_Pnsub1(m::Integer, settype::T) where T<:Real = sqrt(2*m+3*one(T))/sqrt(2*m-one(T))
normconst_Pmnmix(n::Integer, m::Integer, settype::T) where T<:Real = sqrt(2*m+3*one(T))*sqrt(2*n+one(T))/(sqrt(2*m+one(T))*sqrt(2*n-one(T)))
# these explicit initial cases are needed to kick off the recurrence
function PLnorminitial00(t::Real, a::Real)
    return ((t+1)^(a+1)-(t-1)^(a+1))/(2*(a+1))
end
function PLnorminitial01(t::T, a::T) where T<:Real
    return sqrt(one(T)*3)*((t+1)^(a+1)*(-a+t-1)-(a+t+1)*(t-1)^(a+1))/(2*(a+1)*(a+2))
end
function PLnorminitial11(t::T, a::T) where T<:Real
    return 3*((t+1)^(a+1)*(a^2+a*(3-2*t)+2*(t-1)*t+2)-(t-1)^(a+1)*(a^2+a*(2*t+3)+2*(t^2+t+1)))/(2*(a+1)*(a+2)*(a+3))
end
function PLnorminitial12(t::T, a::T) where T<:Real
    return -sqrt(one(T)*15)*(((1+t)^(1+a)*((1+a)^2*(3+a)-(3+2*a*(5+2*a))*t+9*(1+a)*t^2-9*t^3)+(-1+t)^(1+a)*((1+a)^2*(3+a)+(3+2*a*(5+2*a))*t+9*(1+a)*t^2+9*t^3))/(2*(1+a)*(2+a)*(3+a)*(4+a)))
end

# the following version takes a previously computed block that has been resized and fills in the missing data guided by indices in inds
function fillcoeffmatrix!(K, inds)
    # the remaining cases can be constructed iteratively
    a = K.a; t = K.t;
    @inbounds for m in inds
        m = m-2
        # first row
        K.data[1,m+2] = (t/((a+m+2)/(2*m+1))*normconst_Pnadd1(m,t)*K.data[1,m+1]+((a-m+1)/(2*m+1))/((a+m+2)/(2*m+1))*normconst_Pnsub1(m,t)*K.data[1,m])
        # second row
        K.data[2,m+2] = (t/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pnadd1(m,t)*K.data[2,m+1]+((a+1)/(2-m*(m+1)))/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pmnmix(1,m,t)*K.data[1,m+1]-(m*(a+2-m)/((2*m+1)*(1-m)))*1/((m+1)*(a+m+3)/((2*m+1)*(m+2)))*normconst_Pnsub1(m,t)*K.data[2,m])
        # build remaining row elements
        @inbounds for j = 1:m
            K.data[j+2,m+2] = (t/((j+1)*(a+m+j+3)/((2*j+1)*(m+j+2)))*normconst_Pnadd1(j,t)*K.data[j+1,m+2]+((a+1)*(m+1)/((m+1)*(m+2)-j*(j+1)))/((j+1)*(a+m+j+3)/((2*j+1)*(m+j+2)))*normconst_Pmnmix(m+1,j,t)*K.data[j+1,m+1]-(j*(a+m-j+2)/((2*j+1)*(m+1-j)))*1/((j+1)*(a+m+j+3)/((2*j+1)*(m+j+2)))*normconst_Pnsub1(j,t)*K.data[j,m+2])
        end
    end
    # matrix is symmetric
    @inbounds for m in reverse(inds)
        K.data[m,1:end] = K.data[1:end,m]
    end
end