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
    if z in axes(P,1) # use Hilbert
        transpose((inv.(x .- x') * wP)[z,:])
    end
    transpose((X'-z*I) \ [-sum(w)*_p0(P); zeros(∞)])
end

sqrtx2(z::Number) = sqrt(z-1)*sqrt(z+1)
sqrtx2(x::Real) = sign(x)*sqrt(x^2-1)

@simplify function *(S::StieltjesPoint, wP::Weighted{<:Any,<:ChebyshevU})
    z, x = parent(S).args[1].args
    ξ = inv(z + sqrtx2(z))
    transpose(π * ξ.^oneto(∞))
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
    function PowerLawMatrix{T, PP}(P::PP, a::T, t::T) where {T, PP<:AbstractQuasiMatrix}
        new{T, PP}(P,a,t, gennormalizedpower(a,t,50),(50,50))
    end
end
PowerLawMatrix(P::AbstractQuasiMatrix, a::T, t::T) where T = PowerLawMatrix{T,typeof(P)}(P,a,t)
size(K::PowerLawMatrix) = (ℵ₀,ℵ₀) # potential to add maximum size of operator
copy(K::PowerLawMatrix{T,PP}) where {T,PP} = K # Immutable entries

# data filling
#TODO: fix the weird inds
cache_filldata!(K::PowerLawMatrix, inds, _) = fillcoeffmatrix!(K, inds)

# because it really only makes sense to compute this symmetric operator in square blocks, we have to slightly rework some of LazyArrays caching and resizing
function getindex(K::PowerLawMatrix, k::Int, j::Int)
    resizedata!(K, k, j)
    K.data[k, j]
end
function getindex(K::PowerLawMatrix, kr::AbstractUnitRange, jr::AbstractUnitRange)
    resizedata!(K, maximum(kr),maximum(jr))
    K.data[kr, jr]
end
function resizedata!(K::PowerLawMatrix, n::Integer, m::Integer)
    olddata = K.data
    νμ = size(olddata)
    nm = max.(νμ,max(n,m))
    if νμ ≠ nm
        K.data = similar(K.data, nm...)
        K.data[axes(olddata)...] = olddata
    end
    if maximum(nm) > maximum(νμ)
        inds = maximum(νμ):maximum(nm)
        cache_filldata!(K, inds, inds)
        K.datasize = nm
    end
    K
end

####
# methods
####
function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), K::PowKernelPoint, Q::Normalized{<:Any,<:Legendre})
    tx,a = K.args
    t = tx[zero(typeof(a))]
    return Q*PowerLawMatrix(Q,a,t)
end

####
# operator generation
####

# evaluates ∫(t-x)^a Pn(x)P_m(x) dx for the case m=0, i.e. first row of operator, via backwards recursion
function powerleg_backwardsfirstrow(a::T, t::T, ℓ::Int) where T <: Real
    ℓ = ℓ+200
    coeff = zeros(BigFloat,ℓ)
    coeff[end-1] =  one(BigFloat)
    for m = reverse(1:ℓ-2)
        coeff[m] = (coeff[m+2]-t/((a+m+2)/(2*m+1))*normconst_Pnadd1(m,t)*coeff[m+1])/(((a-m+1)/(2*m+1))/((a+m+2)/(2*m+1))*normconst_Pnsub1(m,t))
    end
    coeff = PLnorminitial00(t,a)/coeff[1].*coeff
    return T.(coeff[1:ℓ-200])
end
# modify recurrence coefficients to work for normalized Legendre
normconst_Pnadd1(m::Int, settype::T) where T<:Real = sqrt(2*m+3*one(T))/sqrt(2*m+one(T))
normconst_Pnsub1(m::Int, settype::T) where T<:Real = sqrt(2*m+3*one(T))/sqrt(2*m-one(T))
normconst_Pmnmix(n::Int, m::Int, settype::T) where T<:Real = sqrt(2*m+3*one(T))*sqrt(2*n+one(T))/(sqrt(2*m+one(T))*sqrt(2*n-one(T)))
# useful explicit initial case
function PLnorminitial00(t::Real, a::Real)
    return ((t+1)^(a+1)-(t-1)^(a+1))/(2*(a+1))
end

# compute r-th coefficient of product expansion of order p and order q normalized Legendre polynomials
productseriescfs(p::T, q::T, r::T) where T = sqrt((2*p+1)*(2*q+1)/((2*(p+q-2*r)+1)))*(2*(p+q-2*r)+1)/(2*(p+q-r)+1)*exp(loggamma(r+one(T)/2)+loggamma(p-r+one(T)/2)+loggamma(q-r+one(T)/2)-loggamma(q-r+one(T))-loggamma(p-r+one(T))-loggamma(r+one(T))-loggamma(q+p-r+one(T)/2)+loggamma(q+p-r+one(T)))/π

# # this generates the entire operator via normalized product Legendre decomposition
# # this is very stable but scales rather poorly with high orders, so we only use it for testing
# function productoperator(a::T, t::T, ℓ::Int) where T
#     op::Matrix{T} = zeros(T,ℓ,ℓ)
#     # first row where n arbitrary and m==0
#     first = powerleg_backwardsfirstrow(a,t,2*ℓ+1)
#     op[1,:] = first[1:ℓ]
#     # generate remaining rows
#     for p = 1:ℓ-1
#         for q = p:ℓ-1
#             productcfs = zeros(T,2*ℓ+1)
#             for i = 0:min(p,q)
#                 productcfs[1+q+p-2*i] = productseriescfs(p,q,i)
#             end
#             op[p+1,q+1] = dot(first,productcfs)
#         end
#     end
#     # matrix is symmetric
#     for m = 1:ℓ
#         for n = m+1:ℓ
#             op[n,m] = op[m,n]
#         end
#     end
#     return op
# end

# This function returns the full ℓ×ℓ dot product operator, relying on several different methods for first row, second row, diagonal and remaining elements. We don't use this outside of the initial block.
function gennormalizedpower(a::T, t::T, ℓ::Int) where T <: Real
    # initialization
    ℓ = ℓ+3
    coeff = zeros(T,ℓ,ℓ)
    # construct first row via stable backwards recurrence
    first = powerleg_backwardsfirstrow(a,t,2*ℓ+1)
    coeff[1,:] = first[1:ℓ]
    # contruct second row via normalized product Legendre decomposition
    @inbounds for q = 1:ℓ-1
        productcfs = zeros(T,2*ℓ+1)
        productcfs[q+2] = productseriescfs(1,q,0)
        productcfs[q] = productseriescfs(1,q,1)
        coeff[2,q+1] = dot(first,productcfs)
    end
    # contruct the diagonal via normalized product Legendre decomposition
    @inbounds for q = 2:ℓ-1
        productcfs = zeros(T,2*ℓ+1)
        @inbounds for i = 0:q
            productcfs[1+2*q-2*i] = productseriescfs(q,q,i)
        end
        coeff[q+1,q+1] = dot(first,productcfs)
    end
    #the remaining cases can be constructed iteratively by means of a T-shaped recurrence
    @inbounds for m = 2:ℓ-2
        # build remaining row elements
        @inbounds for j = 1:m-2
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

# the following version takes a previously computed block that has been resized and fills in the missing data guided by indices in inds
function fillcoeffmatrix!(K::PowerLawMatrix, inds::AbstractUnitRange)
    # the remaining cases can be constructed iteratively
    a = K.a; t = K.t; T = eltype(promote(a,t));
    ℓ = maximum(inds)
    # fill in first row via stable backwards recurrence
    first = powerleg_backwardsfirstrow(a,t,2*ℓ+1)
    K.data[1,inds] = first[inds]
    # fill in second row via normalized product Legendre decomposition
    @inbounds for q = minimum(inds):ℓ-1
        productcfs = zeros(T,2*ℓ+1)
        productcfs[q+2] = productseriescfs(1,q,0)
        productcfs[q] = productseriescfs(1,q,1)
        K.data[2,q+1] = dot(first,productcfs)
    end
    # fill in the diagonal via normalized product Legendre decomposition
    @inbounds for q = minimum(inds):ℓ-1
        productcfs = zeros(T,2*ℓ+1)
        @inbounds for i = 0:q
            productcfs[1+2*q-2*i] = productseriescfs(q,q,i)
        end
        K.data[q+1,q+1] = dot(first,productcfs)
    end
    @inbounds for m in inds
        m = m-2
        # build remaining row elements
        @inbounds for j = 1:m-1
            K.data[j+2,m+2] = (t/((j+1)*(a+m+j+3)/((2*j+1)*(m+j+2)))*normconst_Pnadd1(j,t)*K.data[j+1,m+2]+((a+1)*(m+1)/((m+1)*(m+2)-j*(j+1)))/((j+1)*(a+m+j+3)/((2*j+1)*(m+j+2)))*normconst_Pmnmix(m+1,j,t)*K.data[j+1,m+1]-(j*(a+m-j+2)/((2*j+1)*(m+1-j)))*1/((j+1)*(a+m+j+3)/((2*j+1)*(m+j+2)))*normconst_Pnsub1(j,t)*K.data[j,m+2])
        end
    end
    # matrix is symmetric
    @inbounds for m in reverse(inds)
        K.data[m,1:end] = K.data[1:end,m]
    end
end

function dot(v::AbstractVector{T}, W::PowerLawMatrix, q::AbstractVector{T}) where T
    vpad, qpad = paddeddata(v), paddeddata(q)
    vl, ql = length(vpad), length(qpad)
    return dot(vpad,W[1:vl,1:ql]*qpad)
end