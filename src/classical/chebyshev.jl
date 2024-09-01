"""
ChebyshevWeight{kind,T}()

is a quasi-vector representing the Chebyshev weight of the specified kind on -1..1.
That is, `ChebyshevWeight{1}()` represents `1/sqrt(1-x^2)`, and
`ChebyshevWeight{2}()` represents `sqrt(1-x^2)`.
"""
struct ChebyshevWeight{kind,T} <: AbstractJacobiWeight{T} end
ChebyshevWeight{kind}() where kind = ChebyshevWeight{kind,Float64}()
ChebyshevWeight() = ChebyshevWeight{1,Float64}()

AbstractQuasiArray{T}(::ChebyshevWeight{kind}) where {T,kind} = ChebyshevWeight{kind,T}()
AbstractQuasiVector{T}(::ChebyshevWeight{kind}) where {T,kind} = ChebyshevWeight{kind,T}()

const ChebyshevTWeight = ChebyshevWeight{1}
const ChebyshevUWeight = ChebyshevWeight{2}

getproperty(w::ChebyshevTWeight{T}, ::Symbol) where T = -one(T)/2
getproperty(w::ChebyshevUWeight{T}, ::Symbol) where T = one(T)/2

hasboundedendpoints(::ChebyshevUWeight) = true

"""
Chebyshev{kind,T}()

is a quasi-matrix representing Chebyshev polynomials of the specified kind (1, 2, 3, or 4)
on -1..1.
"""
struct Chebyshev{kind,T} <: AbstractJacobi{T} end
Chebyshev{kind}() where kind = Chebyshev{kind,Float64}()

AbstractQuasiArray{T}(::Chebyshev{kind}) where {T,kind} = Chebyshev{kind,T}()
AbstractQuasiMatrix{T}(::Chebyshev{kind}) where {T,kind} = Chebyshev{kind,T}()

const ChebyshevT = Chebyshev{1}
const ChebyshevU = Chebyshev{2}

show(io::IO, P::ChebyshevTWeight{Float64}) = summary(io, P)
show(io::IO, P::ChebyshevUWeight{Float64}) = summary(io, P)
summary(io::IO, ::ChebyshevTWeight{Float64}) = print(io, "ChebyshevTWeight()")
summary(io::IO, ::ChebyshevUWeight{Float64}) = print(io, "ChebyshevUWeight()")

# conveniences...perhaps too convenient
Chebyshev() = Chebyshev{1}()


broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), ::ChebyshevWeight{kind,T}, ::Chebyshev{kind,V}) where {kind,T,V} = Weighted(Chebyshev{kind,promote_type(T,V)}())
broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), ::ChebyshevWeight{kind,T}, ::Normalized{V,Chebyshev{kind,V}}) where {kind,T,V} = Weighted(Normalized(Chebyshev{kind,promote_type(T,V)}()))

chebyshevt() = ChebyshevT()
chebyshevt(d::AbstractInterval{T}) where T = ChebyshevT{float(T)}()[affine(d, ChebyshevInterval{T}()), :]
chebyshevt(d::Inclusion) = chebyshevt(d.domain)
chebyshevt(S::AbstractQuasiMatrix) = chebyshevt(axes(S,1))
chebyshevu() = ChebyshevU()
chebyshevu(d::AbstractInterval{T}) where T = ChebyshevU{float(T)}()[affine(d, ChebyshevInterval{T}()), :]
chebyshevu(d::Inclusion) = chebyshevu(d.domain)
chebyshevu(S::AbstractQuasiMatrix) = chebyshevu(axes(S,1))

"""
     chebyshevt(n, z)

computes the `n`-th Chebyshev polynomial of the first kind at `z`.
"""
chebyshevt(n::Integer, z::Number) = Base.unsafe_getindex(ChebyshevT{typeof(z)}(), z, n+1)
"""
     chebyshevt(n, z)

computes the `n`-th Chebyshev polynomial of the second kind at `z`.
"""
chebyshevu(n::Integer, z::Number) = Base.unsafe_getindex(ChebyshevU{typeof(z)}(), z, n+1)

chebysevtweight(d::AbstractInterval{T}) where T = ChebyshevTWeight{float(T)}[affine(d,ChebyshevInterval{T}())]
chebysevuweight(d::AbstractInterval{T}) where T = ChebyshevUWeight{float(T)}[affine(d,ChebyshevInterval{T}())]

==(a::Chebyshev{kind}, b::Chebyshev{kind}) where kind = true
==(a::Chebyshev, b::Chebyshev) = false
==(::Chebyshev, ::Jacobi) = false
==(::Jacobi, ::Chebyshev) = false
==(::Chebyshev, ::Legendre) = false
==(::Legendre, ::Chebyshev) = false

show(io::IO, w::ChebyshevT{Float64}) = summary(io, w)
show(io::IO, w::ChebyshevU{Float64}) = summary(io, w)

summary(io::IO, w::ChebyshevT{Float64}) = print(io, "ChebyshevT()")
summary(io::IO, w::ChebyshevU{Float64}) = print(io, "ChebyshevU()")

OrthogonalPolynomial(w::ChebyshevWeight{kind,T}) where {kind,T} = Chebyshev{kind,T}()
orthogonalityweight(P::Chebyshev{kind,T}) where {kind,T} = ChebyshevWeight{kind,T}()

function getindex(w::ChebyshevTWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    inv(sqrt(1-x^2))
end

function getindex(w::ChebyshevUWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    sqrt(1-x^2)
end

sum(::ChebyshevTWeight{T}) where T = convert(T,π)
sum(::ChebyshevUWeight{T}) where T = convert(T,π)/2

function _cumsum(w::ChebyshevTWeight{V}, dims) where V
    @assert dims == 1
    x = axes(w,1)
    (V(π)/2 .+ asin.(x))
end

normalizationconstant(::ChebyshevT{T}) where T = Vcat(sqrt(inv(convert(T,π))), Fill(sqrt(2/convert(T,π)),∞))



Jacobi(C::ChebyshevT{T}) where T = Jacobi(-one(T)/2,-one(T)/2)
Jacobi(C::ChebyshevU{T}) where T = Jacobi(one(T)/2,one(T)/2)


#######
# transform
#######


plan_transform(::ChebyshevT{T}, szs::NTuple{N,Int}, dims...) where {T,N} = plan_chebyshevtransform(T, szs, dims...)
plan_transform(::ChebyshevU{T}, szs::NTuple{N,Int}, dims...) where {T,N} = plan_chebyshevutransform(T, szs, dims...)


########
# Jacobi Matrix
########

jacobimatrix(C::ChebyshevT{T}) where T =
    Tridiagonal(Vcat(one(T), Fill(one(T)/2,∞)), Zeros{T}(∞), Fill(one(T)/2,∞))

jacobimatrix(C::ChebyshevU{T}) where T =
    SymTridiagonal(Zeros{T}(∞), Fill(one(T)/2,∞))



# These return vectors A[k], B[k], C[k] are from DLMF.
recurrencecoefficients(C::ChebyshevT) = (Vcat(1, Fill(2,∞)), Zeros{Int}(∞), Ones{Int}(∞))
recurrencecoefficients(C::ChebyshevU) = (Fill(2,∞), Zeros{Int}(∞), Ones{Int}(∞))

# special clenshaw!
# function copyto!(dest::AbstractVector{T}, v::SubArray{<:Any,1,<:Expansion{<:Any,<:ChebyshevT}, <:Tuple{AbstractVector{<:Number}}}) where T
#     f = parent(v)
#     (x,) = parentindices(v)
#     P,c = arguments(f)
#     clenshaw!(paddeddata(c), x, dest)
# end

###
# Mass matrix
###

weightedgrammatrix(::ChebyshevT{V}) where V = Diagonal([convert(V,π); Fill(convert(V,π)/2,∞)])
weightedgrammatrix(::ChebyshevU{V}) where V = Diagonal(Fill(convert(V,π)/2,∞))

@simplify *(A::QuasiAdjoint{<:Any,<:Weighted{<:Any,<:ChebyshevT}}, B::ChebyshevT) = weightedgrammatrix(ChebyshevT{promote_type(eltype(A),eltype(B))}())
@simplify *(A::QuasiAdjoint{<:Any,<:Weighted{<:Any,<:ChebyshevU}}, B::ChebyshevU) = weightedgrammatrix(ChebyshevU{promote_type(eltype(A),eltype(B))}())

function grammatrix(A::ChebyshevT{T}) where T
    f = (k,j) -> isodd(j-k) ? zero(T) : -((T(2)*(-1 + j^2 + k^2))/((-1 + j - k)*(-1 + j + k)*(1 + j - k)*(1 + j + k))
    BroadcastMatrix{T}(f, 0:∞, (0:∞)')
end

@simplify function *(A::QuasiAdjoint{<:Any,<:ChebyshevT}, B::ChebyshevT)
    T = promote_type(eltype(A), eltype(B))
    grammatrix(ChebyshevT{T}())
end

@simplify function *(A::QuasiAdjoint{<:Any,<:ChebyshevT}, B::ChebyshevU)
    T = promote_type(eltype(A), eltype(B))
    f = (k,j) -> isodd(j-k) ? zero(T) : (T(2)*(1 + j))/((1 + j - k)*(1 + j + k))
    BroadcastMatrix{T}(f, 0:∞, (0:∞)')
end


function grammatrix(A::Weighted{T,<:ChebyshevU}) where T
    f = (k,j) ->  isodd(j-k) ? zero(T) : -((T(4)*(1 + j)*(1 + k))/((-1 + j - k)*(1 + j - k)*(1 + j + k)*(3 + j + k)))
    BroadcastMatrix{T}(f, 0:∞, (0:∞)')
end


@simplify function *(A::QuasiAdjoint{<:Any,<:Weighted{<:Any,<:ChebyshevU}}, B::ChebyshevT)
    T = promote_type(eltype(A), eltype(B))
    W = parent(A)
    U = ChebyshevU{T}()
    (W'U) * (U\B)
end


##########
# Derivatives
##########

# Ultraspherical(1)\(D*Chebyshev())
function diff(S::ChebyshevT{T}; dims=1) where T
    D = _BandedMatrix((zero(T):∞)', ℵ₀, -1,1)
    ApplyQuasiMatrix(*, ChebyshevU{T}(), D)
end

function diff(W::Weighted{T,<:ChebyshevU}; dims=1) where T
    D =  _BandedMatrix((-one(T):-one(T):(-∞))', ℵ₀, 1,-1)
    ApplyQuasiMatrix(*, Weighted(ChebyshevT{T}()), D)
end


#####
# Conversion
#####

\(::Chebyshev{kind,T}, ::Chebyshev{kind,V}) where {kind,T,V} = SquareEye{promote_type(T,V)}(ℵ₀)

function \(U::ChebyshevU, C::ChebyshevT)
    T = promote_type(eltype(U), eltype(C))
    _BandedMatrix(Vcat(-Ones{T}(1,∞)/2,
                        Zeros{T}(1,∞),
                        Hcat(Ones{T}(1,1),Ones{T}(1,∞)/2)), ℵ₀, 0,2)
end

function \(w_A::Weighted{<:Any,<:ChebyshevT}, w_B::Weighted{<:Any,<:ChebyshevU})
    T = promote_type(eltype(w_A), eltype(w_B))
    _BandedMatrix(Vcat(Fill(one(T)/2, 1, ∞), Zeros{T}(1, ∞), Fill(-one(T)/2, 1, ∞)), ℵ₀, 2, 0)
end

\(w_A::Weighted{<:Any,<:ChebyshevU}, w_B::Weighted{<:Any,<:ChebyshevT}) = inv(w_B \ w_A)
\(T::ChebyshevT, U::ChebyshevU) = inv(U \ T)

####
# interrelationships
####

# (18.7.3)

function \(A::ChebyshevT, B::Jacobi)
    J = Jacobi(A)
    Diagonal(J[1,:]) * (J \ B)
end

function \(A::Jacobi, B::ChebyshevT)
    J = Jacobi(B)
    (A \ J) * Diagonal(inv.(J[1,:]))
end

function \(A::Chebyshev, B::Jacobi)
    J = Jacobi(A)
    Diagonal(A[1,:] .\ J[1,:]) * (J \ B)
end

function \(A::Jacobi, B::Chebyshev)
    J = Jacobi(B)
    (A \ J) * Diagonal(J[1,:] .\ B[1,:])
end

function \(A::Jacobi, B::ChebyshevU)
    T = promote_type(eltype(A), eltype(B))
    (A.a == A.b == one(T)/2) || throw(ArgumentError())
    Diagonal(B[1,:] ./ A[1,:])
end


# TODO: Toeplitz dot Hankel will be faster to generate
function \(A::ChebyshevT, B::Legendre)
    T = promote_type(eltype(A), eltype(B))
   UpperTriangular( BroadcastMatrix{T}((k,j) -> begin
            (iseven(k) == iseven(j) && j ≥ k) || return zero(T)
            k == 1 && return Λ(convert(T,j-1)/2)^2/π
            2/π * Λ(convert(T,j-k)/2) * Λ(convert(T,k+j-2)/2)
        end, convert(AbstractVector{T},1:∞), convert(AbstractVector{T},1:∞)'))
end

\(A::AbstractJacobi, B::Chebyshev) = ApplyArray(inv,B \ A)


function \(A::Jacobi, B::WeightedBasis{<:Any,<:JacobiWeight,<:Chebyshev})
    w, T = B.args
    J = Jacobi(T)
    wJ = w .* J
    (A \ wJ) * (J \ T)
end

function \(A::Chebyshev, B::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi})
    J = Jacobi(A)
    (A \ J) * (J \ B)
end

function \(A::Chebyshev, B::WeightedBasis{<:Any,<:JacobiWeight,<:Chebyshev})
    J = Jacobi(A)
    (A \ J) * (J \ B)
end



####
# sum
####

function _sum(::Weighted{T,<:ChebyshevU}, dims) where T
    @assert dims == 1
    Hcat(convert(T, π)/2, Zeros{T}(1,∞))
end

# Same normalization for T,V,W
function _sum(::Weighted{T,<:Chebyshev}, dims) where T
    @assert dims == 1
    Hcat(convert(T, π), Zeros{T}(1,∞))
end

function _cumsum(T::ChebyshevT{V}, dims) where V
    @assert dims == 1
    Σ = _BandedMatrix(Vcat(-one(V) ./ (-2:2:∞)', Zeros{V}(1,∞), Hcat(one(V), one(V) ./ (4:2:∞)')), ℵ₀, 0, 2)
    ApplyQuasiArray(*, T, Vcat((-1).^(0:∞)'* Σ, Σ))
end

function _cumsum(W::Weighted{V, ChebyshevT{V}}, dims) where V
    @assert dims == 1
    [cumsum(ChebyshevTWeight{V}()) Weighted(ChebyshevU{V}())] * Diagonal(Vcat(one(V), -inv.(one(V):∞)))
end

####
# algebra
####

broadcastbasis(::typeof(+), ::ChebyshevT, U::ChebyshevU) = U
broadcastbasis(::typeof(+), U::ChebyshevU, ::ChebyshevT) = U
