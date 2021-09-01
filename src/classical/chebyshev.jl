"""
ChebyshevWeight{kind,T}()

is a quasi-vector representing the Chebyshev weight of the specified kind on -1..1.
That is, `ChebyshevWeight{1}()` represents `1/sqrt(1-x^2)`, and
`ChebyshevWeight{2}()` represents `sqrt(1-x^2)`.
"""
struct ChebyshevWeight{kind,T} <: AbstractJacobiWeight{T} end
ChebyshevWeight{kind}() where kind = ChebyshevWeight{kind,Float64}()
ChebyshevWeight() = ChebyshevWeight{1,Float64}()

getproperty(w::ChebyshevWeight{1,T}, ::Symbol) where T = -one(T)/2
getproperty(w::ChebyshevWeight{2,T}, ::Symbol) where T = one(T)/2


"""
Chebyshev{kind,T}()

is a quasi-matrix representing Chebyshev polynomials of the specified kind (1, 2, 3, or 4)
on -1..1.
"""
struct Chebyshev{kind,T} <: AbstractJacobi{T} end
Chebyshev{kind}() where kind = Chebyshev{kind,Float64}()

AbstractQuasiArray{T}(::Chebyshev{kind}) where {T,kind} = Chebyshev{kind,T}()
AbstractQuasiMatrix{T}(::Chebyshev{kind}) where {T,kind} = Chebyshev{kind,T}()

const WeightedChebyshev{kind,T} = WeightedBasis{T,<:ChebyshevWeight{kind},<:Chebyshev{kind}}

WeightedChebyshev{kind}() where kind = ChebyshevWeight{kind}() .* Chebyshev{kind}()
WeightedChebyshev{kind,T}() where {kind,T} = ChebyshevWeight{kind,T}(λ) .* Chebyshev{kind,T}(λ)

const ChebyshevTWeight = ChebyshevWeight{1}
const ChebyshevUWeight = ChebyshevWeight{2}
const ChebyshevT = Chebyshev{1}
const ChebyshevU = Chebyshev{2}
const WeightedChebyshevT = WeightedChebyshev{1}
const WeightedChebyshevU = WeightedChebyshev{2}

# conveniences...perhaps too convenient
Chebyshev() = Chebyshev{1}()
WeightedChebyshev() = WeightedChebyshevT()

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

sum(::ChebyshevWeight{1,T}) where T = convert(T,π)
sum(::ChebyshevWeight{2,T}) where T = convert(T,π)/2

normalizationconstant(::ChebyshevT{T}) where T = Vcat(sqrt(inv(convert(T,π))), Fill(sqrt(2/convert(T,π)),∞))



Jacobi(C::ChebyshevT{T}) where T = Jacobi(-one(T)/2,-one(T)/2)
Jacobi(C::ChebyshevU{T}) where T = Jacobi(one(T)/2,one(T)/2)


#######
# transform
#######

factorize(L::SubQuasiArray{T,2,<:ChebyshevT,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(L), plan_chebyshevtransform(Array{T}(undef, size(L,2))))

# TODO: extend plan_chebyshevutransform
factorize(L::SubQuasiArray{T,2,<:ChebyshevU,<:Tuple{<:Inclusion,<:OneTo}}) where T<:FastTransforms.fftwNumber =
    TransformFactorization(grid(L), plan_chebyshevutransform(Array{T}(undef, size(L,2))))


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
function copyto!(dest::AbstractVector{T}, v::SubArray{<:Any,1,<:Expansion{<:Any,<:ChebyshevT}, <:Tuple{AbstractVector{<:Number}}}) where T
    f = parent(v)
    (x,) = parentindices(v)
    P,c = arguments(f)
    clenshaw!(paddeddata(c), x, dest)
end

###
# Mass matrix
###

@simplify function *(Tc::QuasiAdjoint{<:Any,<:ChebyshevT}, wT::WeightedChebyshevT)
    V = promote_type(eltype(Tc), eltype(wT))
    Diagonal([convert(V,π); Fill(convert(V,π)/2,∞)])
end

@simplify function *(Tc::QuasiAdjoint{<:Any,<:ChebyshevU}, wT::WeightedChebyshevU)
    V = promote_type(eltype(Tc), eltype(wT))
    Diagonal(Fill(convert(V,π)/2,∞))
end

##########
# Derivatives
##########

# Ultraspherical(1)\(D*Chebyshev())
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::ChebyshevT)
    T = promote_type(eltype(D),eltype(S))
    A = _BandedMatrix((zero(T):∞)', ℵ₀, -1,1)
    ApplyQuasiMatrix(*, ChebyshevU{T}(), A)
end

@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, W::Weighted{<:Any,<:ChebyshevU})
    T = promote_type(eltype(D),eltype(W))
    Weighted(ChebyshevT{T}()) * _BandedMatrix((-one(T):-one(T):(-∞))', ℵ₀, 1,-1)
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

function \(w_A::WeightedChebyshevT, w_B::WeightedChebyshevU)
    wA,A = w_A.args
    wB,B = w_B.args
    T = promote_type(eltype(w_A), eltype(w_B))
    _BandedMatrix(Vcat(Fill(one(T)/2, 1, ∞), Zeros{T}(1, ∞), Fill(-one(T)/2, 1, ∞)), ℵ₀, 2, 0)
end

\(w_A::WeightedChebyshevU, w_B::WeightedChebyshevT) = inv(w_B \ w_A)
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
        end, 1:∞, (1:∞)'))
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

function _sum(A::WeightedBasis{T,<:ChebyshevUWeight,<:ChebyshevU}, dims) where T
    w, U = A.args
    @assert dims == 1
    Hcat(convert(T, π)/2, Zeros{T}(1,∞))
end

function _sum(A::WeightedBasis{T,<:ChebyshevWeight,<:Chebyshev}, dims) where T
    @assert dims == 1
    Hcat(convert(T, π), Zeros{T}(1,∞))
end

function cumsum(T::ChebyshevT{V}; dims::Integer) where V
    @assert dims == 1
    Σ = _BandedMatrix(Vcat(-one(V) ./ (-2:2:∞)', Zeros{V}(1,∞), Hcat(one(V), one(V) ./ (4:2:∞)')), ℵ₀, 0, 2)
    ApplyQuasiArray(*, T, Vcat((-1).^(0:∞)'* Σ, Σ))
end

cumsum(f::Expansion{<:Any,<:ChebyshevT}) = cumsum(f.args[1]; dims=1) * f.args[2]

####
# algebra
####

broadcastbasis(::typeof(+), ::ChebyshevT, U::ChebyshevU) = U
broadcastbasis(::typeof(+), U::ChebyshevU, ::ChebyshevT) = U