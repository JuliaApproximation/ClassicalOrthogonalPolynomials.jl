"""
Represent an Orthogonal polynomial which has a conversion operator to P, that is, Q = P * R.
"""
struct ConvertedOrthogonalPolynomial{T, WW<:AbstractQuasiVector{T}, XX, RR, PP} <: OrthogonalPolynomial{T}
    weight::WW
    X::XX # jacobimatrix
    R::RR # conversion to P
    P::PP
end

# transform to P * U if needed for differentiation, etc.
arguments(::ApplyLayout{typeof(*)}, Q::ConvertedOrthogonalPolynomial) = Q.P, Q.R

# also change all the NormalizedOPLayout
@inline copy(L::Ldiv{Lay,<: AbstractConvertedOPLayout}) where Lay = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))

OrthogonalPolynomial(w::AbstractQuasiVector) = OrthogonalPolynomial(w, orthonormalpolynomial(singularities(w)))
function OrthogonalPolynomial(w::AbstractQuasiVector, P::AbstractQuasiMatrix)
    X = cholesky_jacobimatrix(w, P)
    ConvertedOrthogonalPolynomial(w, X, X.dv.U, P)
end



"""
cholesky_jacobimatrix(w, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x) w_p(x)` where `w_p(x)` is the weight of the polynomials in `P`.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs are `w` as a function or `W` as an infinite matrix representing multiplication with the function `w` on the basis `P`.
"""
cholesky_jacobimatrix(w::Function, P) = cholesky_jacobimatrix(w.(axes(P,1)), P)

function cholesky_jacobimatrix(w::AbstractQuasiVector, P)
    Q = normalized(P)
    W = Symmetric(Q \ (w .* Q)) # Compute weight multiplication via Clenshaw
    return cholesky_jacobimatrix(W, Q)
end
function cholesky_jacobimatrix(W::AbstractMatrix, Q)
    isnormalized(Q) || error("Polynomials must be orthonormal")
    issymmetric(W) || error("Weight modification matrix must be symmetric.")
    U = cholesky(W).U
    X = jacobimatrix(Q)
    UX = ApplyArray(*,U,X)
    return SymTridiagonal(CholeskyJacobiBand{:dv}(U, UX),CholeskyJacobiBand{:ev}(U, UX))
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in cached bands
mutable struct CholeskyJacobiBand{dv,T} <: AbstractCachedVector{T}
    data::Vector{T}       # store band entries, :dv for diagonal, :ev for off-diagonal
    U::UpperTriangular{T} # store upper triangular conversion matrix (needed to extend available entries)
    UX::ApplyArray{T}     # store U*X, where X is the Jacobi matrix of the original P (needed to extend available entries)
    datasize::Int         # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function CholeskyJacobiBand{:dv}(U::AbstractMatrix{T}, UX) where T
    dv = zeros(T,2) # compute a length 2 vector on first go
    dv[1] = dot(view(UX,1,1), U[1,1] \ [one(T)])
    dv[2] = dot(view(UX,2,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    return CholeskyJacobiBand{:dv,T}(dv, U, UX, 2)
end
function CholeskyJacobiBand{:ev}(U::AbstractMatrix{T}, UX) where T
    ev = zeros(T,2) # compute a length 2 vector on first go
    ev[1] = dot(view(UX,1,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[2] = dot(view(UX,2,1:3), U[1:3,1:3] \ [zeros(T,2); one(T)])
    return CholeskyJacobiBand{:ev,T}(ev, U, UX, 2)
end

size(::CholeskyJacobiBand) = (ℵ₀,) # Stored as an infinite cached vector

# Resize and filling functions for cached implementation
function resizedata!(K::CholeskyJacobiBand, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        resize!(K.data,nm)
        cache_filldata!(K, νμ:nm)
        K.datasize = nm
    end
    K
end
function cache_filldata!(J::CholeskyJacobiBand{:dv,T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    getindex(J.U,inds[end]+1,inds[end]+1)
    getindex(J.UX,inds[end]+1,inds[end]+1)

    ek = [zero(T); one(T)]
    @inbounds for k in inds
        J.data[k] = dot(view(J.UX,k,k-1:k), J.U[k-1:k,k-1:k] \ ek)
    end
end
function cache_filldata!(J::CholeskyJacobiBand{:ev, T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    getindex(J.U,inds[end]+1,inds[end]+1)
    getindex(J.UX,inds[end]+1,inds[end]+1)

    ek = [zeros(T,2); one(T)]
    @inbounds for k in inds
        J.data[k] = dot(view(J.UX,k,k-1:k+1), J.U[k-1:k+1,k-1:k+1] \ ek)
    end
end


"""
qr_jacobimatrix(sqrtw, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x) w_p(x)` where `w_p(x)` is the weight of the polynomials in `P`.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs for `sqrtw` are the square root of the weight modification as a function or `sqrtW` as an infinite matrix representing multiplication with the function `sqrt(w)` on the basis `P`.
"""
function qr_jacobimatrix(sqrtw::Function, P)
    Q = normalized(P)
    x = axes(P,1)
    sqrtW = (Q \ (sqrtw.(x) .* Q))  # Compute weight multiplication via Clenshaw
    return qr_jacobimatrix(sqrtW, Q)
end
function qr_jacobimatrix(sqrtW::AbstractMatrix, Q)
    isnormalized(P) || error("Polynomials must be orthonormal")
    issymmetric(sqrtW) || error("Weight modification matrix must be symmetric.")
    K = SymTridiagonal(QRJacobiBand{:dv}(sqrtW,Q),QRJacobiBand{:ev}(sqrtW,Q))
    return K
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in cached bands
mutable struct QRJacobiBand{dv,T} <: AbstractCachedVector{T}
    data::Vector{T}       # store band entries, :dv for diagonal, :ev for off-diagonal
    U::ApplyArray{T}      # store upper triangular conversion matrix (needed to extend available entries)
    UX::ApplyArray{T}     # store U*X, where X is the Jacobi matrix of the original P (needed to extend available entries)
    datasize::Int         # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function QRJacobiBand{:dv}(sqrtW, P::OrthogonalPolynomial{T}) where T
    U = qr(sqrtW).R
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U)
    X = jacobimatrix(P)
    UX = ApplyArray(*,U,X)
    dv = zeros(T,2) # compute a length 2 vector on first go
    dv[1] = dot(view(UX,1,1), U[1,1] \ [one(T)])
    dv[2] = dot(view(UX,2,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    return QRJacobiBand{:dv,T}(dv, U, UX, 2)
end
function QRJacobiBand{:ev}(sqrtW, P::OrthogonalPolynomial{T}) where T
    U = qr(sqrtW).R
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U)
    X = jacobimatrix(P)
    UX = ApplyArray(*,U,X)
    ev = zeros(T,2) # compute a length 2 vector on first go
    ev[1] = dot(view(UX,1,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[2] = dot(view(UX,2,1:3), U[1:3,1:3] \ [zeros(T,2); one(T)])
    return QRJacobiBand{:ev,T}(ev, U, UX, 2)
end

size(::QRJacobiBand) = (ℵ₀,) # Stored as an infinite cached vector

# Resize and filling functions for cached implementation
function resizedata!(K::QRJacobiBand, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        resize!(K.data,nm)
        cache_filldata!(K, νμ:nm)
        K.datasize = nm
    end
    K
end
function cache_filldata!(J::QRJacobiBand{:dv,T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    getindex(J.U,inds[end]+1,inds[end]+1)
    getindex(J.UX,inds[end]+1,inds[end]+1)

    ek = [zero(T); one(T)]
    @inbounds for k in inds
        J.data[k] = dot(view(J.UX,k,k-1:k), J.U[k-1:k,k-1:k] \ ek)
    end
end
function cache_filldata!(J::QRJacobiBand{:ev, T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    getindex(J.U,inds[end]+1,inds[end]+1)
    getindex(J.UX,inds[end]+1,inds[end]+1)

    ek = [zeros(T,2); one(T)]
    @inbounds for k in inds
        J.data[k] = dot(view(J.UX,k,k-1:k+1), J.U[k-1:k+1,k-1:k+1] \ ek)
    end
end
