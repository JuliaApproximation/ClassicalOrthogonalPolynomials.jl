"""
Represent an Orthogonal polynomial which has a conversion operator from P, that is, Q = P * inv(U).
"""
struct ConvertedOrthogonalPolynomial{T, WW<:AbstractQuasiVector{T}, XX, UU, PP} <: OrthonormalPolynomial{T}
    weight::WW
    X::XX # jacobimatrix
    U::UU # conversion to P
    P::PP
end

_p0(Q::ConvertedOrthogonalPolynomial) = _p0(Q.P)

axes(Q::ConvertedOrthogonalPolynomial) = axes(Q.P)
MemoryLayout(::Type{<:ConvertedOrthogonalPolynomial}) = ConvertedOPLayout()
jacobimatrix(Q::ConvertedOrthogonalPolynomial) = Q.X
orthogonalityweight(Q::ConvertedOrthogonalPolynomial) = Q.weight


# transform to P * U if needed for differentiation, etc.
arguments(::ApplyLayout{typeof(*)}, Q::ConvertedOrthogonalPolynomial) = Q.P, ApplyArray(inv, Q.U)

OrthogonalPolynomial(w::AbstractQuasiVector) = OrthogonalPolynomial(w, orthogonalpolynomial(singularities(w)))
function OrthogonalPolynomial(w::AbstractQuasiVector, P::AbstractQuasiMatrix)
    Q = normalized(P)
    X = cholesky_jacobimatrix(w, Q)
    ConvertedOrthogonalPolynomial(w, X, X.dv.U, Q)
end

orthogonalpolynomial(w::AbstractQuasiVector) = OrthogonalPolynomial(w)
orthogonalpolynomial(w::SubQuasiArray) = orthogonalpolynomial(parent(w))[parentindices(w)[1],:]



"""
cholesky_jacobimatrix(w, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x) w_p(x)` where `w_p(x)` is the weight of the polynomials in `P` by computing a Cholesky decomposition of the weight modification.

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
orthogonal with respect to `w(x) w_p(x)` where `w_p(x)` is the weight of the polynomials in `P` by computing a QR decomposition of the square root weight modification.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs for `sqrtw` are the square root of the weight modification as a function or `sqrtW` as an infinite matrix representing multiplication with the function `sqrt(w)` on the basis `P`.

The underlying QR approach allows two methods, one which uses the Q matrix and one which uses the R matrix. To change between methods, an optional argument :Q or :R may be supplied. The default is to use the Q method.
"""
function qr_jacobimatrix(sqrtw::Function, P, method = :Q)
    Q = normalized(P)
    x = axes(P,1)
    sqrtW = (Q \ (sqrtw.(x) .* Q))  # Compute weight multiplication via Clenshaw
    return qr_jacobimatrix(sqrtW, Q, method)
end
function qr_jacobimatrix(sqrtW::AbstractMatrix, Q, method = :Q)
    isnormalized(Q) || error("Polynomials must be orthonormal")
    F = qr(sqrtW)
    SymTridiagonal(QRJacobiBand{:dv,method}(F,Q),QRJacobiBand{:ev,method}(F,Q))
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in cached bands
mutable struct QRJacobiBand{dv,method,T} <: AbstractCachedVector{T}
    data::Vector{T}             # store band entries, :dv for diagonal, :ev for off-diagonal
    U                           # store conversion, Q method: stores QR object. R method: only stores R.
    UX::AbstractMatrix{T}       # Auxilliary matrix. Q method: stores in-progress incomplete modification. R method: stores U*X for efficiency.
    P::OrthogonalPolynomial{T}  # Remember original polynomials
    datasize::Int               # size of so-far computed block
end

# Computes the initial data for the Jacobi operator bands
function QRJacobiBand{:dv,:Q}(F, P::OrthogonalPolynomial{T}) where T
    b = 3+bandwidths(F.R)[2]÷2
    X = jacobimatrix(P)
        # we fill 2 entries on the first run
    dv = zeros(T,2)
        # fill first entry (special case)
    M = X[1:b,1:b]
    getindex(F.factors,b,b) # pre-fill cached array
    v = [one(T);F.factors[1:b,1][2:end]]
    H = I - F.τ[1]*v*v'
    M = H*M*H
    dv[1] = M[1,1]
        # fill second entry
    v = [zero(T);one(T);F.factors[1:b,2][3:end]]
    H = I - F.τ[2]*v*v'
    K = H*M*H
    M = Matrix(X[2:b+1,2:b+1])
    M[1:end-1,1:end-1] = K[2:end,2:end]
    dv[2] = M[1,1] # sign correction due to QR not guaranteeing positive diagonal for R not needed on diagonals since contributions cancel
    return QRJacobiBand{:dv,:Q,T}(dv, F, M, P, 2)
end
function QRJacobiBand{:ev,:Q}(F, P::OrthogonalPolynomial{T}) where T
    b = 3+bandwidths(F.factors)[2]÷2
    X = jacobimatrix(P)
        # we fill 1 entry on the first run
    dv = zeros(T,1)
        # first step does not produce entries for the off-diagonal band (special case)
    M = X[1:b,1:b]
resizedata!(F.factors,b,b)
    v = [one(T);F.factors[1:b,1][2:end]]
    H = I - F.τ[1]*v*v'
    M = H*M*H
        # fill first off-diagonal entry
    v = [zero(T);one(T);F.factors[1:b,2][3:end]]
    H = I - F.τ[2]*v*v'
    K = H*M*H
    dv[1] = K[1,2]*sign(F.R[1,1]*F.R[2,2]) # includes possible correction for sign (only needed in off-diagonal case), since the QR decomposition does not guarantee positive diagonal on R
    M = Matrix(X[2:b+1,2:b+1])
    M[1:end-1,1:end-1] = K[2:end,2:end]
    return QRJacobiBand{:ev,:Q,T}(dv, F, M, P, 1)
end
function QRJacobiBand{:dv,:R}(F, P::OrthogonalPolynomial{T}) where T
    U = F.R
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U)  # QR decomposition does not force positive diagonals on R by default
    X = jacobimatrix(P)
    UX = ApplyArray(*,U,X)
    dv = zeros(T,2) # compute a length 2 vector on first go
    dv[1] = dot(view(UX,1,1), U[1,1] \ [one(T)])
    dv[2] = dot(view(UX,2,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    return QRJacobiBand{:dv,:R,T}(dv, U, UX, P, 2)
end
function QRJacobiBand{:ev,:R}(F, P::OrthogonalPolynomial{T}) where T
    U = F.R
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U) # QR decomposition does not force positive diagonals on R by default
    X = jacobimatrix(P)
    UX = ApplyArray(*,U,X)
    ev = zeros(T,2) # compute a length 2 vector on first go
    ev[1] = dot(view(UX,1,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[2] = dot(view(UX,2,1:3), U[1:3,1:3] \ [zeros(T,2); one(T)])
    return QRJacobiBand{:ev,:R,T}(ev, U, UX, P, 2)
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
function cache_filldata!(J::QRJacobiBand{:dv,:Q,T}, inds::UnitRange{Int}) where T
    b = 1+bandwidths(J.U.factors)[2]÷2
    # pre-fill cached arrays to avoid excessive cost from expansion in loop
    m, jj = inds[end], inds[2:end]
    X = jacobimatrix(J.P)[1:m+b+2,1:m+b+2]
    getindex(J.U.factors,m+b,m+b)
    getindex(J.U.τ,m)
    M = J.UX
    τ = J.U.τ
    F = J.U.factors
    dv = J.data
    @inbounds for n in jj
        v = [zero(T);one(T);F[n-1:n+b,n][3:end]]
        H = I-τ[n]*v*v'
        K = H*M*H
        M = Matrix(X[n:n+b+1,n:n+b+1])
        M[1:end-1,1:end-1] = K[2:end,2:end]
        dv[n] = M[1,1] # sign correction due to QR not guaranteeing positive diagonal for R not needed on diagonals since contributions cancel
    end
    J.UX = M
    J.data[jj] = dv[jj]
end
function cache_filldata!(J::QRJacobiBand{:ev,:Q,T}, inds::UnitRange{Int}) where T
    m, jj = 1+inds[end], inds[2:end]
    b = bandwidths(J.U.factors)[2]÷2
    # pre-fill cached arrays to avoid excessive cost from expansion in loop
    X = jacobimatrix(J.P)[1:m+b+2,1:m+b+2]
    resizedata!(J.U.factors,m+b,m+b)
    getindex(J.U.R,m,m)
    getindex(J.U.τ,m)
    M, τ, F, dv = J.UX, J.U.τ, J.U.factors, J.data
    D = sign.(view(J.U.R,band(0)).*view(J.U.R,band(0))[2:end])
    @inbounds for n in jj
        v = [zero(T);one(T);F[n:n+b+2,n+1][3:end]]
        H = I - τ[n+1]*v*v'
        K = H*M*H
        dv[n] = K[1,2]
        M = Matrix(X[n+1:n+b+3,n+1:n+b+3])
        M[1:end-1,1:end-1] = K[2:end,2:end]
    end
    J.UX = M
    J.data[jj] = dv[jj].*D[jj] # includes possible correction for sign (only needed in off-diagonal case), since the QR decomposition does not guarantee positive diagonal on R
end
function cache_filldata!(J::QRJacobiBand{:dv,:R,T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    m = inds[end]+1
    getindex(J.U,m,m)
    getindex(J.UX,m,m)

    ek = [zero(T); one(T)]
    dv, UX, U = J.data, J.UX, J.U
    @inbounds for k in inds
        dv[k] = dot(view(UX,k,k-1:k), U[k-1:k,k-1:k] \ ek)
    end
    J.data[inds] = dv[inds]
end
function cache_filldata!(J::QRJacobiBand{:ev,:R, T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    m = inds[end]+1
    getindex(J.U,m,m)
    getindex(J.UX,m,m)

    ek = [zeros(T,2); one(T)]
    dv, UX, U = J.data, J.UX, J.U
    @inbounds for k in inds
        dv[k] = dot(view(UX,k,k-1:k+1), U[k-1:k+1,k-1:k+1] \ ek)
    end
    J.data[inds] = dv[inds]
end
