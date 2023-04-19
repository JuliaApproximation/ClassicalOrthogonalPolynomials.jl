"""
cholesky_jacobimatrix(w, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x) w_p(x)` where `w_p(x)` is the weight of the polynomials in `P`.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs are `w` as a function or `W` as an infinite matrix representing multiplication with the function `w` on the basis `P`.

An optional bool can be supplied, i.e. `cholesky_jacobimatrix(sqrtw, P, false)` to disable checks of symmetry for the weight multiplication matrix and orthonormality for the basis (use with caution).
"""
function cholesky_jacobimatrix(w::Function, P::OrthogonalPolynomial, checks::Bool = true)
    checks && !(P isa Normalized) && error("Polynomials must be orthonormal.")
    W = Symmetric(P \ (w.(axes(P,1)) .* P)) # Compute weight multiplication via Clenshaw
    return cholesky_jacobimatrix(W, P, false)     # At this point checks already passed or were entered as false, no need to recheck
end
function cholesky_jacobimatrix(W::AbstractMatrix, P::OrthogonalPolynomial, checks::Bool = true)
    checks && !(P isa Normalized) && error("Polynomials must be orthonormal.")
    checks && !(W isa Symmetric) && error("Weight modification matrix must be symmetric.")
    return SymTridiagonal(CholeskyJacobiBands{:dv}(W,P),CholeskyJacobiBands{:ev}(W,P))
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in cached bands
mutable struct CholeskyJacobiBands{dv,T} <: AbstractCachedVector{T}
    data::Vector{T}       # store band entries, :dv for diagonal, :ev for off-diagonal
    U::UpperTriangular{T} # store upper triangular conversion matrix (needed to extend available entries)
    UX::ApplyArray{T}     # store U*X, where X is the Jacobi matrix of the original P (needed to extend available entries)
    datasize::Int         # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function CholeskyJacobiBands{:dv}(W, P::OrthogonalPolynomial{T}) where T
    U = cholesky(W).U
    X = jacobimatrix(P)
    UX = ApplyArray(*,U,X)
    dv = zeros(T,2) # compute a length 2 vector on first go
    dv[1] = dot(view(UX,1,1), U[1,1] \ [one(T)])
    dv[2] = dot(view(UX,2,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    return CholeskyJacobiBands{:dv,T}(dv, U, UX, 2)
end
function CholeskyJacobiBands{:ev}(W, P::OrthogonalPolynomial{T}) where T
    U = cholesky(W).U
    X = jacobimatrix(P)
    UX = ApplyArray(*,U,X)
    ev = zeros(T,2) # compute a length 2 vector on first go
    ev[1] = dot(view(UX,1,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[2] = dot(view(UX,2,1:3), U[1:3,1:3] \ [zeros(T,2); one(T)])
    return CholeskyJacobiBands{:ev,T}(ev, U, UX, 2)
end

size(::CholeskyJacobiBands) = (ℵ₀,) # Stored as an infinite cached vector

# Resize and filling functions for cached implementation
function resizedata!(K::CholeskyJacobiBands, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        resize!(K.data,nm)
        cache_filldata!(K, νμ:nm)
        K.datasize = nm
    end
    K
end
function cache_filldata!(J::CholeskyJacobiBands{:dv,T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    getindex(J.U,inds[end]+1,inds[end]+1)
    getindex(J.UX,inds[end]+1,inds[end]+1)

    ek = [zero(T); one(T)]
    @inbounds for k in inds
        J.data[k] = dot(view(J.UX,k,k-1:k), J.U[k-1:k,k-1:k] \ ek)
    end
end
function cache_filldata!(J::CholeskyJacobiBands{:ev, T}, inds::UnitRange{Int}) where T
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

An optional bool can be supplied, i.e. `qr_jacobimatrix(sqrtw, P, false)` to disable checks of symmetry for the weight multiplication matrix and orthonormality for the basis (use with caution).
"""
function qr_jacobimatrix(sqrtw::Function, P::OrthogonalPolynomial, checks::Bool = true)
    checks && !(P isa Normalized) && error("Polynomials must be orthonormal.")
    sqrtW = (P \ (sqrtw.(axes(P,1)) .* P))  # Compute weight multiplication via Clenshaw
    return qr_jacobimatrix(sqrtW, P, false) # At this point checks already passed or were entered as false, no need to recheck
end
function qr_jacobimatrix(sqrtW::AbstractMatrix, P::OrthogonalPolynomial, checks::Bool = true)
    checks && !(P isa Normalized) && error("Polynomials must be orthonormal.")
    checks && !(sqrtW isa Symmetric) && error("Weight modification matrix must be symmetric.")
    K = SymTridiagonal(QRJacobiBands{:dv}(sqrtW,P),QRJacobiBands{:ev}(sqrtW,P))
    return K
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in cached bands
mutable struct QRJacobiBands{dv,T} <: AbstractCachedVector{T}
    data::Vector{T}       # store band entries, :dv for diagonal, :ev for off-diagonal
    U::ApplyArray{T}      # store upper triangular conversion matrix (needed to extend available entries)
    UX::ApplyArray{T}     # store U*X, where X is the Jacobi matrix of the original P (needed to extend available entries)
    datasize::Int         # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function QRJacobiBands{:dv}(sqrtW, P::OrthogonalPolynomial{T}) where T
    U = qr(sqrtW).R
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U)
    X = jacobimatrix(P)
    UX = ApplyArray(*,U,X)
    dv = zeros(T,2) # compute a length 2 vector on first go
    dv[1] = dot(view(UX,1,1), U[1,1] \ [one(T)])
    dv[2] = dot(view(UX,2,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    return QRJacobiBands{:dv,T}(dv, U, UX, 2)
end
function QRJacobiBands{:ev}(sqrtW, P::OrthogonalPolynomial{T}) where T
    U = qr(sqrtW).R
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U)
    X = jacobimatrix(P)
    UX = ApplyArray(*,U,X)
    ev = zeros(T,2) # compute a length 2 vector on first go
    ev[1] = dot(view(UX,1,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[2] = dot(view(UX,2,1:3), U[1:3,1:3] \ [zeros(T,2); one(T)])
    return QRJacobiBands{:ev,T}(ev, U, UX, 2)
end

size(::QRJacobiBands) = (ℵ₀,) # Stored as an infinite cached vector

# Resize and filling functions for cached implementation
function resizedata!(K::QRJacobiBands, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        resize!(K.data,nm)
        cache_filldata!(K, νμ:nm)
        K.datasize = nm
    end
    K
end
function cache_filldata!(J::QRJacobiBands{:dv,T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    getindex(J.U,inds[end]+1,inds[end]+1)
    getindex(J.UX,inds[end]+1,inds[end]+1)

    ek = [zero(T); one(T)]
    @inbounds for k in inds
        J.data[k] = dot(view(J.UX,k,k-1:k), J.U[k-1:k,k-1:k] \ ek)
    end
end
function cache_filldata!(J::QRJacobiBands{:ev, T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    getindex(J.U,inds[end]+1,inds[end]+1)
    getindex(J.UX,inds[end]+1,inds[end]+1)

    ek = [zeros(T,2); one(T)]
    @inbounds for k in inds
        J.data[k] = dot(view(J.UX,k,k-1:k+1), J.U[k-1:k+1,k-1:k+1] \ ek)
    end
end
