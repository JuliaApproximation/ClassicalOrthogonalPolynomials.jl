"""
cholesky_jacobimatrix(w, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x) w_p(x)` where `w_p(x)` is the weight of the polynomials in `P`.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs are `w` as a function or `W` as an infinite matrix representing multiplication with the function `w` on the basis `P`.

An optional bool can be supplied, i.e. `cholesky_jacobimatrix(sqrtw, P, false)` to disable checks of symmetry for the weight and orthonormality for the basis (use with caution).
"""
function cholesky_jacobimatrix(w::Function, P::OrthogonalPolynomial, checks::Bool = true)
    checks && !(P isa Normalized) && error("Polynomials must be orthonormal.")
    W = Symmetric(P \ (w.(axes(P,1)) .* P)) # Compute weight multiplication via Clenshaw
    bands = CholeskyJacobiBands(W, P)
    return SymTridiagonal(bands[1,:],bands[2,:])
end
function cholesky_jacobimatrix(W::AbstractMatrix, P::OrthogonalPolynomial, checks::Bool = true)
    checks && !(P isa Normalized) && error("Polynomials must be orthonormal.")
    checks && !(W isa Symmetric) && error("Weight modification matrix must be symmetric.")
    bands = CholeskyJacobiBands(W, P)
    return SymTridiagonal(bands[1,:],bands[2,:])
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data as two bands
mutable struct CholeskyJacobiBands{T} <: AbstractCachedMatrix{T}
    data::Matrix{T}         # store so-far computed block of entries 
    U::UpperTriangular      # store upper triangular conversion matrix (needed to extend available entries)
    X::SymTridiagonal{T}    # store the Jacobi matrix of the original basis (needed to extend available entries)
    datasize::Int           # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function CholeskyJacobiBands(W::Symmetric{T}, P::OrthogonalPolynomial) where T
    U = cholesky(W).U
    bds = bandwidths(U)[2]
    X = jacobimatrix(P)
    dat = zeros(T,2,10)
    UX = U*X
    dat[1,1] = (UX * (U \ [1; zeros(∞)]))[1]
    dat[2,1] = (UX * (U \ [zeros(1); 1; zeros(∞)]))[1]
    @inbounds for k in 2:10
        yk = view(UX,k,k-1:k+bds+1)
        dat[1,k] = dot(yk,pad(U[k-1:k,k-1:k] \ [0; 1],length(yk)))
        dat[2,k] = dot(yk,pad(U[k-1:k+1,k-1:k+1] \ [zeros(2); 1],length(yk)))
    end
    return CholeskyJacobiBands{T}(dat, U, X, 10)
end

size(::CholeskyJacobiBands) = (2,ℵ₀) # Stored as two infinite bands

# Resize and filling functions for cached implementation
function resizedata!(K::CholeskyJacobiBands, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        olddata = copy(K.data)
        K.data = similar(K.data, 2, nm)
        K.data[axes(olddata)...] = olddata
        inds = νμ:nm
        cache_filldata!(K, inds)
        K.datasize = nm
    end
    K
end
function cache_filldata!(J::CholeskyJacobiBands, inds::UnitRange{Int})
    UX = J.U*J.X
    bds = bandwidths(J.U)[2]
    @inbounds for k in inds
        yk = view(UX,k,k-1:k+bds+1)
        J.data[1,k] = dot(yk,pad(J.U[k-1:k,k-1:k] \ [0; 1],length(yk)))
        J.data[2,k] = dot(yk,pad(J.U[k-1:k+1,k-1:k+1] \ [zeros(2); 1],length(yk)))
    end
end

function getindex(K::CholeskyJacobiBands, k::Integer, j::Integer)
    resizedata!(K, max(k,j))
    K.data[k, j]
end
function getindex(K::CholeskyJacobiBands, kr::Integer, jr::UnitRange{Int})
    resizedata!(K, maximum(jr))
    K.data[kr, jr]
end
function getindex(K::CholeskyJacobiBands, I::Vararg{Int,2})
    resizedata!(K,maximum(I))
    getindex(K.data,I[1],I[2])
end

"""
qr_jacobimatrix(sqrtw, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x) w_p(x)` where `w_p(x)` is the weight of the polynomials in `P`.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs for `sqrtw` are the square root of the weight modification as a function or `sqrtW` as an infinite matrix representing multiplication with the function `sqrt(w)` on the basis `P`.

An optional bool can be supplied, i.e. `qr_jacobimatrix(sqrtw, P, false)` to disable checks of symmetry for the weight and orthonormality for the basis (use with caution).
"""
function qr_jacobimatrix(sqrtw::Function, P::OrthogonalPolynomial, checks::Bool = true)
    checks && !(P isa Normalized) && error("Polynomials must be orthonormal.")
    sqrtW = (P \ (sqrtw.(axes(P,1)) .* P)) # Compute weight multiplication via Clenshaw
    bands = QRJacobiBands(sqrtW,P)
    K = SymTridiagonal(bands[1,:],bands[2,:])
    return K
end
function qr_jacobimatrix(sqrtW::AbstractMatrix, P::OrthogonalPolynomial, checks::Bool = true)
    checks && !(P isa Normalized) && error("Polynomials must be orthonormal.")
    checks && !(sqrtW isa Symmetric) && error("Weight modification matrix must be symmetric.")
    bands = QRJacobiBands(sqrtW,P)
    K = SymTridiagonal(bands[1,:],bands[2,:])
    return K
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data as two bands
mutable struct QRJacobiBands{T} <: AbstractCachedMatrix{T}
    data::Matrix{T}         # store so-far computed block of entries 
    U::UpperTriangular      # store upper triangular conversion matrix (needed to extend available entries)
    X::SymTridiagonal{T}    # store the Jacobi matrix of the original basis (needed to extend available entries)
    datasize::Int           # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function QRJacobiBands(sqrtW, P::OrthogonalPolynomial{T}) where T
    U = qr(sqrtW).R
    bds = bandwidths(U)[2]
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U)
    X = jacobimatrix(P)
    UX = U*X
    dat = zeros(T,2,10)
    dat[1,1] = (UX * (U \ [1; zeros(∞)]))[1]
    dat[2,1] = (UX * (U \ [zeros(1); 1; zeros(∞)]))[1]
    @inbounds for k in 2:10
        yk = view(UX,k,k-1:k+bds+1)
        dat[1,k] = dot(yk,pad(U[k-1:k,k-1:k] \ [0; 1],length(yk)))
        dat[2,k] = dot(yk,pad(U[k-1:k+1,k-1:k+1] \ [zeros(2); 1],length(yk)))
    end
    return QRJacobiBands{T}(dat, UpperTriangular(U), X, 10)
end

size(::QRJacobiBands) = (2,ℵ₀) # Stored as two infinite bands

# Resize and filling functions for cached implementation
function resizedata!(K::QRJacobiBands, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        olddata = copy(K.data)
        K.data = similar(K.data, 2, nm)
        K.data[axes(olddata)...] = olddata
        inds = νμ:nm
        cache_filldata!(K, inds)
        K.datasize = nm
    end
    K
end
function cache_filldata!(J::QRJacobiBands, inds::UnitRange{Int})
    UX = J.U*J.X
    bds = bandwidths(J.U)[2]
    @inbounds for k in inds
        yk = view(UX,k,k-1:k+bds+1)
        J.data[1,k] = dot(yk,pad(J.U[k-1:k,k-1:k] \ [0; 1],length(yk)))
        J.data[2,k] = dot(yk,pad(J.U[k-1:k+1,k-1:k+1] \ [zeros(2); 1],length(yk)))
    end
end

function getindex(K::QRJacobiBands, k::Integer, j::Integer)
    resizedata!(K, max(k,j))
    K.data[k, j]
end
function getindex(K::QRJacobiBands, kr::Integer, jr::UnitRange{Int})
    resizedata!(K, maximum(jr))
    K.data[kr, jr]
end
function getindex(K::QRJacobiBands, I::Vararg{Int,2})
    resizedata!(K,maximum(I))
    getindex(K.data,I[1],I[2])
end
