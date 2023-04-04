function cholesky_jacobimatrix(w::Function, P::OrthogonalPolynomial)
    !(P isa Normalized) && error("Polynomials must be orthonormal.")
    W = Symmetric(P \ (w.(axes(P,1)) .* P)) # Compute weight multiplication via Clenshaw
    bands = CholeskyJacobiBands(W, P)
    return SymTridiagonal(bands[1,:],bands[2,:])
end
function cholesky_jacobimatrix(W::AbstractMatrix, P::OrthogonalPolynomial)
    !(P isa Normalized) && error("Polynomials must be orthonormal.")
    !(W isa Symmetric) && error("Weight modification matrix must be symmetric.")
    bands = CholeskyJacobiBands(W, P)
    return SymTridiagonal(bands[1,:],bands[2,:])
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data as two bands
mutable struct CholeskyJacobiBands{T} <: AbstractCachedMatrix{T}
    data::Matrix{T}
    U::UpperTriangular
    X::SymTridiagonal{T}
    datasize::Int
    array
end

# Computes the initial data for the Jacobi operator bands
function CholeskyJacobiBands(W::Symmetric{T}, P::OrthogonalPolynomial) where T
    U = cholesky(W).U
    X = jacobimatrix(P)
    dat = zeros(T,2,10)
    for k in 1:10
        dat[1,k] = (U * (X * (U \ [zeros(k-1); 1; zeros(∞)])))[k]
        dat[2,k] = (U * (X * (U \ [zeros(k); 1; zeros(∞)])))[k]
    end
    return CholeskyJacobiBands{T}(dat, U, X, 10, dat)
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
    for k in inds
        J.data[1,k] = (J.U * (J.X * (J.U \ [zeros(k-1); 1; zeros(∞)])))[k]
        J.data[2,k] = (J.U * (J.X * (J.U \ [zeros(k); 1; zeros(∞)])))[k]
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

# Note: The sqrtW passed to qr_jacobimatrix is square root of the weight modification
function qr_jacobimatrix(sqrtw::Function, P::OrthogonalPolynomial)
    !(P isa Normalized) && error("Polynomials must be orthonormal.")
    sqrtW = (P \ (sqrtw.(axes(P,1)) .* P)) # Compute weight multiplication via Clenshaw
    bands = QRJacobiBands(sqrtW,P)
    return SymTridiagonal(bands[1,:],bands[2,:])
end
function qr_jacobimatrix(sqrtW::AbstractMatrix, P::OrthogonalPolynomial)
    !(P isa Normalized) && error("Polynomials must be orthonormal.")
    !(sqrtW isa Symmetric) && error("Weight modification matrix must be symmetric.")
    bands = QRJacobiBands(sqrtW,P)
    return SymTridiagonal(bands[1,:],bands[2,:])
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data as two bands
mutable struct QRJacobiBands{T} <: AbstractCachedMatrix{T}
    data::Matrix{T}
    U
    X::SymTridiagonal{T}
    datasize::Int
    array
end

# Computes the initial data for the Jacobi operator bands
function QRJacobiBands(sqrtW, P::OrthogonalPolynomial{T}) where T
    U = qr(sqrtW).R
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U)
    X = jacobimatrix(P)
    dat = zeros(T,2,10)
    for k in 1:10
        dat[1,k] = (U * (X * (U \ [zeros(k-1); 1; zeros(∞)])))[k]
        dat[2,k] = (U * (X * (U \ [zeros(k); 1; zeros(∞)])))[k]
    end
    return QRJacobiBands{T}(dat, U, X, 10, dat)
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
    for k in inds
        J.data[1,k] = (J.U * (J.X * (J.U \ [zeros(k-1); 1; zeros(∞)])))[k]
        J.data[2,k] = (J.U * (J.X * (J.U \ [zeros(k); 1; zeros(∞)])))[k]
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
