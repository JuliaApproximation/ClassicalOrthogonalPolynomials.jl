# This currently takes the weight multiplication operator as input.
# I will probably change this to take the weight function instead.
function cholesky_jacobimatrix(W::Symmetric)
    bands = CholeskyJacobiBands(W) # the cached array only needs to store two bands bc of symmetry
    return SymTridiagonal(bands[1,:],bands[2,:])
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data as two bands
mutable struct CholeskyJacobiBands{T} <: AbstractCachedMatrix{T}
    data::Matrix{T}
    U::UpperTriangular
    X::Symmetric{T}
    datasize::Int
    array
end

# SymTridiagonal currently doesn't parse as Symmetric, so here's a Q&D workaround for conversion
symmjacobim(J::SymTridiagonal) = Symmetric(BandedMatrix(0=>J.dv, 1=>J.ev))

# Computes the initial data for the Jacobi operator bands
function CholeskyJacobiBands(W::Symmetric{T}) where T
    U = cholesky(W).U
    X = symmjacobim(jacobimatrix(Normalized(Legendre()[affine(zero(T)..one(T),Inclusion(-one(T)..one(T))),:])))
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