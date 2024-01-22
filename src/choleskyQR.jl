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
    ConvertedOrthogonalPolynomial(w, X, parent(X.dv).U, Q)
end

orthogonalpolynomial(wP...) = OrthogonalPolynomial(wP...)
orthogonalpolynomial(w::SubQuasiArray) = orthogonalpolynomial(parent(w))[parentindices(w)[1],:]

OrthogonalPolynomial(w::Function, P::AbstractQuasiMatrix) = OrthogonalPolynomial(w.(axes(P,1)), P)
orthogonalpolynomial(w::Function, P::AbstractQuasiMatrix) = orthogonalpolynomial(w.(axes(P,1)), P)


"""
cholesky_jacobimatrix(w, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x)` by computing a Cholesky decomposition of the weight modification.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs are `w` as a function or `W` as an infinite matrix representing the weight modifier multiplication by the function `w / w_P` on `P` where `w_P` is the orthogonality weight of `P`.
"""
cholesky_jacobimatrix(w::Function, P) = cholesky_jacobimatrix(w.(axes(P,1)), P)

function cholesky_jacobimatrix(w::AbstractQuasiVector, P)
    Q = normalized(P)
    w_P = orthogonalityweight(P)
    W = Symmetric(Q \ ((w ./ w_P) .* Q)) # Compute weight multiplication via Clenshaw
    return cholesky_jacobimatrix(W, Q)
end
function cholesky_jacobimatrix(W::AbstractMatrix, Q)
    isnormalized(Q) || error("Polynomials must be orthonormal")
    U = cholesky(W).U
    X = jacobimatrix(Q)
    CJD = CholeskyJacobiData(U,cache(simplify(Mul(U,X))))
    return SymTridiagonal(view(CJD,:,1),view(CJD,:,2))
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in two cached bands which are generated in tandem but can be accessed separately.
mutable struct CholeskyJacobiData{T} <: LazyMatrix{T}
    dv::AbstractVector{T} # store diagonal band entries in adaptively sized vector
    ev::AbstractVector{T} # store off-diagonal band entries in adaptively sized vector
    U::UpperTriangular{T} # store upper triangular conversion matrix (needed to extend available entries)
    UX                    # stores cached U*X to speed up extension of entries
    datasize::Int         # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function CholeskyJacobiData(U::AbstractMatrix{T}, UX) where T
    # compute a length 2 vector on first go and circumvent BigFloat issue
    dv = zeros(T,2)
    ev = zeros(T,2)
    dv[1] = UX[1,1]/U[1,1] # this is dot(view(UX,1,1), U[1,1] \ [one(T)])
    dv[2] = -U[1,2]*UX[2,1]/(U[1,1]*U[2,2])+UX[2,2]/U[2,2] # this is dot(view(UX,2,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[1] = -UX[1,1]*U[1,2]/(U[1,1]*U[2,2])+UX[1,2]/U[2,2] # this is dot(view(UX,1,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[2] = UX[2,1]/U[3,3]*(-U[1,3]/U[1,1]+U[1,2]*U[2,3]/(U[1,1]*U[2,2]))+UX[2,2]/U[3,3]*(-U[2,3]/U[2,2])+UX[2,3]/U[3,3] # this is dot(view(UX,2,1:3), U[1:3,1:3] \ [zeros(T,2); one(T)])[1:3,1:3] \ [zeros(T,2); one(T)])
    return CholeskyJacobiData{T}(dv, ev, U, UX, 2)
end

size(::CholeskyJacobiData) = (ℵ₀,2) # Stored as two infinite cached bands

function getindex(C::SymTridiagonal{<:Any, <:SubArray{<:Any, 1, <:CholeskyJacobiData, <:Tuple, false}, <:SubArray{<:Any, 1, <:CholeskyJacobiData, <:Tuple, false}}, kr::UnitRange, jr::UnitRange)
    m = maximum(max(kr,jr))+1
    resizedata!(C.dv.parent,m,2)
    resizedata!(C.ev.parent,m,2)
    return copy(view(C,kr,jr))
end

function getindex(K::CholeskyJacobiData, n::Integer, m::Integer)
    @assert (m==1) || (m==2)
    resizedata!(K,n,m)
    m == 1 && return K.dv[n]
    m == 2 && return K.ev[n]
end

# Resize and filling functions for cached implementation
function resizedata!(K::CholeskyJacobiData, n::Integer, m::Integer)
    nm = max(n,m)
    νμ = K.datasize
    if nm > νμ
        resize!(K.dv,nm)
        resize!(K.ev,nm)
        _fillcholeskybanddata!(K, νμ:nm)
        K.datasize = nm
    end
    K
end

function _fillcholeskybanddata!(J::CholeskyJacobiData{T}, inds::UnitRange{Int}) where T
    # pre-fill U to prevent expensive step-by-step filling in
    m = inds[end]+1
    partialcholesky!(J.U.data, m)
    resizedata!(J.UX, m, m)
    dv, ev, U, UX = J.dv, J.ev, J.U, J.UX

    @inbounds Threads.@threads for k = inds  
        # this is dot(view(UX,k,k-1:k), U[k-1:k,k-1:k] \ ek)
        dv[k] = -U[k-1,k]*UX[k,k-1]/(U[k-1,k-1]*U[k,k])+UX[k,k]/U[k,k]
        ev[k] = UX[k,k-1]/U[k+1,k+1]*(-U[k-1,k+1]/U[k-1,k-1]+U[k-1,k]*U[k,k+1]/(U[k-1,k-1]*U[k,k]))+UX[k,k]/U[k+1,k+1]*(-U[k,k+1]/U[k,k])+UX[k,k+1]/U[k+1,k+1]  
    end
end


"""
qr_jacobimatrix(w, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x)` by computing a QR decomposition of the square root weight modification.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs for `w` are the target weight as a function or `sqrtW`, representing the multiplication operator of square root weight modification on the basis `P`.

The underlying QR approach allows two methods, one which uses the Q matrix and one which uses the R matrix. To change between methods, an optional argument :Q or :R may be supplied. The default is to use the Q method.
"""
function qr_jacobimatrix(w::Function, P, method = :Q)
    Q = normalized(P)
    x = axes(P,1)
    w_P = orthogonalityweight(P)
    sqrtW = (Q \ (sqrt.((w ./ w_P)) .* Q))  # Compute weight multiplication via Clenshaw
    return qr_jacobimatrix(sqrtW, Q, method)
end
function qr_jacobimatrix(sqrtW::AbstractMatrix{T}, Q, method = :Q) where T
    isnormalized(Q) || error("Polynomials must be orthonormal")
    F = qr(sqrtW)
    QRJD = QRJacobiData{method,T}(F,Q)
    SymTridiagonal(view(QRJD,:,1),view(QRJD,:,2))
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in two cached bands which are generated in tandem but can be accessed separately.
mutable struct QRJacobiData{method,T} <: LazyMatrix{T}
    dv::AbstractVector{T} # store diagonal band entries in adaptively sized vector
    ev::AbstractVector{T} # store off-diagonal band entries in adaptively sized vector
    U                     # store conversion, Q method: stores QR object. R method: only stores R.
    UX                    # Auxilliary matrix. Q method: stores in-progress incomplete modification. R method: stores U*X for efficiency.
    P                     # Remember original polynomials
    datasize::Int         # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function QRJacobiData{:Q,T}(F, P) where T
    b = 3+bandwidths(F.R)[2]÷2
    X = jacobimatrix(P)
    # we fill 1 entry on the first run and circumvent BigFloat issue
    dv = zeros(T,2) 
    ev = zeros(T,1)
    # fill first entry (special case)
    M = Matrix(X[1:b,1:b])
    resizedata!(F.factors,b,b)
    # special case for first entry double Householder product
    v = view(F.factors,1:b,1)
    reflectorApply!(v, F.τ[1], M)
    reflectorApply!(v, F.τ[1], M')
    dv[1] = M[1,1]
        # fill second entry
    # computes H*M*H in-place, overwriting M
    v = view(F.factors,2:b,2)
    reflectorApply!(v, F.τ[2], view(M,1,2:b))
    M[1,2:b] .= view(M,1,2:b) # symmetric matrix, avoid recomputation
    reflectorApply!(v, F.τ[2], view(M,2:b,2:b))
    reflectorApply!(v, F.τ[2], view(M,2:b,2:b)')
    ev[1] = M[1,2]*sign(F.R[1,1]*F.R[2,2]) # includes possible correction for sign (only needed in off-diagonal case), since the QR decomposition does not guarantee positive diagonal on R
    K = Matrix(X[2:b+1,2:b+1])
    K[1:end-1,1:end-1] .= view(M,2:b,2:b)
    return QRJacobiData{:Q,T}(dv, ev, F, K, P, 1)
end
function QRJacobiData{:R,T}(F, P) where T
    U = F.R
    U = ApplyArray(*,Diagonal(sign.(view(U,band(0)))),U)  # QR decomposition does not force positive diagonals on R by default
    X = jacobimatrix(P)
    UX = cache(simplify(Mul(U,X)))
    # compute a length 2 vector on first go and circumvent BigFloat issue
    dv = zeros(T,2) 
    ev = zeros(T,2)
    dv[1] = UX[1,1]/U[1,1] # this is dot(view(UX,1,1), U[1,1] \ [one(T)])
    dv[2] = -U[1,2]*UX[2,1]/(U[1,1]*U[2,2])+UX[2,2]/U[2,2] # this is dot(view(UX,2,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[1] = -UX[1,1]*U[1,2]/(U[1,1]*U[2,2])+UX[1,2]/U[2,2] # this is dot(view(UX,1,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[2] = UX[2,1]/U[3,3]*(-U[1,3]/U[1,1]+U[1,2]*U[2,3]/(U[1,1]*U[2,2]))+UX[2,2]/U[3,3]*(-U[2,3]/U[2,2])+UX[2,3]/U[3,3] # this is dot(view(UX,2,1:3), U[1:3,1:3] \ [zeros(T,2); one(T)])
    return QRJacobiData{:R,T}(dv, ev, U, UX, P, 2)
end


size(::QRJacobiData) = (ℵ₀,2) # Stored as two infinite cached bands

function getindex(K::QRJacobiData, n::Integer, m::Integer)
    @assert (m==1) || (m==2)
    resizedata!(K,n,m)
    m == 1 && return K.dv[n]
    m == 2 && return K.ev[n]
end

function getindex(C::SymTridiagonal{<:Any, <:SubArray{<:Any, 1, <:QRJacobiData, <:Tuple, false}, <:SubArray{<:Any, 1, <:QRJacobiData, <:Tuple, false}}, kr::UnitRange, jr::UnitRange)
    m = maximum(max(kr,jr))+1
    resizedata!(C.dv.parent,m,2)
    resizedata!(C.ev.parent,m,2)
    return copy(view(C,kr,jr))
end

# Resize and filling functions for cached implementation
function resizedata!(K::QRJacobiData, n::Integer, m::Integer)
    nm = max(n,m)
    νμ = K.datasize
    if nm > νμ
        resize!(K.dv,nm)
        resize!(K.ev,nm)
        _fillqrbanddata!(K, νμ:nm)
        K.datasize = nm
    end
    K
end
function _fillqrbanddata!(J::QRJacobiData{:Q,T}, inds::UnitRange{Int}) where T
    b = bandwidths(J.U.factors)[2]÷2
    # pre-fill cached arrays to avoid excessive cost from expansion in loop
    m, jj = 1+inds[end], inds[2:end]
    X = jacobimatrix(J.P)[1:m+b+2,1:m+b+2]
    resizedata!(J.U.factors,m+b,m+b)
    resizedata!(J.U.τ,m)
    K, τ, F, dv, ev = J.UX, J.U.τ, J.U.factors, J.dv, J.ev
    D = sign.(view(J.U.R,band(0)).*view(J.U.R,band(0))[2:end])
    M = zeros(T,b+3,b+3)
    if isprimitivetype(T)
        M = Matrix{T}(undef,b+3,b+3) 
    else
        M = zeros(T,b+3,b+3)
    end
    @inbounds for n in jj
        dv[n] = K[1,1] # no sign correction needed on diagonal entry due to cancellation
        # doublehouseholderapply!(K,τ[n+1],view(F,n+2:n+b+2,n+1),w)
        v = view(F,n+1:n+b+2,n+1)
        reflectorApply!(v, τ[n+1], view(K,1,2:b+3))
        M[1,2:b+3] .= view(M,1,2:b+3) # symmetric matrix, avoid recomputation
        reflectorApply!(v, τ[n+1], view(K,2:b+3,2:b+3))
        reflectorApply!(v, τ[n+1], view(K,2:b+3,2:b+3)')
        ev[n] = K[1,2]*D[n] # contains sign correction from QR not forcing positive diagonals
        M .= view(X,n+1:n+b+3,n+1:n+b+3)
        M[1:end-1,1:end-1] .= view(K,2:b+3,2:b+3)
        K .= M
    end
end

function _fillqrbanddata!(J::QRJacobiData{:R,T}, inds::UnitRange{Int}) where T
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    m = inds[end]+1
    resizedata!(J.U,m,m)
    resizedata!(J.UX,m,m)

    dv, ev, UX, U = J.dv, J.ev, J.UX, J.U
    @inbounds Threads.@threads for k in inds
        dv[k] = -U[k-1,k]*UX[k,k-1]/(U[k-1,k-1]*U[k,k])+UX[k,k]./U[k,k] # this is dot(view(UX,k,k-1:k), U[k-1:k,k-1:k] \ ek)
        ev[k] = UX[k,k-1]/U[k+1,k+1]*(-U[k-1,k+1]/U[k-1,k-1]+U[k-1,k]*U[k,k+1]/(U[k-1,k-1]*U[k,k]))+UX[k,k]/U[k+1,k+1]*(-U[k,k+1]/U[k,k])+UX[k,k+1]/U[k+1,k+1]  # this is dot(view(UX,k,k-1:k+1), U[k-1:k+1,k-1:k+1] \ ek)
    end
end
