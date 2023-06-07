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
    CJD = CholeskyJacobiData(U,UX)
    return SymTridiagonal(view(CJD,:,1),view(CJD,:,2))
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in two cached bands which are generated in tandem but can be accessed separately.
mutable struct CholeskyJacobiData{T} <: AbstractMatrix{T}
    dv::AbstractVector{T} # store diagonal band entries in adaptively sized vector
    ev::AbstractVector{T} # store off-diagonal band entries in adaptively sized vector
    U::UpperTriangular{T} # store upper triangular conversion matrix (needed to extend available entries)
    UX::ApplyArray{T}     # store U*X, where X is the Jacobi matrix of the original P (needed to extend available entries)
    datasize::Int         # size of so-far computed block 
end

# Computes the initial data for the Jacobi operator bands
function CholeskyJacobiData(U::AbstractMatrix{T}, UX) where T
    dv = Vector{T}(undef,2) # compute a length 2 vector on first go
    ev = Vector{T}(undef,2)
    dv[1] = UX[1,1]/U[1,1] # this is dot(view(UX,1,1), U[1,1] \ [one(T)])
    dv[2] = -U[1,2]*UX[2,1]/(U[1,1]*U[2,2])+UX[2,2]/U[2,2] # this is dot(view(UX,2,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[1] = -UX[1,1]*U[1,2]/(U[1,1]*U[2,2])+UX[1,2]/U[2,2] # this is dot(view(UX,1,1:2), U[1:2,1:2] \ [zero(T); one(T)])
    ev[2] = UX[2,1]/U[3,3]*(-U[1,3]/U[1,1]+U[1,2]*U[2,3]/(U[1,1]*U[2,2]))+UX[2,2]/U[3,3]*(-U[2,3]/U[2,2])+UX[2,3]/U[3,3] # this is dot(view(UX,2,1:3), U[1:3,1:3] \ [zeros(T,2); one(T)])[1:3,1:3] \ [zeros(T,2); one(T)])
    return CholeskyJacobiData{T}(dv, ev, U, UX, 2)
end

size(::CholeskyJacobiData) = (ℵ₀,2) # Stored as two infinite cached bands

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
    # pre-fill U and UX to prevent expensive step-by-step filling in of cached U and UX in the loop
    resizedata!(J.U,inds[end]+1,inds[end]+1)
    resizedata!(J.UX,inds[end]+1,inds[end]+1)

    dv, ev, UX, U = J.dv, J.ev, J.UX, J.U
    @inbounds for k in inds
        # this is dot(view(UX,k,k-1:k), U[k-1:k,k-1:k] \ ek)
        dv[k] = -U[k-1,k]*UX[k,k-1]/(U[k-1,k-1]*U[k,k])+UX[k,k]/U[k,k]
        ev[k] = UX[k,k-1]/U[k+1,k+1]*(-U[k-1,k+1]/U[k-1,k-1]+U[k-1,k]*U[k,k+1]/(U[k-1,k-1]*U[k,k]))+UX[k,k]/U[k+1,k+1]*(-U[k,k+1]/U[k,k])+UX[k,k+1]/U[k+1,k+1]  
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
function qr_jacobimatrix(sqrtW::AbstractMatrix{T}, Q, method = :Q) where T
    isnormalized(Q) || error("Polynomials must be orthonormal")
    F = qr(sqrtW)
    QRJD = QRJacobiData{method,T}(F,Q)
    SymTridiagonal(view(QRJD,:,1),view(QRJD,:,2))
end

# The generated Jacobi operators are symmetric tridiagonal, so we store their data in two cached bands which are generated in tandem but can be accessed separately.
mutable struct QRJacobiData{method,T} <: AbstractMatrix{T}
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
        # we fill 1 entry on the first run
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
    reflectorApply!(v, F.τ[2], view(M,2:b,1))
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
    UX = ApplyArray(*,U,X)
    dv = Vector{T}(undef,2) # compute a length 2 vector on first go
    ev = Vector{T}(undef,2)
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
    M = Matrix{T}(undef,b+3,b+3)
    @inbounds for n in jj
        dv[n] = K[1,1] # no sign correction needed on diagonal entry due to cancellation
        # doublehouseholderapply!(K,τ[n+1],view(F,n+2:n+b+2,n+1),w)
        v = view(F,n+1:n+b+2,n+1)
        reflectorApply!(v, τ[n+1], view(K,1,2:b+3))
        reflectorApply!(v, τ[n+1], view(K,2:b+3,1))
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
    @inbounds for k in inds
        dv[k] = -U[k-1,k]*UX[k,k-1]/(U[k-1,k-1]*U[k,k])+UX[k,k]./U[k,k] # this is dot(view(UX,k,k-1:k), U[k-1:k,k-1:k] \ ek)
        ev[k] = UX[k,k-1]/U[k+1,k+1]*(-U[k-1,k+1]/U[k-1,k-1]+U[k-1,k]*U[k,k+1]/(U[k-1,k-1]*U[k,k]))+UX[k,k]/U[k+1,k+1]*(-U[k,k+1]/U[k,k])+UX[k,k+1]/U[k+1,k+1]  # this is dot(view(UX,k,k-1:k+1), U[k-1:k+1,k-1:k+1] \ ek)
    end
end