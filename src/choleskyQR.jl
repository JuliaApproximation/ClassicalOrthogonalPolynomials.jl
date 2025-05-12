"""
Represents orthonormal polynomials defined via a conversion operator from P, that is, Q = P * inv(U).
"""
struct ConvertedOrthogonalPolynomial{T, WW<:AbstractQuasiVector{T}, XX, UU, PP} <: OrthonormalPolynomial{T}
    weight::WW
    X::XX # jacobimatrix
    U::UU # conversion to P
    P::PP
end

_p0(Q::ConvertedOrthogonalPolynomial) = _p0(Q.P)/Q.U[1,1]

axes(Q::ConvertedOrthogonalPolynomial) = axes(Q.P)


struct ConvertedOPLayout <: AbstractNormalizedOPLayout end
MemoryLayout(::Type{<:ConvertedOrthogonalPolynomial}) = ConvertedOPLayout()




jacobimatrix(Q::ConvertedOrthogonalPolynomial) = Q.X
orthogonalityweight(Q::ConvertedOrthogonalPolynomial) = Q.weight


# transform to P * inv(U) if needed for differentiation, etc.
arguments(::ApplyLayout{typeof(*)}, Q::ConvertedOrthogonalPolynomial) = Q.P, ApplyArray(inv, Q.U)

OrthogonalPolynomial(w::AbstractQuasiVector) = OrthogonalPolynomial(w, orthogonalpolynomial(axes(w,1), singularities(w)))
function OrthogonalPolynomial(w::AbstractQuasiVector, P::AbstractQuasiMatrix)
    Q = normalized(P)
    X,U = cholesky_jacobimatrix(w, Q)
    ConvertedOrthogonalPolynomial(w, X, U, Q)
end

orthogonalpolynomial(wP...) = OrthogonalPolynomial(wP...)
orthogonalpolynomial(w::SubQuasiArray) = orthogonalpolynomial(parent(w))[parentindices(w)[1],:]

OrthogonalPolynomial(w::Function, P::AbstractQuasiMatrix) = OrthogonalPolynomial(w.(axes(P,1)), P)
orthogonalpolynomial(w::Function, P::AbstractQuasiMatrix) = orthogonalpolynomial(w.(axes(P,1)), P)

orthogonalpolynomial(ax, ::NoSingularities) = legendre(ax)
orthogonalpolynomial(ax, w) = orthogonalpolynomial(w)
resizedata!(P::ConvertedOrthogonalPolynomial, ::Colon, n::Int) = resizedata!(P.X.dv, n)


"""
cholesky_jacobimatrix(w, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x)` by computing a Cholesky decomposition of the weight modification.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs are `w` as a function or `W` as an infinite matrix representing the weight modifier multiplication by the function `w / w_P` on `P` where `w_P` is the orthogonality weight of `P`.
"""
cholesky_jacobimatrix(w::Function, P) = cholesky_jacobimatrix(w.(axes(P,1)), P)

function cholesky_jacobimatrix(w::AbstractQuasiVector, P::AbstractQuasiMatrix)
    Q = normalized(P)
    w_P = orthogonalityweight(P)
    W = Symmetric(Q \ ((w ./ w_P) .* Q)) # Compute weight multiplication via Clenshaw
    return cholesky_jacobimatrix(W, jacobimatrix(Q))
end
function cholesky_jacobimatrix(W::AbstractMatrix, X::AbstractMatrix)
    U = cholesky(W).U
    return SymTridiagonalConjugation(U, X), U
end


"""
qr_jacobimatrix(w, P)

returns the Jacobi matrix `X` associated to a quasi-matrix of polynomials
orthogonal with respect to `w(x)` by computing a QR decomposition of the square root weight modification.

The resulting polynomials are orthonormal on the same domain as `P`. The supplied `P` must be normalized. Accepted inputs for `w` are the target weight as a function or `sqrtW`, representing the multiplication operator of square root weight modification on the basis `P`.

The underlying QR approach allows two methods, one which uses the Q matrix and one which uses the R matrix. To change between methods, an optional argument :Q or :R may be supplied. The default is to use the Q method.
"""
qr_jacobimatrix(w::Function, P) = qr_jacobimatrix(w.(axes(P,1)), P)
function qr_jacobimatrix(w::AbstractQuasiVector, P)
    Q = normalized(P)
    w_P = orthogonalityweight(P)
    sqrtW = Symmetric(Q \ (sqrt.(w ./ w_P) .* Q)) # Compute weight multiplication via Clenshaw
    return qr_jacobimatrix(sqrtW, jacobimatrix(Q))
end
function qr_jacobimatrix(sqrtW::AbstractMatrix{T}, X::AbstractMatrix) where T
    R = qr(sqrtW).R
    return SymTridiagonalConjugation(R, X), R
end
