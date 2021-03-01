"""
    OrthogonalPolynomialRatio(P,x)

is a is equivalent to the vector `P[x,:] ./ P[x,2:end]`` but built from the
recurrence coefficients of `P`.
"""
mutable struct OrthogonalPolynomialRatio{T, PP<:AbstractQuasiMatrix{T}} <: AbstractCachedVector{T}
    P::PP # OPs
    x::T
    data::Vector{T}
    datasize::Tuple{Int}

    function OrthogonalPolynomialRatio{T, PP}(P::PP, x::T) where {T,PP<:AbstractQuasiMatrix{T}}
        μ = inv(sqrt(sum(orthogonalityweight(P))))
        new{T, PP}(P, x, [Base.unsafe_getindex(P,x,1)/Base.unsafe_getindex(P,x,2)], (1,))
    end
end

OrthogonalPolynomialRatio(P::AbstractQuasiMatrix{T}, x) where T = OrthogonalPolynomialRatio{T,typeof(P)}(P, convert(T, x))

size(K::OrthogonalPolynomialRatio) = (ℵ₀,)


function LazyArrays.cache_filldata!(R::OrthogonalPolynomialRatio, inds)
    A,B,C = recurrencecoefficients(R.P)
    x = R.x
    data = R.data
    @inbounds for n in inds
        data[n] = inv(A[n]*x + B[n] - C[n] * data[n-1])
    end

end
