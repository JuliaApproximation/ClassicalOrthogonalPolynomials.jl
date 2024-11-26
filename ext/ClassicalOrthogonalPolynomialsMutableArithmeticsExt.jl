module ClassicalOrthogonalPolynomialsMutableArithmeticsExt
using ClassicalOrthogonalPolynomials, MutableArithmetics
import ClassicalOrthogonalPolynomials: initiateforwardrecurrence, recurrencecoefficients, _p0

Base.unsafe_getindex(P::OrthogonalPolynomial, x::AbstractMutable, n::Number) = initiateforwardrecurrence(n-1, recurrencecoefficients(P)..., x, _p0(P))[end]

recurrencecoefficients(::Legendre{T}) where T<:AbstractMutable = recurrencecoefficients(Legendre())
end