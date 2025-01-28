using ClassicalOrthogonalPolynomials, FastTransforms

P = Legendre()
x = axes(P,1)
w = @.(1-x^2)

μ = P'w
X = jacobimatrix(P)
n = 20
@test GramMatrix(μ[1:2n], X[1:2n,1:2n]) ≈ (P' * (w .* P))[1:n,1:n]