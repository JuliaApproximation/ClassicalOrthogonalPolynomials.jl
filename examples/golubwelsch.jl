using ClassicalOrthogonalPolynomials, FastGaussQuadrature

P = Legendre()
x = axes(P,1)
x .* P