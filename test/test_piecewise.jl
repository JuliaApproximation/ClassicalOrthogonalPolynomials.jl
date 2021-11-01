using ClassicalOrthogonalPolynomials, Test
import ClassicalOrthogonalPolynomials: PiecewisePolynomial

P = PiecewisePolynomial{1}(range(0,1;length=4))

xx = range(0,1;length=100)
P[xx,1:10]
P[1,1:10]