using ClassicalOrthogonalPolynomials, Test
import ClassicalOrthogonalPolynomials: ContinuousPolynomial, PiecewisePolynomial


using LazyArrays, LazyBandedMatrices, BlockArrays, FillArrays
T = Float64

import BlockBandedMatrices: _BandedBlockBandedMatrix

P = PiecewisePolynomial(Legendre(), range(0,1;length=4))
C = ContinuousPolynomial{1}(range(0,1;length=4))

P \ C

plot(C[:,1:4])

xx = range(0,1;length=100)
plot(xx,P[xx,1:6])
plot(xx,C[xx,1:6])
C[1,1:10]

x = axes(C,1)
C \ x