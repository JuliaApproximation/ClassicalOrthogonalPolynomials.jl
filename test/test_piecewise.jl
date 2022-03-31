using ClassicalOrthogonalPolynomials, Test



using LazyArrays, LazyBandedMatrices, BlockArrays, FillArrays
T = Float64

import BlockBandedMatrices: _BandedBlockBandedMatrix



plot(C[:,1:4])

xx = range(0,1;length=100)
plot(xx,P[xx,1:6])
plot(xx,C[xx,1:6])
C[1,1:10]

x = axes(C,1)
C \ x