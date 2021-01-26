using ContinuumArrays, QuasiArrays, FillArrays, InfiniteArrays
import QuasiArrays: Inclusion, QuasiDiagonal

using Plots; pyplot();

S = LinearSpline(range(0,1; length=10))
xx = range(0,1; length=1000)

S = Jacobi(true,true)


P = Legendre()
X = QuasiDiagonal(Inclusion(-1..1))

@test X[-1:0.1:1,-1:0.1:1] == Diagonal(-1:0.1:1)

axes(X)
J = pinv(P)*X*P

J - I
Vcat(Hcat(1, Zeros(1,∞)), J)
