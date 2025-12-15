####
# solve
#
# -u'' + u^2 = 1
# u(±1) = 0
# 
# using integrated Legendre w/ Newton iteration.
###

using ClassicalOrthogonalPolynomials, Plots

P = Legendre()
C = Ultraspherical(3/2)
W = Weighted(C)
Δ = -diff(W)'diff(W)
M = W'W
b = W'one(axes(W,1))
F = u -> Δ*u + M * (W\((W*u) .^ 2)) - b
J = u -> Δ + M * (W \ ((P * (P\(W*u))) .* W))


u = b
while norm(F(u)) ≥ 1E-12
    u -= J(u) \ F(u)
end

plot(W*u)