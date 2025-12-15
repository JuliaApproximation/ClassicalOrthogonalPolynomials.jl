####
# solve
#
# -u'' + u^2 = 1 + x
# u(±1) = 0
# 
# using integrated Legendre w/ Newton iteration.
###

using ClassicalOrthogonalPolynomials, Plots

P = Legendre(); C = Ultraspherical(3/2); W = Weighted(C)
x = axes(P,1)
Δ = diff(W)'diff(W)
M = W'W
b = W'*(1 .+ x)
F = u -> Δ*u + M * (W\((W*u) .^ 2)) - b
J = u -> Δ + 2*M * (W \ ((P * (P\(W*u))) .* W))


u = zeros(∞)
while norm(F(u)) ≥ 1E-12
    u -= J(u) \ F(u)
end


plot(W*u)
