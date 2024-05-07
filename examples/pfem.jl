using ClassicalOrthogonalPolynomials, Plots, Test


#
# We can solve ODEs like an inhomogenous Airy equation
# $$
# u'' - x*u = f
# $$
# with zero Dirichlet conditions using a basis built from ultraspherical polynomials
# of the form
# $$
# (1-x^2)*C_n^{(3/2)}(x) = c_n (1-x^2) P_n^{(1,1)}(x)
# $$
# We construct this basis as follows:

C = Ultraspherical(3/2)
W = Weighted(C)
plot(W[:,1:4])

# We can differentiate using the `diff` command. Unlike arrays, for quasi-arrays `diff(W)` defaults to
# `diff(W; dims=1)`. 

plot(diff(W)[:,1:4])

# We can get out a differentiation matrix via

D² = C \ diff(diff(W))

# We can construct the multiplication by $x$ with the connection between `W` and `C`
# via:

x = axes(W,1)
X = C \ (x .* W)

# Thus our operator becomes:

L = D² - X

# We can compute the coefficients using:

c = L \ transform(C, exp)
u = W*c
plot(u)


# Weak form for the Laplacian with W as a basis for test/trial  is given by
# $$
# ⟨dv/dx, du/dx ⟩ ≅ diff(v)'diff(u) = diff(W*d)'*diff(W*c) == d'*diff(W)'diff(W)*c
# $$
# where $v = W*d$ for some vector of coeficients d.  For $x$ we have
# $$
# ⟨v, x*u ⟩ ≅ v'(x .* u) == d'*W'(x .* W)*c
# $$

Δ  = -(diff(W)'diff(W)) # or weaklaplacian(W)
x = axes(W,1)
X = W' * (x .* W)
L = Δ - X

# We can solve an ODE via:

c = L \ W'exp.(x)
u = W*c
plot!(u)


# The two solvers match!
