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

g = range(-1,1,100)
plot(g, diff(W)[g,1:4])

# We can get out a differentiation matrix via

P = Legendre()
D_W = P \ diff(W)

# If we wanted strong-form we hit a missing case for weighted differentiation.
# But we can differentiate `P` directly:


D_P = C \ diff(P)
D² = D_P * D_W

# We can also do a conversion matrix via:

R = (C\P)  * (P \ W)
X = jacobimatrix(C)
L = D² - X*R

# We can solve a strong form ODE via

c = L \ transform(C, exp)
u = W*c
plot(g, u[g])


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

(W' * P) *  transform(P, exp)



