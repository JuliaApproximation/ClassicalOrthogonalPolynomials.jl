using ClassicalOrthogonalPolynomials, Plots

# MP law
r = 0.5
lmin, lmax = (1-sqrt(r))^2,  (1+sqrt(r))^2
U = chebyshevu(lmin..lmax)
x = axes(U,1)
w = @. 1/(2π) * sqrt((lmax-x)*(x-lmin))/(x*r)

# Q is a quasimatrix such that Q[x,k+1] is equivalent to
# qₖ(x), the k-th orthogonal polynomial wrt to w
Q = LanczosPolynomial(w, U)

# The Jacobi matrix associated with Q, as an ∞×∞ SymTridiagonal
J = jacobimatrix(Q)

# plot q₀,…,q₆
plot(Q[:,1:7])

