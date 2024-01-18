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

# Wachter law
a,b = 5,10
c,d = sqrt(a/(a+b) * (1-1/(a+b))), sqrt(1/(a+b) * (1-a/(a+b)))
lmin,lmax = (c-d)^2,(c+d)^2
U = chebyshevu(lmin..lmax)
x = axes(U,1)
w = @. (a+b) * sqrt((x-lmin)*(lmax-x))/(2π*x*(1-x))
Q = LanczosPolynomial(w, U)
plot(Q[:,1:7])