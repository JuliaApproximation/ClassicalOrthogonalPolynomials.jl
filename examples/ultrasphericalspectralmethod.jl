using ClassicalOrthogonalPolynomials, Plots

T = Chebyshev()
C = Ultraspherical(2)
D = Derivative(axes(T,1))
A = (C\(D^2*T))+100(C\T)

c = Vcat(T[[-1,1],:], A) \ [1;2;zeros(∞)]
u = T*c

plot(u)