using ClassicalOrthogonalPolynomials, Plots, Test

###
# We can solve ODEs like the Airy equation
#
# u(-1) = airyai(-1)
# u(1) = airyai(1)
# u'' = x * u
#
# using the ultraspherical spectral method. 

T = Chebyshev()
C = Ultraspherical(2)
x = axes(T,1)
D = Derivative(x)

c = [T[[begin,end],:]; C \ (D^2 * T - x .* T)] \ [airyai(-1); airyai(1); zeros(∞)]
u = T*c

@test u[0.0] ≈ airyai(0.0)
plot(u)