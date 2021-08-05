using ClassicalOrthogonalPolynomials, Plots, Test
import ArrayLayouts: diagonal

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

c = [T[[begin,end],:]; C \ ((D^2 - diagonal(x))*T)] \ [airyai(-1); airyai(1); zeros(∞)]
u = T*c

@test u[0.0] ≈ airyai(0.0)
plot(u)


##
# Lee & Greengard
# ε*u'' - x*u' + u = 0, u(-1) = 1, u(1) = 2
##

T = ChebyshevT()
C = Ultraspherical(2)
x = axes(T,1)
D = Derivative(x)

ε = 1/100
A = [T[[begin,end],:]; C \ ((ε*D^2 - x .* D + I) * T)]
c = A \ [1; 2; zeros(∞)]
u = T*c
plot(u)

C \ (ε*D^2 - x .* D + I) 