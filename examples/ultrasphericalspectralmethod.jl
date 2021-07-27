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


##
# Lee & Greengard
##

T = ChebyshevT()
C = Ultraspherical(2)
x = axes(T,1)
D = Derivative(x)

ε = 0.5

c = [T[[begin,end],:]; C \ (ε*D^2 * T - x .* (D* T) + T)] \ [1; 2; zeros(∞)]
u = T*c
plot(u)

# odd
f = T \ (x .* exp.(x.^2))
c = [T[[begin,end],:]; C \ (ε*D^2 * T - x .* (D* T) + T)] \ [0; 0; f]
u = T*c
plot(u)

import ForwardDiff: derivative


plot((T * (T \ (@. -2ε^(3/2) * exp(1/2ε)*sqrt(π/2) * (x + exp(-(x+1)/ε) - exp((x-1)/ε))))))
plot!(u)
D

let x = 0.1
    2ε^(3/2) * (D*T*f)[0] * exp(1/2ε)*sqrt(π/2) * (x + exp(-(x+1)/ε) - exp((x-1)/ε))
end
u[0.1]


# even
f = (x.^2 .* exp.(x.^2))
χ = -sum(T * (T\ (f./x.^2)))/2

A = randn(5,5)
T \ cumsum(T; dims=1)
U = ChebyshevU()
x = axes(T,1)
D = Derivative(x)
(U \ (D * T))[:,2:end] \ (U \ T)

using LazyArrays, FillArrays, Infinities
import BandedMatrices: _BandedMatrix
V = Float64




U \ U


sum(T * (T\ (f./x.^2)))


(T\ (f./x.^2))



Legendre() \ 

@ent Chebyshev() \ Legendre()