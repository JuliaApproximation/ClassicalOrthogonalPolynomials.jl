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

f = (C \T)*(T\(@.(x*exp(x^2))))
ε = 1/100
A = [T[[begin,end],:]; C \ ((ε*D^2 - x .* D + I) * T)]
b = [0; 0; f]
c = A \ b
u = T*c

u[1/2]


# odd
T = ChebyshevT{BigFloat}()
C = Ultraspherical{BigFloat}(2)
x = axes(T,1)
D = Derivative(x)

f = (C \T)*(T\(@.(x*exp(x^2))))
ε = big(1)/100
# c = [T[[begin,end],:]; C \ ((ε*D^2 - x .* D + I) * T)] \ [0; 0; f]

A = [T[[begin,end],:]; C \ ((ε*D^2 - x .* D + I) * T)]
b = [0; 0; f]
n = 10_000; c = [A[1:n,1:n] \ b[1:n]; zeros(BigFloat,∞)]
u = T*c

let x = big(1)/2
    u[x], -2ε^(3/2) * exp(1/2ε)*sqrt(convert(typeof(x),π)/2) * (x + exp(-(x+1)/ε) - exp((x-1)/ε))
end

u[big(1)/2] - (-5.20717919984334519032651253279531326480746523413400676407697947991776188139192)

plot((T * (T \ (@. -2ε^(3/2) * exp(1/2ε)*sqrt(π/2) * (x + exp(-(x+1)/ε) - exp((x-1)/ε))))))
plot!(u)
D


u[0.1]


# even
f = (x.^2 .* exp.(x.^2))
χ = -sum(T * (T\ (f./x.^2)))/2

diff(f)

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