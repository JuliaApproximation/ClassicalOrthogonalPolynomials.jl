using ClassicalOrthogonalPolynomials, Plots

##
#  Ideal fluid flow consists of level sets of the imagainary part of a function
# that is asymptotic to c*z and whose imaginary part vanishes on Γ
#
#
# On the unit interval, -2*H*u gives the imaginary part of cauchy(u,x)
#  So if we want to find u defined on Γ so that hilbert(u,x) = imag(c*x)
#  then c*z + 2cauchy(u,z) vanishes on Γ
##

T = ChebyshevT()
U = ChebyshevU()
x = axes(U,1)
H = inv.(x .- x')

c = exp(0.5im)


u = Weighted(U) * ((H * Weighted(U)) \ imag(c * x))

ε  = eps(); (inv.(0.1+ε*im .- x') * u + inv.(0.1-ε*im .- x') * u)/2 ≈ imag(c*0.1)
ε  = eps(); real(inv.(0.1+ε*im .- x') * u ) ≈ imag(c*0.1)

v = (s,t) -> (z = (s + im*t); imag(c*z) - real(inv.(z .- x') * u))


xx = range(-3,3; length=100)
yy = range(-1,1; length=100)
plot([-1,1],[0,0]; color=:black)
contour!(xx,yy,v.(xx',yy))