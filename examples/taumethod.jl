using ClassicalOrthogonalPolynomials

T = ChebyshevT()
C⁴ = Ultraspherical(4)

n = 10 # truncation size
D¹ = (C⁴\diff(T))[1:n,1:n]
D⁴ = (C⁴\diff(T,4))[1:n,1:n]
γ₀ = T[[begin,end],1:n]
γ₁ = diff(T)[[begin,end],1:n]

z = zeros(2,n)

A = [γ₀     z       z;
     z      γ₀      z;
     γ₁     z       z;
     z      γ₁      z;
     I      -D¹     0I;
     D⁴     0I      I;
     0I     D⁴      D¹]