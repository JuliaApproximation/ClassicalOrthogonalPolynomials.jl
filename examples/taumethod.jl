using ClassicalOrthogonalPolynomials

T = ChebyshevT()
C⁴ = Ultraspherical(4)

n = 20 # truncation size
R  = (C⁴\T)[1:n,1:n] # discretization of I
D¹ = (C⁴\diff(T))[1:n,1:n]
D⁴ = (C⁴\diff(T,4))[1:n,1:n]
γ₀ = T[[begin,end],1:n]
γ₁ = diff(T)[[begin,end],1:n]

φ₁ = (C⁴\T)[1:n,n-1]
φ₂ = (C⁴\T)[1:n,n]

Z = zeros(2,n)
𝐳₂ = zeros(2)
𝐳 = zeros(n)

A = [γ₀     Z       Z       𝐳₂   𝐳₂;
     Z      γ₀      Z       𝐳₂   𝐳₂;
     γ₁     Z       Z       𝐳₂   𝐳₂;
     Z      γ₁      Z       𝐳₂   𝐳₂;
     R      -D¹     0I      𝐳   𝐳;
     D⁴     0I      R       𝐳   𝐳;
     0I     D⁴      D¹      φ₁  φ₂]