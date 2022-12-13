

using ClassicalOrthogonalPolynomials, Plots

ρ = 0.5
T,U = chebyshevt(ρ..1),chebyshevu(T); C = ultraspherical(2, ρ..1); r = axes(T,1); D = Derivative(r);

L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity

m = 2
Δₘ = L - m^2*M # r^2 * Laplacian for exp(im*m*θ)*u(r), i.e. (r^2 * ∂^2 + r*∂ - m^2*I)*u



# Poisson solve
c = C \ exp.(r)
R = C \ (r .* C)


d = [T[[begin,end],:];
            Δₘ] \ [1; 2; R^2 * c]

plot(T*d)


# Helmholtz


Q = R^2 * M # r^2, needed for Helmholtz (Δ + k^2)*u = f

k = 5 # f

d = [T[[begin,end],:];
            Δₘ+k^2*Q] \ [1; 2; R^2 * c]


# transform

f = (r,θ) -> exp(r*cos(θ)+sin(θ))
T,F = chebyshevt(ρ..1),Fourier()
n = 1000 # create a 1000 x 1000 transform
𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, n),ClassicalOrthogonalPolynomials.grid(F, n)
PT,PF = plan_transform(T, (n,n), 1),plan_transform(F, (n,n), 2)
 
@time X = PT * (PF * f.(𝐫, 𝛉'))

@test T[0.1,1:n]'*X*F[0.2,1:n] ≈ f(0.1,0.2)

