using ClassicalOrthogonalPolynomials, Plots

ρ = 0.5; T = chebyshevt(ρ..1); U = chebyshevu(T); C = ultraspherical(2, ρ..1); r = axes(T,1); D = Derivative(r);

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

f = (r,θ) -> exp(r*cos(θ))
n = 10 # create a 10 x 10 transform
T,F = Chebyshev(),Fourier()
𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, n),ClassicalOrthogonalPolynomials.grid(F, n)
PF = plan_transform(F, (n,n), 2)

transform(T, transform(F, f.(𝐫, 𝛉'); dims=2); dims=1)
