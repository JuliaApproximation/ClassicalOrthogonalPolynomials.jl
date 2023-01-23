using ClassicalOrthogonalPolynomials, Plots
plotly()

###
# Laplace's equation
###

ρ = 0.5
T,C,F = chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
r = axes(T,1)
D = Derivative(r)

L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity
R = C \ (r .* C) # mult by r

uᵨ = transform(F, θ -> exp(cos(θ)+sin(θ-1)))
u₁ = transform(F, θ -> cos(100cos(θ)-sin(θ+1)-1))

n = 300
X = zeros(n,n)

for j = 1:n
    m = j ÷ 2
    Δₘ = L - m^2*M # r^2 * Laplacian for exp(im*m*θ)*u(r), i.e. (r^2 * ∂^2 + r*∂ - m^2*I)*u
    d = [T[[begin,end],:]; Δₘ] \ [uᵨ[j]; u₁[j]; zeros(∞)]
    X[:,j] .= d[1:n]
end

𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, n),ClassicalOrthogonalPolynomials.grid(F, n)
PT,PF = plan_transform(T, (n,n), 1),plan_transform(F, (n,n), 2)
U = PT \ (PF \ X); U = [U U[:,1]]

𝐱 = 𝐫 .* cos.([𝛉; 2π]')
𝐲 = 𝐫 .* sin.([𝛉; 2π]')
surface(𝐱, 𝐲, U; zlims=(-2,2))


###
# Helmholtz equation
###

uᵨ = transform(F, θ -> exp(cos(θ)+sin(θ-1)))
u₁ = transform(F, θ -> cos(cos(θ)-sin(θ+1)-1))


k = 100

for j = 1:n
    m = j ÷ 2
    Δₘ = L - m^2*M + k^2 * R^2*M
    d = [T[[begin,end],:]; Δₘ] \ [uᵨ[j]; u₁[j]; zeros(∞)]
    X[:,j] .= d[1:n]
end

U = PT \ (PF \ X); U = [U U[:,1]]
surface(𝐱, 𝐲, U; zlims=(-10,10))


###
# Poisson
###

n = 1000
𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, n),ClassicalOrthogonalPolynomials.grid(F, n)
PT,PF = plan_transform(T, (n,n), 1),plan_transform(F, (n,n), 2)
f = (x,y) -> exp(-4000((x-0.8)^2 + (y-0.1)^2))
𝐱 = 𝐫 .* cos.(𝛉')
𝐲 = 𝐫 .* sin.(𝛉')

F = PT * (PF * f.(𝐱, 𝐲))



X = zeros(n+2, n+2)

# multiply RHS by r^2 and convert to C
S = (R^2*M)[1:n,1:n]

for j = 1:n
    m = j ÷ 2
    Δₘ = L - m^2*M
    X[:,j] = [T[[begin,end],:]; Δₘ][1:n+2,1:n+2] \ [0; 0; S*F[:,j]]
end

U = PT \ (PF \ X[1:n,1:n]); U = [U U[:,1]]
𝐱 = 𝐫 .* cos.([𝛉; 2π]')
𝐲 = 𝐫 .* sin.([𝛉; 2π]')
surface(𝐱, 𝐲, U)
