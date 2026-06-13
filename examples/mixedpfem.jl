using ContinuumArrays, ClassicalOrthogonalPolynomials, FillArrays, Plots

#####
# Poisson
#  -Δu = f
#   𝐮 := ∇u
# Strong form:
#    <𝐯, ∇u> - <𝐯, 𝐮> = 0
#    -<v, ∇⋅𝐮> = <v,f>
# Mixed form:
#    <𝐯, 𝐮> + <∇⋅𝐯,u> = 0
#    <v, ∇⋅𝐮>         = -<v,f>
######


C = Ultraspherical(-1/2)
P = Legendre()

n = 100
D = (P'diff(Q))[1:n-1,1:n]
M = (Q'Q)[1:n,1:n]
Z = Zeros(n-1,n-1)

u = expand(P, x -> cos(π/2*x))
λ = -π^2/4
@test diff(u,2) ≈ λ*u
v = diff(u)
𝐜 = [(C\v)[1:n]; (P\u)[1:n-1]]
A = [M D'; D Z]
B = [Zeros(n,n) Zeros(n,n-1); Zeros(n-1,n) (P'P)[1:n-1,1:n-1]]
@test A*𝐜 ≈ λ*B*𝐜

λ,Q = eigen(A, B)
k = findmin(abs.(λ))[2]
ũ = P[:,1:n-1]* Q[n+1:end,k]
plot(ũ)

@test ũ/ũ[0] ≈ u



W = Weighted(Ultraspherical(3/2))
P = Legendre()

n = 100
D = (P'diff(W))[1:n+1,1:n]
M = (W'W)[1:n,1:n]
Z = Zeros(n+1,n+1)

u = expand(P, x -> sin(π/2*x))
λ = -π^2/4
@test diff(u,2) ≈ λ*u
v = diff(u)
𝐜 = [(C\v)[1:n]; (P\u)[1:n-1]]
A = [M D'; D Z]
B = [Zeros(n,n) Zeros(n,n+1); Zeros(n+1,n) (P'P)[1:n+1,1:n+1]]
@test A*𝐜 ≈ λ*B*𝐜

λ,Q = eigen(A, B)
k = searchsortedfirst(λ, 0)-1
ũ = P[:,1:n+1]* Q[n+1:end,k]
plot(ũ)

@test ũ/ũ[1] ≈ u/u[1]
