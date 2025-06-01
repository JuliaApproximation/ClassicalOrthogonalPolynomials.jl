using ClassicalOrthogonalPolynomials, OrdinaryDiffEq, Plots


####
# 1D Helmholtz w/ Dirichlet conditions
#
# We solve `u'' + 4^2*u = f`, `u(-1) = 1`, `u(1) = 2` with rectangular collocation. That is,
# we discretise `u` at `n` second kind Chebyshev points (containing ±1)
# but `f` at `n-2` first kind Chebyshev points (interior to [-1,1]).
# The reduction in degrees of freedom allows us to impose boundary conditions.
####

T = ChebyshevT() # basis
n = 100
x₁ = reverse(ChebyshevGrid{1}(n-2)) # 1st kind points, sorted
x₂ = reverse(ChebyshevGrid{2}(n)) # 2nd kind points, sorted
V = T[x₂,1:n] # Vandermonde matrix, its inverse is transform from values to coefficients
D₂ = diff(T,2)[x₁,1:n] / V # 2nd derivative from x₂ to x₁
R = T[x₁,1:n] / V # discretisation of identity matrix

B_l = [1; zeros(n-1)]' # Left Dirichlet conditions
B_r = [zeros(n-1); 1]' # Right Dirichlet conditions

u = [B_l; D₂ + 4^2*R; B_r] \ [1; exp.(x₁); 2]
plot(x₂, u)

####
# Heat equation
#
# We solve `u_t = u_xx` with rectangular collocation with zero Dirchlet conditions. That is,
# we discretise `u` at 2nd kind Chebyshev points (containing ±1)
# but the range of `u_xx` at 1st kind Chebyshev points (interior to [-1,1]).
####

u₀ = x -> (1-x^2) * exp(x) # initial condition

function heat!(du, u, D₂, t)
    du[1] = u[1] # left bc
    mul!(@view(du[2:end-1]), D₂, u)
    du[end] = u[end] # right bc
end
prob = ODEProblem(ODEFunction(heat!, mass_matrix = [B_l; R; B_r]), u₀.(x₂), (0., 1.), D₂) 
sol = solve(prob, Rodas5(), reltol = 1e-8, abstol = 1e-8)

t = range(0,1,100)
contourf(t, x₂,   hcat(sol.(t)...))


####
# Burgers equation
#
# We solve `u_t = u_xx - ν*u*u'` with rectangular collocation with zero Dirchlet conditions.
####

D₁ = diff(T,1)[x₁,1:n] / V # 1st derivative from x₂ to x₁

function burgers!(du, u, (D₁, D₂, R, ν), t)
    du[1] = u[1] # left bc
    mul!(@view(du[2:end-1]), D₂, u)
    @view(du[2:end-1]) .-= ν .* (R*u) .* (D₁*u) # currently allocating, pass temp
    du[end] = u[end] # right bc
end
prob = ODEProblem(ODEFunction(burgers!, mass_matrix = [B_l; R; B_r]), u₀.(x₂), (0., 1.), (D₁, D₂, R, 3)) 
sol = solve(prob, Rodas5(), reltol = 1e-8, abstol = 1e-8)

t = range(0,1,100)
contourf(t, x₂,   hcat(sol.(t)...))
