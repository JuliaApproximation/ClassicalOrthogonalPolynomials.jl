using ClassicalOrthogonalPolynomials, ContinuumArrays, DifferentialEquations, Plots

###
# Heat
###

S = JacobiWeight(1.0,1.0) .* Jacobi(1.0,1.0)
D = Derivative(axes(S,1))
Δ = -((D*S)'*(D*S))
M = S'S

function evolution(u, (M,A), t) 
    M\(A*u)
end
n = 50
u0 = [[1,2,3]; zeros(n-3)]
prob = ODEProblem(evolution,u0,(0.0,3.0),(cholesky(Symmetric(M[1:n,1:n])),Δ[1:n,1:n]))
@time u = solve(prob,TRBDF2()); u(1.0)

###
# Heat natural
###
L = LinearSpline(range(-1,1;length=2))
P = apply(hcat,L,S)
ApplyArray(hcat,Legendre() \ (D*L), Legendre() \ (D*S))

(D*L)'*(D*L)

(D*S)'*(D*S)

((D*S)'*(D*L)).args[3]
(D*L)'*(D*S)
H = (D*L).args[1]
P = Legendre()
(P\H)' * P'P * (P \ (D*S))



###
# Beam
###

# [u, u_t]_t = [0 I; M] * [u, u_t]

S = JacobiWeight(2.0,2.0) .* Jacobi(2.0,2.0)
D = Derivative(axes(S,1))
Δ² = (D^2*S)'*(D^2*S)
M = S'S
function wave(uv, (M,A), t) 
    n = size(M,1)
    u,v = uv[1:n],uv[n+1:end]
    [v; M\(A*u)]
end
n = 30
u0 = [1; 2; 3; zeros(n-3)]
prob = ODEProblem(wave,[u0; zeros(n)],(0.0,3.0),(cholesky(Symmetric(M[1:n,1:n])),-(Δ²)[1:n,1:n]))
@time u = solve(prob,TRBDF2()); u(1.0)

g = range(-1,1;length=200)
V = S[g,1:n]
@gif for t in 0.0:0.01:3.0
    plot(g, V*u(t)[1:n]; ylims=(-5,5))
end


prob = ODEProblem(wave,[u0; zeros(n)],(0.0,3.0),(qr(M),-(Δ²)))