using ClassicalOrthogonalPolynomials, Plots
using BlockArrays, FillArrays

W = Weighted(Jacobi(1,1))
x = axes(W,1)
Q1 = [x.+1 W]
Q2 = [(1 .-x) W]
D = Derivative(x)

Δ1 = -((D*Q1)'*(D*Q1))
Δ2 = -((D*Q2)'*(D*Q2))

M1 = Q1'Q1
M2 = Q2'Q2

n = 10
S = PseudoBlockArray(zeros(2n+2,2n+1), [1; n; 1; n], [1; n; n])
S[1,1] = 1
S[Block(2,2)] = Eye(n,n)
S[Block(3,1)] .= 1
S[Block(4,3)] = Eye(n,n)

M̃ = [Matrix(M1[1:n+1,1:n+1]) Zeros(n+1,n+1);
    Zeros(n+1,n+1) Matrix(M2[1:n+1,1:n+1])]

Δ̃ = [Matrix(Δ1[1:n+1,1:n+1]) Zeros(n+1,n+1);
        Zeros(n+1,n+1) Matrix(Δ2[1:n+1,1:n+1])]

M = S' * M̃ * S
Δ = S' * Δ̃ * S

P = Legendre()

f1 = Q1'*P*(P\exp.(x .- 1))
f2 = Q2'*P*(P\exp.(x .+ 1))

f = S'*([f1[1:n+1]; f2[1:n+1]])
u = S * (Δ \ f)
u1 = u[Block.(1:2)]
u2 = u[Block.(3:4)]

x1 = range(-2,0; length=1000)
x2 = range( 0,2; length=1000)
plot(x1, [x1.+2 W[x1.+1,1:length(u1)-1]] * u1)
plot!(x2, [(2 .-x2) W[x2.-1,1:length(u2)-1]] * u2)
Q1[x1 .+ 1,1:3]

T = chebyshevt(-2..2)
x = axes(T,1)
D = Derivative(x)
C = ultraspherical(2,-2..2)

c = Vcat(T[[begin,end],:], (C\(D^2)*T) + 0 * (C\T)) \ [0; 0; C \ exp.(x)]
plot(T * c)


plot([x1; x2], (T * c)[[x1; x2]])
plot!(x2, (T * c)[x2])






D = sqrt(-Diagonal(Δ))

B = inv(D) * M * inv(D)

kron(B,Eye(size(B))) + kron(Eye(size(B)),B)

kron(Δ,M) + kron(M,Δ) |> Symmetric |> eigvals


eigvals(Symmetric(Matrix(B)))

W = Weighted(Jacobi(1,1))

x = axes(W,1)
Q1 = [x.+1 W]
D = Derivative(x)
Δ1 = ((D*Q1)'*(D*Q1))



(x.+1)'*(x.+1)

@ent Derivative(x) * x