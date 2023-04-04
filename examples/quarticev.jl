using ClassicalOrthogonalPolynomials
import ClassicalOrthogonalPolynomials: OrthonormalWeighted

# find the 1m eigenvalue of -u'' + x^4*u

H = Hermite()
Q = OrthonormalWeighted(H)
x = axes(H,1)
X = Q'*(x.*Q)
D = Derivative(x)
Δ = Q'*(D^2*Q)

L = -Δ + X^4

n = 1_000_000
Lₙ = L[1:n,1:n]
Fₙ = factorize(Symmetric(Lₙ));

v = [1; zeros(n-1)];

# find the smallest eigenvalue with inverse iteration
for _=1:10
    w = Fₙ \ v; v = w/norm(w); v
end
λ = v'*Lₙ*v

@test λ ≈ eigvals(Symmetric(L[1:100,1:100]))[1]

# we can acellerate using Rayleigh shifts

v = [1; zeros(n-1)];

# find the millionth eigenvalue with inverse iteration
n = 50_000_000
E = 218506784;
Lₙ = (L-E*I)[1:n,1:n]

Fₙ = factorize(Symmetric(Lₙ));

v = randn(n)


for _=1:10
    w = Fₙ \ v; v = w/norm(w); v
    @show λ = v'*Lₙ*v
end




