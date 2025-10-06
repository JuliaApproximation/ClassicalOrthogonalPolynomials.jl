using ClassicalOrthogonalPolynomials, Plots
using ClassicalOrthogonalPolynomials: sample

x = Inclusion(ChebyshevInterval())

function christoffel(A)
    Q,R = qr(A)
    n = size(A,2)
    sum(expand(Q[:,k] .^2) for k=1:n)/n
end

function dpp(A)
    m = size(A,2)
    Q,R = qr(A)
    r = Float64[]
    for n = m:-1:1
        Kₙ = sum(expand(Q[:,k] .^2) for k=1:n)/n
        r₁ = sample(Kₙ)
        push!(r, r₁)
        Q = Q*nullspace(Q[r₁, :]')
    end
    r
end

m = 10
A = cos.((0:m)' .* x)
r = union([dpp(A) for _=1:1000]...)
histogram(r; nbins=50, normalized=true)
plot!(christoffel(A); ylims=(0,1))

## DPPs are much better condtioned
Q,R = qr(A)
@test cond(Q[dpp(A),:]) ≤ 100
@test cond(Q[sample(christoffel(A),11),:]) ≥ 1000
@test cond(Q[range(-1,1,11),:]) > 1E13