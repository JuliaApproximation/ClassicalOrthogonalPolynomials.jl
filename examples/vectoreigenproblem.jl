using ClassicalOrthogonalPolynomials, Plots

#####
# -u'' - a*w = λb*u
# -w'' - u = λb*u
# u(0) = 0
# w(0) = 0
# w'(1) + λ*w(1) + u(1)
####

T = chebyshevt(0..1) # solution basis is T_n 
C = ultraspherical(2,0..1) # RHS basis is C_n^(2)
z = axes(T,1)
a = 1 .+ z
b = z

n = 20
D² = (C\diff(T,2))[1:n-2,1:n] # 2nd derivative discretisation
A = (C\(a .* T))[1:n-2,1:n] # multiplication by a
B = (C\(b .* T))[1:n-2,1:n] # multiplication by b
R = (C\T)[1:n-2,1:n]

𝐞₀ = T[0, 1:n] # evaluate at 0
𝐞₁ = T[1, 1:n] # evaluate at 1
𝐝₁ = diff(T)[1,1:n] # evaluate derivative at 1
𝐳 = zeros(n)
Z = zeros(n-2,n)

bcs = [𝐞₀'    𝐳';
       𝐳'     𝐞₀';
       𝐞₁'    𝐝₁';
       𝐞₀'     -𝐞₁']
       

ops = [-D²            -A;
       -R             -D²]


M = [𝐳' 𝐳'; # first bc is zero
     𝐳' 𝐳'; # second bc is zero
     𝐳' 𝐞₁'; # 3rd bc is λ*w(1)
     𝐳' 𝐳'; # 4th bc is zero
     B  Z;  # λ*b *u
     Z  B]  # λ*b *w


λ, 𝐮 = eigen([bcs; ops], M)

@test 8 ≤ real(λ[3]) ≤ 9 # a reasonable eigenvalue

u,w = T[:,1:n]*real(𝐮[:,3][1:n]), T[:,1:n]*real(𝐮[:,3][n+1:end])

plot(u); plot!(w)