using ContinuumArrays, ClassicalOrthogonalPolynomials, FillArrays, Plots


######
# Strong v Weak v Mixed p-FEM on intervals and squares
# Here we give examples of solving the Laplacian eigenvalue problem
#  -Δu = λu
# with Dirichlet/Neumann conditions.
######

# we first introduce the different bases we will use:

W = Weighted(Ultraspherical(3/2)) # integrated Legendre bubbles
C = Ultraspherical(-1/2) # all integrated Legendre functions (but not zero at -1)
P = Legendre()

######
# 1D
#####

#####
# Strong form:
# -⟨v,Δu⟩ = λ*⟨v,u⟩
# we can include dirichlet conditions in the test/trial basis:
######

n = 100
M = (W'W)[1:n,1:n] # mass matrix
Δ = -(W'diff(W,2))[1:n,1:n] # stiffness matrix

λ,U = eigen(Symmetric(Δ),Symmetric(M))
@test λ[1] ≈ π^2/4
uᵈ = W[:,1:n]U[:,1]
𝐮ᵈ = diff(uᵈ)

@test uᵈ[0] ≈ 1 # somehow it picked up the right normalisation
@test uᵈ ≈ [cos(π/2*x) for x in -1..1] # we match the value
@test 𝐮ᵈ ≈ [-π/2*sin(π/2*x) for x in -1..1] # and derivative

# without the vanishing conditions we don't get symmetric matrices or a good solver:

n = 100
M = (C'C)[1:n,1:n]
Δ = -(C'diff(C,2))[1:n,1:n]

λ,U = eigen(Δ, M) # we lose symmetry
@test !isreal(λ[1]) # and it computes nonsense


#####
# Weak form: integrating by parts we get:
#
# ⟨∇v, ∇u⟩ = λ*⟨v,u⟩
#
# This is precisely the k = 0 Hodge Laplace equation. 
# we can still include dirichlet conditions in the test/trial basis:
######

n = 100
M = (W'W)[1:n,1:n] # mass matrix
Δ = (diff(W)'diff(W))[1:n,1:n] # stiffness matrix

λ,U = eigen(Symmetric(Δ),Symmetric(M))
@test λ[1] ≈ π^2/4
@test uᵈ ≈ W[:,1:n]U[:,1]
@test 𝐮ᵈ ≈ diff(W[:,1:n]U[:,1])

# but now without vanishing conditions we get symmetry and Neumann conditions:

M = (C'C)[1:n,1:n] # mass matrix
Δ = (diff(C)'diff(C))[1:n,1:n] # stiffness matrix

λ,U = eigen(Symmetric(Δ),Symmetric(M))
@test λ[1] == 0 # Neumann has a kernel
@test λ[2] ≈ π^2/4
uⁿ = C[:,1:n]U[:,2] # we will use the first non-trivial eigenfunction as a test
𝐮ⁿ = diff(uⁿ)

@test uⁿ ≈ [sin(π/2 * x) for x in -1..1]
@test 𝐮ⁿ ≈ [π/2*cos(π/2 * x) for x in -1..1]


######
# Mixed form:
# We can augment the equation with a derivative:
#
#   𝐮 := ∇u
#
# where we impose this definition weakly, obtaining a system:
#
#    -<𝐯, 𝐮> + <∇𝐯,u> = 0
#    <v, ∇𝐮>          = λ<v,u>
#
# This is equivalent to the k = 1 Hodge–Laplacian mixed formulation,
# that is, we view u as a 1-form, and 𝐮 = δ*u a 0-form where δ = ∇' = -div = -d/dx
# and here d = ∇ = d/dx maps from 0-forms to 1-forms.
#
# This is a bit mind-bending as in this formulation we differentiate the derivative 𝐮,
# not the solution u, and hence we need 𝐮 to be differentiable (i.e. expanded in either W or C)
# whilst u is only L^2 (i.e. expandable in P).
#
# We need to be careful with the discretisation sizes to ensure we decompose
# the space into subcomplexes.

# First we try without imposing constraints where now we get Dirichlet conditions:
######

n = 100 # discretisation size for u
D = (P'diff(C))[1:n-1,1:n]
M_C = (C'C)[1:n,1:n] # we expand 𝐮 in C, this is mass matrix
M_P = (P'P)[1:n-1,1:n-1] # we expand u in P, this is mass matrix for right-hand side
Z = Zeros(n-1,n-1)
A = [-M_C D'; D Z] # This the operator
B = zero(A); B[n+1:end,n+1:end] = M_P

# the above does not work with generalised eigenvalue solvers so we have to
# be naughty and invert and use a non-symmetric eigenvalue solver:

λ,U = eigen(A\B); λ = inv.(λ)
@test λ[end] ≈ π^2/4
u = P[:,1:n-1]*U[n+1:end,end]; u /= u[0]
𝐮 = C[:,1:n]*U[1:n,end]; 𝐮 *= -π/(2*𝐮[1])
@test uᵈ ≈ u
@test 𝐮ᵈ ≈ 𝐮 ≈ diff(u)

# We can switch to Neumann conditions by using a weighted basis for the derivative:

n = 100 # discretisation size for u
D = (P'diff(W))[1:n+1,1:n]
M_W = (W'W)[1:n,1:n] # we expand 𝐮 in C, this is mass matrix
M_P = (P'P)[1:n+1,1:n+1] # we expand u in P, this is mass matrix for right-hand side
Z = Zeros(n+1,n+1)
A = [-M_W D'; D Z] # This the operator
B = zero(A); B[n+1:end,n+1:end] = M_P

λ,U = eigen(A,B)
k = searchsortedfirst(λ,0)+1
@test λ[k] ≈ π^2/4
u = P[:,1:n+1]*U[n+1:end,k]; u /= u[1]
𝐮 = W[:,1:n]*U[1:n,k]; 𝐮 *= π/(2*𝐮[0])

@test uⁿ ≈ u
@test 𝐮ⁿ ≈ diff(u) ≈ 𝐮




##########
# 2D
##########


# Weak form: the k = 0 Hodge–Laplace equation is equivalent to
#
# ⟨∇v, ∇u⟩ = λ*⟨v,u⟩
#
# We can use the basis Wₖ(x)Wⱼ(y) where Wₖ(x) = (1-x^2)*Cₖ^(3/2)(x)


n = 30
M_W = (W'W)[1:n,1:n] # mass matrix
D² = (diff(W)'diff(W))[1:n,1:n] # stiffness matrix
Δ = kron(D²,M_W) + kron(M_W,D²)
M = kron(M_W, M_W)
λ,U = eigen(Symmetric(Δ),Symmetric(M))
@test λ[1] ≈ π^2/2
uᵈ = W[:,1:n]reshape(U[:,1],n,n)W[:,1:n]'; uᵈ /= uᵈ[0,0]
𝐮ᵈ_x,𝐮ᵈ_y = diff(uᵈ; dims=1), diff(uᵈ; dims=2)
@test uᵈ[0.1,0.2] ≈ cos(π/2*0.1)*cos(π/2*0.2)
@test 𝐮ᵈ_x[0.1,0.2] ≈ -π/2*sin(π/2*0.1)*cos(π/2*0.2)
@test 𝐮ᵈ_y[0.1,0.2] ≈ -π/2*cos(π/2*0.1)*sin(π/2*0.2)

# but now without vanishing conditions we get symmetry and Neumann conditions:

M_C = (C'C)[1:n,1:n] # mass matrix
D² = (diff(C)'diff(C))[1:n,1:n] # stiffness matrix
Δ = kron(D²,M_C) + kron(M_C,D²)
M = kron(M_C, M_C)

λ,U = eigen(Symmetric(Δ),Symmetric(M))
@test λ[1] == 0 # Neumann has a kernel
@test all(λ[2:3] .≈ π^2/4)
@test λ[4] ≈ 2π^2/4
uⁿ = C[:,1:n]reshape(U[:,4],n,n)C[:,1:n]'; uⁿ /= uⁿ[1,1]
𝐮ⁿ_x,𝐮ⁿ_y = diff(uⁿ; dims=1), diff(uⁿ; dims=2)
@test uⁿ[0.1,0.2] ≈ sin(π/2*0.1)*sin(π/2*0.2)
@test 𝐮ⁿ_x[0.1,0.2] ≈ π/2*cos(π/2*0.1)*sin(π/2*0.2)
@test 𝐮ⁿ_y[0.1,0.2] ≈ π/2*sin(π/2*0.1)*cos(π/2*0.2)


######
# For k = 2 Hodge–Laplacian we define
#
#   𝐮 := ∇^⟂ u
#
# obtaining a system:
#
#    -<𝐯, 𝐮> + <∇×𝐯,u> = 0
#    <v, ∇×𝐮>          = λ<v,u>
#
# where the 2D curl is ∇× = [-∂_y ∂_x]. Here we view u as a 2-form, and 𝐮 = -δ*u a 1-form in H_curl where δ = ∇×' = ∇^⟂ = [-∂_y; ∂_x]
#
# we use the basis [P_k(x)*C_j(y),0] and [0,C_k(x)*P_j(y)] for [u_1,u_2] = 𝐮 ∈ H_curl, thus we actually
# get a 3-vector system (using the fact that [u_1,0] and [0,u_2] are automatically orthogonal):
#
#    -<v_1, u_1>  - <v_2, u_2>  - <∂_y v_1,u> + <∂_x v_2,u> = 0
#    - <v, ∂_y u_1> + <v, ∂_x v_2>          = λ<v,u>
#
######

n = 20
M_C = (C'C)[1:n,1:n]
M_P = Diagonal((P'P).diag[1:n-1])
D = (P'diff(C))[1:n-1,1:n]

M_curl1 = kron(M_C,M_P)
M_curl2 = kron(M_P,M_C)
D_x = kron(M_P, D)
D_y = kron(D, M_P)
M = kron(M_P,M_P)
Z = zero(M_curl1)
Z₂ = zero(M)
Z₃ = zero(D_y)

A = [M_curl1    Z           -D_y';
     Z          M_curl2     D_x';
    -D_y        D_x        Z₂]

B =  [Z         Z           Z₃';
      Z         Z           Z₃';
      Z₃        Z₃        M]


λ,Q = eigen(-(A\B)); λ = inv.(λ)

@test λ[end] ≈ π^2/2

u = P[:,1:n-1]reshape(Q[2n*(n-1)+1:end,end], n-1, n-1)*P[:,1:n-1]'; κ = 1/u[0,0]; u *= κ
u_1 = P[:,1:n-1]*reshape(Q[1:n*(n-1),end], n-1, n)*C[:,1:n]'; u_1 *= κ
u_2 = C[:,1:n]*reshape(Q[n*(n-1)+1:2n*(n-1),end], n, n-1)*P[:,1:n-1]'; u_2 *= κ

@test u[0.1,0.2] ≈ uᵈ[0.1,0.2]
@test -diff(u;dims=2)[0.1,0.2] ≈ u_1[0.1,0.2] ≈ -𝐮ᵈ_y[0.1,0.2]
@test diff(u;dims=1)[0.1,0.2] ≈ u_2[0.1,0.2] ≈ 𝐮ᵈ_x[0.1,0.2]

# using a weighted basis for u_1,u_2 imposes Neumann conditions:

n = 20
M_W = (W'W)[1:n-1,1:n-1]
M_P = Diagonal((P'P).diag[1:n])
D = (P'diff(W))[1:n,1:n-1]

M_curl1 = kron(M_W,M_P)
M_curl2 = kron(M_P,M_W)
D_x = kron(M_P, D)
D_y = kron(D, M_P)
M = kron(M_P,M_P)
Z = zero(M_curl1)
Z₂ = zero(M)
Z₃ = zero(D_y)

A = [M_curl1    Z           -D_y';
     Z          M_curl2     D_x';
    -D_y        D_x        Z₂]

B =  [Z         Z           Z₃';
      Z         Z           Z₃';
      Z₃        Z₃        M]


λ,Q = eigen(A,B); Q = real(Q)

k = searchsortedfirst(real(λ),0)
@test λ[k] == 0
@test all(λ[k-2:k-1] .≈ -π^2/4)
@test λ[k-3] ≈ -π^2/2

u = P[:,1:n]reshape(Q[2n*(n-1)+1:end,k-3], n, n)*P[:,1:n]'; κ = 1/u[1,1]; u *= κ
u_1 = P[:,1:n]*reshape(Q[1:n*(n-1),k-3], n, n-1)*W[:,1:n-1]'; u_1 *= κ
u_2 = W[:,1:n-1]*reshape(Q[n*(n-1)+1:2n*(n-1),k-3], n-1, n)*P[:,1:n]'; u_2 *= κ

@test u[0.1,0.2] ≈ uⁿ[0.1,0.2]
@test -diff(u;dims=2)[0.1,0.2] ≈ u_1[0.1,0.2] ≈ -𝐮ⁿ_y[0.1,0.2]
@test diff(u;dims=1)[0.1,0.2] ≈ u_2[0.1,0.2] ≈ 𝐮ⁿ_x[0.1,0.2]
