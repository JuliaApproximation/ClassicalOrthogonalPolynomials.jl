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
@test_broken isreal(λ[1]) # and it computes nonsense


#####
# Weak form: integrating by parts we get:
# ⟨∇v, ∇u⟩ = λ*⟨v,u⟩
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
𝐮 = Q[:,1:n]*U[1:n,end]; 𝐮 *= -π/(2*𝐮[1])
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
𝐮 = W[:,1:n]*U[1:n,k]; 𝐮 *= π/(2*𝐮[1])

@test uⁿ ≈ u
@test 𝐮ⁿ ≈ diff(u)


# We can also solve the k = 0 Hodge Laplace equation. 



##########
# 2D
##########

######
# Mixed form:
# We can augment the equation with a derivative:
#
#   𝐮 := ∇u
#
# where we impose this definition weakly, obtaining a system:
#
#    -<𝐯, 𝐮> + <∇×𝐯,u> = 0
#    <v, ∇×𝐮>          = λ<v,u>
#
# This is equivalent to the k = 2 Hodge–Laplacian mixed formulation,
# that is, we view u as a 2-form, and 𝐮 = δ*u a 1-form in H_curl where δ = ∇' = -div = -d/dx
#
# we use the basis [P_k(x)*C_j(y),0] and [0,C_k(x)*P_j(y)] for H_curl. 
######

n = 20
M_C = (C'C)[1:n,1:n]
M_P = (P'P)[1:n,1:n]

M_curl1 = kron(M_C,M_P)
M_curl2 = kron(M_P,M_C)
Z = zero(M_curl2)

[M_curl1    Z           
 Z          M_curl2     ]
