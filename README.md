# ClassicalOrthogonalPolynomials.jl
A Julia package for classical orthogonal polynomials and expansions

[![Build Status](https://github.com/JuliaApproximation/ClassicalOrthogonalPolynomials.jl/workflows/CI/badge.svg)](https://github.com/JuliaApproximation/FastGaussQuadrature.jl/actions)
[![codecov](https://codecov.io/gh/JuliaApproximation/ClassicalOrthogonalPolynomials.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/ClassicalOrthogonalPolynomials.jl)


This package implements classical orthogonal polynomials as quasi-arrays where one one axes is continuous and the other axis is discrete (countably infinite), as implemented in [QuasiArrays.jl](https://github.com/JuliaApproximation/QuasiArrays.jl) and  [ContinuumArrays.jl](https://github.com/JuliaApproximation/ContinuumArrays.jl).  
```julia
julia> using ClassicalOrthogonalPolynomials, ContinuumArrays

julia> P = Legendre(); # Legendre polynomials

julia> size(P) # uncountable ∞ x countable ∞
(ℵ₁, ∞)

julia> axes(P) # essentially (-1..1, 1:∞), Inclusion plays the same role as Slice
(Inclusion(-1.0..1.0 (Chebyshev)), OneToInf())

julia> P[0.1,1:10] # [P_0(0.1), …, P_9(0.1)]
10-element Array{Float64,1}:
  1.0                
  0.1                
 -0.485              
 -0.14750000000000002
  0.3379375          
  0.17882875         
 -0.2488293125       
 -0.19949294375000004
  0.180320721484375  
  0.21138764183593753

julia> @time P[range(-1,1; length=10_000), 1:10_000]; # construct 10_000^2 Vandermonde matrix
  1.624796 seconds (10.02 k allocations: 1.491 GiB, 6.81% gc time)
```
This also works for associated Legendre polynomials as weighted Ultraspherical polynomials:
```julia
julia> associatedlegendre(m) = ((-1)^m*prod(1:2:(2m-1)))*(UltrasphericalWeight((m+1)/2).*Ultraspherical(m+1/2))
associatedlegendre (generic function with 1 method)

julia> associatedlegendre(2)[0.1,1:10]
10-element Array{Float64,1}:
   2.9699999999999998
   1.4849999999999999
  -6.9052500000000006
  -5.041575          
  10.697754375       
  10.8479361375      
 -13.334647528125    
 -18.735466024687497 
  13.885467170308594 
  28.220563705988674 
```

## p-Finite Element Method

The language of quasi-arrays gives a natural framework for constructing p-finite element methods. The convention
is that adjoint-products are understood as inner products over the axes with uniform weight. Thus to solve Poisson's equation
using its weak formulation with Dirichlet conditions we can expand in a weighted Jacobi basis:
```julia
julia> P¹¹ = Jacobi(1.0,1.0); # Quasi-matrix of Jacobi polynomials

julia> w = JacobiWeight(1.0,1.0); # quasi-vector correspoinding to (1-x^2)

julia> w[0.1] ≈ (1-0.1^2)
true

julia> S = w .* P¹¹; # Quasi-matrix of weighted Jacobi polynomials

julia> D = Derivative(axes(S,1)); # quasi-matrix corresponding to derivative

julia> Δ = (D*S)'*(D*S) # weak laplacian corresponding to inner products of weighted Jacobi polynomials
∞×∞ LazyArrays.ApplyArray{Float64,2,typeof(*),Tuple{Adjoint{Int64,BandedMatrices.BandedMatrix{Int64,Adjoint{Int64,InfiniteArrays.InfStepRange{Int64,Int64}},InfiniteArrays.OneToInf{Int64}}},LazyArrays.BroadcastArray{Float64,2,typeof(*),Tuple{LazyArrays.BroadcastArray{Float64,1,typeof(/),Tuple{Int64,InfiniteArrays.InfStepRange{Int64,Int64}}},BandedMatrices.BandedMatrix{Int64,Adjoint{Int64,InfiniteArrays.InfStepRange{Int64,Int64}},InfiniteArrays.OneToInf{Int64}}}}}} with indices OneToInf()×OneToInf():
 2.66667   ⋅     ⋅        ⋅        ⋅        ⋅        ⋅        ⋅      …  
  ⋅       6.4    ⋅        ⋅        ⋅        ⋅        ⋅        ⋅         
  ⋅        ⋅   10.2857    ⋅        ⋅        ⋅        ⋅        ⋅         
  ⋅        ⋅     ⋅      14.2222    ⋅        ⋅        ⋅        ⋅         
  ⋅        ⋅     ⋅        ⋅      18.1818    ⋅        ⋅        ⋅         
  ⋅        ⋅     ⋅        ⋅        ⋅      22.1538    ⋅        ⋅      …  
  ⋅        ⋅     ⋅        ⋅        ⋅        ⋅      26.1333    ⋅         
  ⋅        ⋅     ⋅        ⋅        ⋅        ⋅        ⋅      30.1176     
  ⋅        ⋅     ⋅        ⋅        ⋅        ⋅        ⋅        ⋅         
  ⋅        ⋅     ⋅        ⋅        ⋅        ⋅        ⋅        ⋅         
  ⋅        ⋅     ⋅        ⋅        ⋅        ⋅        ⋅        ⋅      …  
  ⋅        ⋅     ⋅        ⋅        ⋅        ⋅        ⋅        ⋅         
 ⋮                                         ⋮                         ⋱  
```


