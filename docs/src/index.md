# ClassicalOrthogonalPolynomials.jl
A Julia package for classical orthogonal polynomials and expansions

## Evaluation

The simplest usage of this package is to evaluate classical
orthogonal polynomials:
```jldoctest
julia> using ClassicalOrthogonalPolynomials

julia> chebyshevt(5, 0.1) # T_5(0.1) == cos(5acos(0.1))
0.48016
```