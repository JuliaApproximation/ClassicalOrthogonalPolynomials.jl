# ClassicalOrthogonalPolynomials.jl
A Julia package for classical orthogonal polynomials and expansions

```@meta
CurrentModule = ClassicalOrthogonalPolynomials
```

## Definitions

We follow the [Digital Library of Mathematical Functions](https://dlmf.nist.gov/18.3),
which defines the following classical orthogonal polynomials:

1. Legendre: $P_n(x)$, defined over $[-1, 1]$ with weight $w(x) = 1$.
2. Chebyshev (1st kind, 2nd kind): $T_n(x)$ and $U_n(x)$, defined over $[-1, 1]$ with weights $w(x) = 1/\sqrt{1-x^2}$ and $w(x) = \sqrt{1-x^2}$, respectively.
3. Ultraspherical: $C_n^{(\lambda)}(x)$, defined over $[-1, 1]$ with weight $w(x) = (1-x^2)^{\lambda-1/2}$.
4. Jacobi: $P_n^{(a,b)}(x)$, defined over $[-1, 1]$ with weight $w(x) = (1-x)^a(1+x)^b$.
5. Laguerre: $L_n^{(\alpha)}(x)$, defined over $[0, ∞)$ with weight $w(x) = x^\alpha \mathrm{e}^{-x}$.
6. Hermite: $H_n(x)$, defined over $(-∞, ∞)$ with weight $w(x) = \mathrm{e}^{-x^2}$.

## Evaluation

The simplest usage of this package is to evaluate classical
orthogonal polynomials:
```@repl userguide
using ClassicalOrthogonalPolynomials
n, x = 5, 0.1;
legendrep(n, x) # P_n(x)
chebyshevt(n, x) # T_n(x) == cos(n*acos(x))
chebyshevu(n, x) # U_n(x) == sin((n+1)*acos(x))/sin(acos(x))
λ = 0.3; ultrasphericalc(n, λ, x) # C_n^(λ)(x)
a,b = 0.1,0.2; jacobip(n, a, b, x) # P_n^(a,b)(x)
laguerrel(n, x) # L_n(x)
α = 0.1; laguerrel(n, α, x) # L_n^(α)(x)
hermiteh(n, x) # H_n(x)
```

## Continuum arrays

For expansions, recurrence relationships, and other operations linked with linear equations, it is useful to treat the families of orthogonal 
polynomials as _continuum arrays_, as implemented in [ContinuumArrays.jl](https://github.com/JuliaApproximation/ContinuumArrays.jl). The continuum arrays implementation is accessed as follows:
```@repl userguide
T = ChebyshevT() # Or just Chebyshev()
axes(T) # [-1,1] by 1:∞
T[x, n+1] # T_n(x) = cos(n*acos(x))
```
We can thereby access many points and indices efficiently using array-like language:
```@repl userguide
x = range(-1, 1; length=1000);
T[x, 1:1000] # [T_j(x[k]) for k=1:1000, j=0:999]
```

## Expansions

We view a function expansion in say Chebyshev polynomials in terms of continuum arrays as follows:
```math
f(x) = \sum_{k=0}^∞ c_k T_k(x) = \begin{bmatrix}T_0(x) | T_1(x) | … \end{bmatrix} 
\begin{bmatrix}c_0\\ c_1 \\ \vdots \end{bmatrix} = T[x,:] * 𝐜
```
To be more precise, we think of functions as continuum-vectors. Here is a simple example:
```@repl userguide
f = T * [1; 2; 3; zeros(∞)]; # T_0(x) + T_1(x) + T_2(x)
f[0.1]
```
To find the coefficients for a given function we consider this as the problem of finding `𝐜`
such that `T*𝐜 == f`, that is:
```@repl userguide
T \ f
```
For a function given only pointwise we broadcast over `x`, e.g., the following are the coefficients of `\exp(x)`:
```@repl userguide
x = axes(T, 1);
c = T \ exp.(x)
f = T*c; f[0.1] # ≈ exp(0.1)
```
With a little cheeky usage of Julia's order-of-operations this can be written succicently as:
```@repl userguide
f = T / T \ exp.(x);
f[0.1]
```

(Or for more clarity just write `T * (T \ exp.(x))`.)


## Jacobi matrices

Orthogonal polynomials satisfy well-known three-term recurrences:
```math
x p_n(x) = c_{n-1} p_{n-1}(x) + a_n p_n(x) + b_n p_{n+1}(x).
```
In continuum-array language this has the  form of a comuting relationship:
```math
x \begin{bmatrix} p_0 | p_1 | \cdots \end{bmatrix} = \begin{bmatrix} p_0 | p_1 | \cdots \end{bmatrix} \begin{bmatrix} a_0 & c_0  \\ b_0 & a_1 & c_1 \\ & b_1 & a_2 & \ddots \\ &&\ddots & \ddots \end{bmatrix}
```
We can therefore find the Jacobi matrix naturally as follows:
```@repl userguide
T \ (x .* T)
```
Alternatively, just call `jacobimatrix(T)` (noting its the transpose of the more traditional convention).


## Derivatives

The derivatives of classical orthogonal polynomials are also classical OPs, and this can be seen as follows:
```@repl userguide
U = ChebyshevU();
D = Derivative(x);
U\D*T
```
Similarly, the derivative of _weighted_ classical OPs are weighted classical OPs:
```@repl userguide
Weighted(T)\D*Weighted(U)
```

## Other recurrence relationships

Many other sparse recurrence relationships are implemented. Here's one:
```@repl userguide
U\T
```
(Probably best to ignore the type signature 😅)


## Index

### Polynomials

```@docs
ClassicalOrthogonalPolynomials.Chebyshev
ClassicalOrthogonalPolynomials.chebyshevt
ClassicalOrthogonalPolynomials.chebyshevu
```
```@docs
ClassicalOrthogonalPolynomials.Legendre
ClassicalOrthogonalPolynomials.legendrep
```
```@docs
ClassicalOrthogonalPolynomials.Jacobi
ClassicalOrthogonalPolynomials.jacobip
```
```@docs
ClassicalOrthogonalPolynomials.laguerrel
```
```@docs
ClassicalOrthogonalPolynomials.hermiteh
```


### Weights

```@docs
ClassicalOrthogonalPolynomials.OrthonormalWeighted
```
```@docs
ClassicalOrthogonalPolynomials.HermiteWeight
```
```@docs
ClassicalOrthogonalPolynomials.Weighted
```
```@docs
ClassicalOrthogonalPolynomials.LegendreWeight
ClassicalOrthogonalPolynomials.ChebyshevWeight
ClassicalOrthogonalPolynomials.JacobiWeight
```
```@docs
ClassicalOrthogonalPolynomials.LaguerreWeight
```
```@docs
ClassicalOrthogonalPolynomials.HalfWeighted
```

### Affine-mapped
```@docs
ClassicalOrthogonalPolynomials.legendre
ClassicalOrthogonalPolynomials.jacobi
ClassicalOrthogonalPolynomials.legendreweight
ClassicalOrthogonalPolynomials.jacobiweight
```

### Recurrences

```@docs
ClassicalOrthogonalPolynomials.normalizationconstant
```
```@docs
ClassicalOrthogonalPolynomials.OrthogonalPolynomialRatio
```
```@docs
ClassicalOrthogonalPolynomials.singularities
```
```@docs
ClassicalOrthogonalPolynomials.jacobimatrix
```
```@docs
ClassicalOrthogonalPolynomials.recurrencecoefficients
```


### Internal

```@docs
ClassicalOrthogonalPolynomials.ShuffledFFT
ClassicalOrthogonalPolynomials.ShuffledIFFT
ClassicalOrthogonalPolynomials.ShuffledR2HC
ClassicalOrthogonalPolynomials.ShuffledIR2HC
```
```@docs
ClassicalOrthogonalPolynomials.qr_jacobimatrix
ClassicalOrthogonalPolynomials.cholesky_jacobimatrix
```
```@docs
ClassicalOrthogonalPolynomials.AbstractNormalizedOPLayout
ClassicalOrthogonalPolynomials.MappedOPLayout
ClassicalOrthogonalPolynomials.WeightedOPLayout
```
```@docs
ClassicalOrthogonalPolynomials.legendre_grammatrix
```
```@docs
ClassicalOrthogonalPolynomials.weightedgrammatrix
```
```@docs
ClassicalOrthogonalPolynomials.interlace!
```
```@docs
ClassicalOrthogonalPolynomials._tritrunc
```
```@docs
ClassicalOrthogonalPolynomials.SetindexInterlace
```
```@docs
ClassicalOrthogonalPolynomials.ConvertedOrthogonalPolynomial
```
```@docs
ClassicalOrthogonalPolynomials.PiecewiseInterlace
```

