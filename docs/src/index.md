# ClassicalOrthogonalPolynomials.jl
A Julia package for classical orthogonal polynomials and expansions

## Definitions

We follow the [Digital Library of Mathematical Functions](https://dlmf.nist.gov/18.3),
which defines the following classical orthogonal polynomials:

1. Legendre: $P_n(x)$
2. Chebyshev (1st kind, 2nd kind): $T_n(x)$, $U_n(x)$
3. Ultraspherical: $C_n^{(λ)}(x)$
4. Jacobi: $P_n^{(a,b)}(x)$
5. Laguerre: $L_n^{(α)}(x)$
6. Hermite: $H_n(x)$

## Evaluation

The simplest usage of this package is to evaluate classical
orthogonal polynomials:
```jldoctest
julia> using ClassicalOrthogonalPolynomials

julia> n, x = 5, 0.1;

julia> legendrep(n, x) # P_n(x)
0.17882875

julia> chebyshevt(n, x) # T_n(x) == cos(n*acos(x))
0.48016

julia> chebyshevu(n, x) # U_n(x) == sin((n+1)*acos(x))/sin(acos(x))
0.56832

julia> λ = 0.3; ultrasphericalc(n, λ, x) # C_n^(λ)(x)
0.08578714248

julia> a,b = 0.1,0.2; jacobip(n, a, b, x) # P_n^(a,b)(x)
0.17459116797117194

julia> laguerrel(n, x) # L_n(x)
0.5483540833333331

julia> α = 0.1; laguerrel(n, α, x) # L_n^(α)(x)
0.732916666666666

julia> hermiteh(n, x) # H_n(x)
11.84032
```

## Continuum arrays

For expansions, recurrence relationships, and other operations linked with linear equations, it is useful to treat the families of orthogonal 
polynomials as _continuum arrays_, as implemented in [ContinuumArrays.jl](https://github.com/JuliaApproximation/ContinuumArrays.jl). The continuum arrays implementation is accessed as follows:
```jldoctest
julia> T = ChebyshevT() # Or just Chebyshev()
ChebyshevT()

julia> axes(T) # [-1,1] by 1:∞
(Inclusion(-1.0..1.0 (Chebyshev)), OneToInf())

julia> T[x, n+1] # T_n(x) = cos(n*acos(x))
0.48016
```
We can thereby access many points and indices efficiently using array-like language:
```jldoctest
julia> x = range(-1, 1; length=1000);

julia> T[x, 1:1000] # [T_j(x[k]) for k=1:1000, j=0:999]
1000×1000 Matrix{Float64}:
 1.0  -1.0       1.0       -1.0       1.0       -1.0       1.0       …  -1.0        1.0       -1.0        1.0       -1.0
 1.0  -0.997998  0.992     -0.98203   0.968128  -0.95035   0.928766     -0.99029    0.979515  -0.964818   0.946258  -0.92391
 1.0  -0.995996  0.984016  -0.964156  0.936575  -0.901494  0.859194     -0.448975   0.367296  -0.282676   0.195792  -0.107341
 1.0  -0.993994  0.976048  -0.946378  0.90534   -0.853427  0.791262      0.660163  -0.738397   0.807761  -0.867423   0.916664
 1.0  -0.991992  0.968096  -0.928695  0.874421  -0.806141  0.72495      -0.942051   0.892136  -0.827934   0.750471  -0.660989
 1.0  -0.98999   0.96016   -0.911108  0.843816  -0.75963   0.660237  …   0.891882  -0.946786   0.982736  -0.999011   0.995286
 1.0  -0.987988  0.952241  -0.893616  0.813524  -0.713888  0.597101      0.905338  -0.828835   0.73242   -0.618409   0.489542
 ⋮                                               ⋮                   ⋱   ⋮                                          
 1.0   0.987988  0.952241   0.893616  0.813524   0.713888  0.597101     -0.905338  -0.828835  -0.73242   -0.618409  -0.489542
 1.0   0.98999   0.96016    0.911108  0.843816   0.75963   0.660237     -0.891882  -0.946786  -0.982736  -0.999011  -0.995286
 1.0   0.991992  0.968096   0.928695  0.874421   0.806141  0.72495   …   0.942051   0.892136   0.827934   0.750471   0.660989
 1.0   0.993994  0.976048   0.946378  0.90534    0.853427  0.791262     -0.660163  -0.738397  -0.807761  -0.867423  -0.916664
 1.0   0.995996  0.984016   0.964156  0.936575   0.901494  0.859194      0.448975   0.367296   0.282676   0.195792   0.107341
 1.0   0.997998  0.992      0.98203   0.968128   0.95035   0.928766      0.99029    0.979515   0.964818   0.946258   0.92391
 1.0   1.0       1.0        1.0       1.0        1.0       1.0           1.0        1.0        1.0        1.0        1.0
```

## Expansions

We view a function expansion in say Chebyshev polynomials in terms of continuum arrays as follows:
$$
f(x) = \sum_{k=0}^∞ c_k T_k(x) = \begin{bmatrix}T_0(x) | T_1(x) | … \end{bmatrix} 
\begin{bmatrix}c_0\\ c_1 \\ \vdots \end{bmatrix} = T[x,:] * 𝐜
$$
To be more precise, we think of functions as continuum-vectors. Here is a simple example:
```jldoctest
julia> f = T * [1; 2; 3; zeros(∞)]; # T_0(x) + T_1(x) + T_2(x)

julia> f[0.1]
-1.74
```
To find the coefficients for a given function we consider this as the problem of finding $𝐜$
such that $T*𝐜 == f$, that is:
```julia
julia> T \ f
vcat(3-element Vector{Float64}, ℵ₀-element FillArrays.Zeros{Float64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}} with indices OneToInf()) with indices OneToInf():
 1.0
 2.0
 3.0
  ⋅ 
  ⋅ 
  ⋅ 
  ⋅ 
 ⋮
```
For a function given only pointwise we broadcast over `x`, e.g., the following are the coefficients of $\exp(x)$:
```julia
julia> x = axes(T, 1);

julia> c = T \ exp.(x)
vcat(14-element Vector{Float64}, ℵ₀-element FillArrays.Zeros{Float64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}} with indices OneToInf()) with indices OneToInf():
 1.2660658777520084
 1.1303182079849703
 0.27149533953407656
 0.04433684984866379
 0.0054742404420936785
 0.0005429263119139232
 4.497732295427654e-5
 ⋮

julia> f = T*c; f[0.1] # ≈ exp(0.1)
1.1051709180756477
```
With a little cheeky usage of Julia's order-of-operations this can be written succicently as:
```julia
julia> f = T / T \ exp.(x);

julia> f[0.1]
1.1051709180756477
```
(Or for more clarity just write `T * (T \ exp.(x))`.)


## Jacobi matrices

Orthogonal polynomials satisfy well-known three-term recurrences:
$$
x p_n(x) = c_{n-1} p_{n-1}(x) + a_n p_n(x) + b_n p_{n+1}(x).
$$
In continuum-array language this has the  form of a comuting relationship:
$$
x \begin{bmatrix} p_0 | p_1 | \cdots \end{bmatrix} = \begin{bmatrix} p_0 | p_1 | \cdots \end{bmatrix} \begin{bmatrix} a_0 & c_0  \\ b_0 & a_1 & c_1 \\ & b_1 & a_2 & \ddots \\ &&\ddots & \ddots \end{bmatrix}
$$
We can therefore find the Jacobi matrix naturally as follows:
```julia
julia> T \ (x .* T)
ℵ₀×ℵ₀ LazyBandedMatrices.Tridiagonal{Float64, LazyArrays.ApplyArray{Float64, 1, typeof(vcat), Tuple{Float64, FillArrays.Fill{Float64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}, FillArrays.Zeros{Float64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Fill{Float64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}} with indices OneToInf()×OneToInf():
 0.0  0.5   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   …  
 1.0  0.0  0.5   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋅   0.5  0.0  0.5   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋅    ⋅   0.5  0.0  0.5   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋅    ⋅    ⋅   0.5  0.0  0.5   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋅    ⋅    ⋅    ⋅   0.5  0.0  0.5   ⋅    ⋅    ⋅    ⋅    ⋅   …  
  ⋅    ⋅    ⋅    ⋅    ⋅   0.5  0.0  0.5   ⋅    ⋅    ⋅    ⋅      
 ⋮                        ⋮                        ⋮         ⋱  
```
Alternatively, just call `jacobimatrix(T)` (noting its the transpose of the more traditional convention).


## Derivatives

The derivatives of classical orthogonal polynomials are also classical OPs, and this can be seen as follows:
```julia
julia> U = ChebyshevU();

julia> D = Derivative(x);

julia> U\D*T
ℵ₀×ℵ₀ BandedMatrix{Float64} with bandwidths (-1, 1) with data 1×ℵ₀ adjoint(::InfiniteArrays.InfStepRange{Float64, Float64}) with eltype Float64 with indices Base.OneTo(1)×OneToInf() with indices OneToInf()×OneToInf():
  ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   …  
  ⋅    ⋅   2.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋅    ⋅    ⋅   3.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋅    ⋅    ⋅    ⋅   4.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋅    ⋅    ⋅    ⋅    ⋅   5.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   6.0   ⋅    ⋅    ⋅    ⋅    ⋅   …  
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   7.0   ⋅    ⋅    ⋅    ⋅      
 ⋮                        ⋮                        ⋮         ⋱  
```
Similarly, the derivative of _weighted_ classical OPs are weighted classical OPs:
```julia
lia> Weighted(T)\D*Weighted(U)
ℵ₀×ℵ₀ BandedMatrix{Float64} with bandwidths (1, -1) with data 1×ℵ₀ adjoint(::InfiniteArrays.InfStepRange{Float64, Float64}) with eltype Float64 with indices Base.OneTo(1)×OneToInf() with indices OneToInf()×OneToInf():
   ⋅     ⋅     ⋅     ⋅     ⋅     ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   …  
 -1.0    ⋅     ⋅     ⋅     ⋅     ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
   ⋅   -2.0    ⋅     ⋅     ⋅     ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
   ⋅     ⋅   -3.0    ⋅     ⋅     ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
   ⋅     ⋅     ⋅   -4.0    ⋅     ⋅    ⋅    ⋅    ⋅    ⋅    ⋅      
   ⋅     ⋅     ⋅     ⋅   -5.0    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   …  
   ⋅     ⋅     ⋅     ⋅     ⋅   -6.0   ⋅    ⋅    ⋅    ⋅    ⋅      
  ⋮                             ⋮                        ⋮    ⋱  
```

# Other recurrence relationships

Many other sparse recurrence relationships are implemented. Here's one:
```julia
julia> U\T
ℵ₀×ℵ₀ BandedMatrix{Float64} with bandwidths (0, 2) with data vcat(1×ℵ₀ FillArrays.Fill{Float64, 2, Tuple{Base.OneTo{Int64}, InfiniteArrays.OneToInf{Int64}}} with indices Base.OneTo(1)×OneToInf(), 1×ℵ₀ FillArrays.Zeros{Float64, 2, Tuple{Base.OneTo{Int64}, InfiniteArrays.OneToInf{Int64}}} with indices Base.OneTo(1)×OneToInf(), hcat(1×1 Ones{Float64}, 1×ℵ₀ FillArrays.Fill{Float64, 2, Tuple{Base.OneTo{Int64}, InfiniteArrays.OneToInf{Int64}}} with indices Base.OneTo(1)×OneToInf()) with indices Base.OneTo(1)×OneToInf()) with indices Base.OneTo(3)×OneToInf() with indices OneToInf()×OneToInf():
 1.0  0.0  -0.5    ⋅     ⋅     ⋅     ⋅     ⋅     ⋅    ⋅    ⋅   …  
  ⋅   0.5   0.0  -0.5    ⋅     ⋅     ⋅     ⋅     ⋅    ⋅    ⋅      
  ⋅    ⋅    0.5   0.0  -0.5    ⋅     ⋅     ⋅     ⋅    ⋅    ⋅      
  ⋅    ⋅     ⋅    0.5   0.0  -0.5    ⋅     ⋅     ⋅    ⋅    ⋅      
  ⋅    ⋅     ⋅     ⋅    0.5   0.0  -0.5    ⋅     ⋅    ⋅    ⋅      
  ⋅    ⋅     ⋅     ⋅     ⋅    0.5   0.0  -0.5    ⋅    ⋅    ⋅   …  
  ⋅    ⋅     ⋅     ⋅     ⋅     ⋅    0.5   0.0  -0.5   ⋅    ⋅      
 ⋮                            ⋮                           ⋮    ⋱  
```
(Probably best to ignore the type signature 😅)