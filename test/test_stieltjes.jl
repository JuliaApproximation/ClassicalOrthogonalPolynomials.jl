using ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets, Test
import ClassicalOrthogonalPolynomials: Hilbert, StieltjesPoint, PowerLawIntegral, PowKernelIntegralPoint, pointwisecoeffmatrixdense, *, dot

@testset "Stieltjes" begin
    T = Chebyshev()
    wT = ChebyshevWeight() .* T
    x = axes(wT,1)
    z = 0.1+0.2im
    S = inv.(z .- x')
    @test S isa StieltjesPoint{ComplexF64,Float64,ChebyshevInterval{Float64}}
    f = wT * [[1,2,3]; zeros(∞)];
    J = T \ (x .* T)
    @test π*((z*I-J) \ f.args[2])[1,1] ≈ (S*f)[1]

    x = Inclusion(0..1)
    y = 2x .- 1
    wT2 = wT[y,:]
    S = inv.(z .- x')
    f = wT2 * [[1,2,3]; zeros(∞)];
    
    @test (π/2*(((z-1/2)*I - J/2) \ f.args[2]))[1] ≈ (S*f)[1]
end

@testset "Hilbert" begin
    wT = ChebyshevWeight() .* Chebyshev()
    wU = UltrasphericalWeight(1) .*  Ultraspherical(1)
    x = axes(wT,1)
    H = inv.(x .- x')
    @test H isa Hilbert{Float64,ChebyshevInterval{Float64}}

    @test (Ultraspherical(1) \ (H*wT))[1:10,1:10] == diagm(1 => fill(-π,9))
    @test (Chebyshev() \ (H*wU))[1:10,1:10] == diagm(-1 => fill(1.0π,9))

    # check consistency
    @test (Ultraspherical(1) \ (H*wT) * (wT \ wU))[1:10,1:10] == 
                ((Ultraspherical(1) \ Chebyshev()) * (Chebyshev() \ (H*wU)))[1:10,1:10]

    # Other axes
    x = Inclusion(0..1)
    y = 2x .- 1
    H = inv.(x .- x')

    wT2 = wT[y,:]
    wU2 = wU[y,:]
    @test (Ultraspherical(1)[y,:]\(H*wT2))[1:10,1:10] == diagm(1 => fill(-π,9))
    @test (Chebyshev()[y,:]\(H*wU2))[1:10,1:10] == diagm(-1 => fill(1.0π,9))
end

@testset "Log kernel" begin
    T = Chebyshev()
    wT = ChebyshevWeight() .* Chebyshev()
    x = axes(wT,1)
    L = log.(abs.(x .- x'))
    D = T \ (L * wT)
    @test (L * (wT * (T \ exp.(x))))[0.] ≈ -2.3347795490945797  # Mathematica

    x = Inclusion(-1..1)
    T = Chebyshev()[1x, :]
    L = log.(abs.(x .- x'))
    wT = (ChebyshevWeight() .* Chebyshev())[1x, :]
    @test (T \ (L*wT))[1:10,1:10] ≈ D[1:10,1:10]

    x = Inclusion(0..1)
    T = Chebyshev()[2x.-1, :]
    wT = (ChebyshevWeight() .* Chebyshev())[2x .- 1, :]
    L = log.(abs.(x .- x'))
    u =  wT * (2 *(T \ exp.(x)))
    @test u[0.1] ≈ exp(0.1)/sqrt(0.1-0.1^2)
    @test (L * u)[0.5] ≈ -7.471469928754152 # Mathematica
end

#################################################
# ∫f(x)g(x)(t-x)^a dx evaluation where f and g in Legendre
#################################################

@testset "dot() for PowerKernelPoint of Legendre" begin
    @testset "Cached Legendre power law integral" begin
        a = 2*rand(1)[1]
        t = 1.0000000001
        Acached = PowerLawIntegral(Legendre(),a,t)
        @test size(Acached) == (∞,∞)
        @test Acached[1:20,1:20] == pointwisecoeffmatrixdense(a,t,20)
    end
    @testset "PowKernelPoint dot evaluation finite" begin
        # basis
        P = Legendre()
        x = axes(P,1)
        # set 1, same length coefficient vectors
        f = P \ abs.(π*x.^7)
        g = P \ (cosh.(x.^3).*exp.(x.^(2)))
        f = f[1:30]
        g = g[1:30]
        a = 1.9127
        t = 1.211
        W = P' * ((t .- x).^a .* P)
        @test W isa PowKernelIntegralPoint
        @test dot(f,W,g) ≈ 5.082145576355614 # Mathematica
        # set 2, different length coefficient vectors
        f = P \ exp.(x.^2)
        g = P \ (sin.(x).*exp.(x.^(2)))
        f = f[1:20]
        g = g[1:40]
        a = 1.23
        t = 1.00001
        W = P' * ((t .- x).^a .* P)
        @test W isa PowKernelIntegralPoint
        @test dot(f,W,g) ≈ -2.656108697646584 # Mathematica
    end
    @testset "PowKernelPoint dot evaluation infinite" begin
        # basis
        P = Legendre()
        x = axes(P,1)
        # set 1
        f = P \ abs.(π*x.^7)
        g = P \ (cosh.(x.^3).*exp.(x.^(2)))
        a = 1.9127
        t = 1.211
        W = P' * ((t .- x).^a .* P)
        @test W isa PowKernelIntegralPoint
        @test dot(f,W,g) ≈ 5.082145576355614 # Mathematica
        # set 2
        f = P \ exp.(x.^2)
        g = P \ (sin.(x).*exp.(x.^(2)))
        a = 1.23
        t = 1.00001
        W = P' * ((t .- x).^a .* P)
        @test W isa PowKernelIntegralPoint
        @test dot(f,W,g) ≈ -2.656108697646584 # Mathematica
       # set 3
        t = 1.2
        a = 1.1
        W = P' * ((t .- x).^a .* P)
        f = P \ exp.(x)
        g = P \ exp.(x.^2)
        @test W isa PowKernelIntegralPoint
        @test dot(f,W,g) ≈ 2.916955525390389 # Mathematica
        # set 4
        t = 1.001
        a = 1.001
        W = P' * ((t .- x).^a .* P)
        f = P \ (sinh.(x).*exp.(x))
        g = P \ cos.(x.^3)
        @test W isa PowKernelIntegralPoint
        @test dot(f,W,g) ≈ -0.1249375144525209 # Mathematica
    end
    @testset "more explicit evaluation tests" begin
        # basis
        a = 2.9184
        t = 1.000001
        P = Legendre()
        x = axes(P,1)
            # operator
        W = P' * ((t .- x).^a .* P)
        @test W isa PowKernelIntegralPoint
            # functions
        f = P \ exp.(x)
        g = P \ sin.(x)
        const1(x) = 1
        onevec = P \ const1.(x)
            # dot() and * methods tests, explicit values via Mathematica
        @test -2.062500116206712 ≈ dot(onevec,W,g) == onevec'*W*g
        @test 2.266485452423447 ≈ dot(onevec,W,f) == onevec'*W*f
        @test -0.954305839543464 ≈ g'*W*f == dot(g,W,f)
        @test 1.544769699288028 ≈ f'*W*f == dot(f,W,f)
        @test 1.420460011606107 ≈ g'*W*g == dot(g,W,g)
    end
end