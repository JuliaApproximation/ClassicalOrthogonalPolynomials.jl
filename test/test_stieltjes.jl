using ClassicalOrthogonalPolynomials, ContinuumArrays, Test
import ClassicalOrthogonalPolynomials: Hilbert, StieltjesPoint, ChebyshevInterval
@testset "Stieltjes" begin
    T = Chebyshev()
    wT = ChebyshevWeight() .* T
    x = axes(wT,1)
    z = 0.1+0.2im
    S = inv.(z .- x')
    @test S isa StieltjesPoint{ComplexF64,Float64,ChebyshevInterval{Float64}}
    f = wT * [[1,2,3]; zeros(∞)];
    J = T \ (x .* T)
    # TODO: fix LazyBandedMatrices.copy(M::Mul{Broadcast...}) to call simplify
    @test_broken π*((z*I-J) \ f.args[2])[1,1] ≈ (S*f)[1]
    @test π*((z*I-J) \ f.args[2])[1,1] ≈ (S*f.args[1]*f.args[2])[1]

    x = Inclusion(0..1)
    y = 2x .- 1
    wT2 = wT[y,:]
    S = inv.(z .- x')
    f = wT2 * [[1,2,3]; zeros(∞)];

    @test (π/2*(((z-1/2)*I - J/2) \ f.args[2]))[1] ≈ (S*f.args[1]*f.args[2])[1]
end

@testset "Hilbert" begin
    wT = ChebyshevTWeight() .* ChebyshevT()
    wU = ChebyshevUWeight() .* ChebyshevU()
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
    @test_broken (Chebyshev()[y,:]\(H*wU2))[1:10,1:10] == diagm(-1 => fill(1.0π,9))
end

@testset "Log kernel" begin
    T = Chebyshev()
    wT = ChebyshevWeight() .* Chebyshev()
    x = axes(wT,1)
    L = log.(abs.(x .- x'))
    D = T \ (L * wT)
    @test ((L * wT) * (T \ exp.(x)))[0.] ≈ -2.3347795490945797  # Mathematica

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
    @test_broken (L * u)[0.5] ≈ -7.471469928754152 # Mathematica
end
