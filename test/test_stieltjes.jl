using ClassicalOrthogonalPolynomials, ContinuumArrays, QuasiArrays, BandedMatrices, ArrayLayouts, LazyBandedMatrices, BlockArrays, Test
import ClassicalOrthogonalPolynomials: Hilbert, StieltjesPoint, ChebyshevInterval, associated, Associated,
        orthogonalityweight, Weighted, gennormalizedpower, *, dot, PowerLawMatrix, PowKernelPoint, LogKernelPoint,
        MemoryLayout, PaddedLayout
import InfiniteArrays: I

@testset "Associated" begin
    T = ChebyshevT()
    U = ChebyshevU()
    @test associated(T) ≡ U
    @test associated(U) ≡ U
    @test Associated(T)[0.1,1:10] == Associated(U)[0.1,1:10] == U[0.1,1:10]

    P = Legendre()
    Q = Associated(P)
    x = axes(P,1)
    u = Q * (Q \ exp.(x))
    @test u[0.1] ≈ exp(0.1)
    @test grid(Q[:,Base.OneTo(5)]) ≈ eigvals(Matrix(jacobimatrix(Normalized(Q))[1:5,1:5]))

    w = orthogonalityweight(Q)
    @test axes(w,1) == axes(P,1)
    @test sum(w) == 1
end

@testset "Singular integrals" begin
    @testset "weights" begin
        w_T = ChebyshevTWeight()
        w_U = ChebyshevUWeight()
        w_P = LegendreWeight()
        x = axes(w_T,1)
        H = inv.(x .- x')
        @test iszero(H*w_T)
        @test (H*w_U)[0.1] ≈ π/10
        @test (H*w_P)[0.1] ≈ log(1.1) - log(1-0.1)

        w_T = orthogonalityweight(chebyshevt(0..1))
        w_U = orthogonalityweight(chebyshevu(0..1))
        w_P = orthogonalityweight(legendre(0..1))
        x = axes(w_T,1)
        H = inv.(x .- x')
        @test iszero(H*w_T)
        @test (H*w_U)[0.1] ≈ (2*0.1-1)*π
        @test (H*w_P)[0.1] ≈ (log(1+(-0.8)) - log(1-(-0.8)))
    end

    @testset "LogKernelPoint" begin
        @testset "Complex point" begin
            wU = Weighted(ChebyshevU())
            x = axes(wU,1)
            z = 0.1+0.2im
            L = log.(abs.(z.-x'))
            @test L isa LogKernelPoint{Float64,ComplexF64,ComplexF64,Float64,ChebyshevInterval{Float64}}
        end

        @testset "Real point" begin
            U = ChebyshevU()
            x = axes(U,1)

            t = 2.0
            @test (log.(abs.(t .- x') )* Weighted(U))[1,1:3] ≈ [1.0362686329607178,-0.4108206734393296, -0.054364775221816465] #mathematica

            t = 0.5
            @test (log.(abs.(t .- x') )* Weighted(U))[1,1:3] ≈ [-1.4814921268505252, -1.308996938995747, 0.19634954084936207] #mathematica

            t = 0.5+0im
            @test (log.(abs.(t .- x') )* Weighted(U))[1,1:3] ≈ [-1.4814921268505252, -1.308996938995747, 0.19634954084936207] #mathematica
        end

        @testset "mapped" begin
            x = Inclusion(1..2)
            wU = Weighted(ChebyshevU())[affine(x, axes(ChebyshevU(),1)),:]
            x = axes(wU,1)
            z = 5
            L = log.(abs.(z .- x'))

            f = wU / wU \ @.(sqrt(2-x)sqrt(x-1)exp(x))
            @test L*f ≈ 2.2374312398976586 # MAthematica

            wU = Weighted(chebyshevu(1..2))
            f = wU / wU \ @.(sqrt(2-x)sqrt(x-1)exp(x))
            @test L*f ≈ 2.2374312398976586 # MAthematica
        end
    end

    @testset "Stieltjes" begin
        T = Chebyshev()
        wT = Weighted(T)
        x = axes(wT,1)
        z = 0.1+0.2im
        S = inv.(z .- x')
        @test S isa StieltjesPoint{ComplexF64,ComplexF64,Float64,ChebyshevInterval{Float64}}

        @test S * ChebyshevWeight() ≈ π/(sqrt(z-1)sqrt(z+1))
        @test S * JacobiWeight(0.1,0.2) ≈ 0.051643014475741864 - 2.7066092318596726im

        f = wT * [[1,2,3]; zeros(∞)];
        J = T \ (x .* T)

        @test π*((z*I-J) \ f.args[2])[1,1] ≈ (S*f)[1]
        @test π*((z*I-J) \ f.args[2])[1,1] ≈ (S*f.args[1]*f.args[2])[1]

        x = Inclusion(0..1)
        y = 2x .- 1
        wT2 = wT[y,:]
        S = inv.(z .- x')
        f = wT2 * [[1,2,3]; zeros(∞)];

        @test (π/2*(((z-1/2)*I - J/2) \ f.args[2]))[1] ≈ (S*f.args[1]*f.args[2])[1]

        @testset "Real point" begin
            t = 2.0
            T = ChebyshevT()
            U = ChebyshevU()
            x = axes(T,1)
            @test inv.(t .- x') * Weighted(T) ≈ inv.((t+eps()im) .- x') * Weighted(T)
            @test (inv.(t .- x') * Weighted(U))[1:10] ≈ (inv.((t+eps()im) .- x') * Weighted(U))[1:10]

            t = 2
            @test inv.(t .- x') * Weighted(T) ≈ inv.((t+eps()im) .- x') * Weighted(T)
            @test (inv.(t .- x') * Weighted(U))[1:10] ≈ (inv.((t+eps()im) .- x') * Weighted(U))[1:10]

            t = 0.5
            @test (inv.(t .- x') * Weighted(T))[1,1:3] ≈ [0,-π,-π]
            @test (inv.(t .- x') * Weighted(U))[1,1:3] ≈ [π/2,-π/2,-π]

            t = 0.5+0im
            @test (inv.(t .- x') * Weighted(T))[1,1:3] ≈ [0,-π,-π]
            @test (inv.(t .- x') * Weighted(U))[1,1:3] ≈ [π/2,-π/2,-π]
        end

        @testset "DimensionMismatch" begin
            x = Inclusion(0..1)
            z = 2.0
            @test_throws DimensionMismatch inv.(z .- x') * Weighted(ChebyshevT())
            @test_throws DimensionMismatch inv.(z .- x') * Weighted(ChebyshevU())
            @test_throws DimensionMismatch inv.(z .- x') * ChebyshevTWeight()
            @test_throws DimensionMismatch inv.(z .- x') * ChebyshevUWeight()
        end
    end

    @testset "Hilbert" begin
        wT = Weighted(ChebyshevT())
        wU = Weighted(ChebyshevU())
        x = axes(wT,1)
        H = inv.(x .- x')
        @test H isa Hilbert{Float64,ChebyshevInterval{Float64}}

        @testset "weights" begin
            @test H * ChebyshevTWeight() ≡ QuasiZeros{Float64}((x,))
            @test H * ChebyshevUWeight() == π*x
            @test (H * LegendreWeight())[0.1] ≈ log((0.1+1)/(1-0.1))
        end

        @test (Ultraspherical(1) \ (H*wT))[1:10,1:10] == diagm(1 => fill(-π,9))
        @test (Chebyshev() \ (H*wU))[1:10,1:10] == diagm(-1 => fill(1.0π,9))

        # check consistency
        @test (Ultraspherical(1) \ (H*wT) * (wT \ wU))[1:10,1:10] ==
                    ((Ultraspherical(1) \ Chebyshev()) * (Chebyshev() \ (H*wU)))[1:10,1:10]

        @testset "Other axes" begin
            x = Inclusion(0..1)
            y = 2x .- 1
            H = inv.(x .- x')

            wT2 = wT[y,:]
            wU2 = wU[y,:]
            @test (Ultraspherical(1)[y,:]\(H*wT2))[1:10,1:10] == diagm(1 => fill(-π,9))
            @test (Chebyshev()[y,:]\(H*wU2))[1:10,1:10] == diagm(-1 => fill(1.0π,9))
        end

        @testset "Legendre" begin
            P = Legendre()
            x = axes(P,1)
            H = inv.(x .- x')
            Q = H*P
            @test Q[0.1,1:3] ≈ [log(0.1+1)-log(1-0.1), 0.1*(log(0.1+1)-log(1-0.1))-2,-3*0.1 + 1/2*(-1 + 3*0.1^2)*(log(0.1+1)-log(1-0.1))]
            X = jacobimatrix(P)
            @test Q[0.1,1:11]'*X[1:11,1:10] ≈ (0.1 * Array(Q[0.1,1:10])' - [2 zeros(1,9)])
        end

        @testset "mapped" begin
            T = chebyshevt(0..1)
            U = chebyshevu(0..1)
            x = axes(T,1)
            H = inv.(x .- x')
            @test U\H*Weighted(T) isa BandedMatrix
        end
    end

    @testset "Log kernel" begin
        T = Chebyshev()
        wT = Weighted(Chebyshev())
        x = axes(wT,1)
        L = log.(abs.(x .- x'))
        D = T \ (L * wT)
        @test ((L * wT) * (T \ exp.(x)))[0.] ≈ -2.3347795490945797  # Mathematica

        x = Inclusion(-1..1)
        T = Chebyshev()[1x, :]
        L = log.(abs.(x .- x'))
        wT = Weighted(Chebyshev())[1x, :]
        @test (T \ (L*wT))[1:10,1:10] ≈ D[1:10,1:10]

        x = Inclusion(0..1)
        T = Chebyshev()[2x.-1, :]
        wT = Weighted(Chebyshev())[2x .- 1, :]
        L = log.(abs.(x .- x'))
        u =  wT * (2 *(T \ exp.(x)))
        @test u[0.1] ≈ exp(0.1)/sqrt(0.1-0.1^2)
        @test (L * u)[0.5] ≈ -7.471469928754152 # Mathematica

        @testset "mapped" begin
            T = chebyshevt(0..1)
            x = axes(T,1)
            L = log.(abs.(x .- x'))
            @test T[0.2,:]'*((T\L*Weighted(T)) * (T\exp.(x))) ≈ -2.9976362326874373 # Mathematica
        end
    end

    @testset "pow kernel" begin
        P = Weighted(Jacobi(0.1,0.2))
        x = axes(P,1)
        S = abs.(x .- x').^0.5
        @test S isa ClassicalOrthogonalPolynomials.PowKernel
        @test_broken S*P
    end

    @testset "Ideal Fluid Flow" begin
        T = ChebyshevT()
        U = ChebyshevU()
        x = axes(U,1)
        H = inv.(x .- x')

        c = exp(0.5im)
        u = Weighted(U) * ((H * Weighted(U)) \ imag(c * x))

        ε  = eps();
        @test (inv.(0.1+ε*im .- x') * u + inv.(0.1-ε*im .- x') * u)/2 ≈ imag(c*0.1)
        @test real(inv.(0.1+ε*im .- x') * u ) ≈ imag(c*0.1)

        v = (s,t) -> (z = (s + im*t); imag(c*z) - real(inv.(z .- x') * u))
        @test v(0.1,0.2) ≈ 0.18496257285081724 # Emperical
    end

    @testset "OffHilbert" begin
        @testset "ChebyshevU" begin
            U = ChebyshevU()
            W = Weighted(U)
            t = axes(U,1)
            x = Inclusion(2..3)
            T = chebyshevt(2..3)
            H = T \ inv.(x .- t') * W;

            @test MemoryLayout(H) isa PaddedLayout

            @test last(colsupport(H,1)) ≤ 20
            @test last(colsupport(H,6)) ≤ 40
            @test last(rowsupport(H)) ≤ 30
            @test T[2.3,1:100]'*(H * (W \ @.(sqrt(1-t^2)exp(t))))[1:100] ≈ 0.9068295340935111
            @test T[2.3,1:100]' * H[1:100,1:100] ≈ (inv.(2.3 .- t') * W)[:,1:100]

            u = (I + H) \ [1; zeros(∞)]
            @test u[3] ≈ -0.011220808241213699 #Emperical


            @testset "properties" begin
                U  = chebyshevu(T)
                X = jacobimatrix(U)
                Z = jacobimatrix(T)

                @test Z * H[:,1] - H[:,2]/2 ≈ [sum(W[:,1]); zeros(∞)]
                @test norm(-H[:,1]/2 + Z * H[:,2] - H[:,3]/2) ≤ 1E-12

                L = U \ ((x.^2 .- 1) .* Derivative(x) * T - x .* T)
                c = T \ sqrt.(x.^2 .- 1)
                @test [T[begin,:]'; L] \ [sqrt(2^2-1); zeros(∞)] ≈ c
            end
        end

        @testset "mapped" begin
            U = chebyshevu(-1..0)
            W = Weighted(U)
            t = axes(U,1)
            x = Inclusion(2..3)
            T = chebyshevt(2..3)
            H = T \ inv.(x .- t') * W
            N = 100
            @test T[2.3,1:N]' * H[1:N,1:N] ≈ (inv.(2.3 .- t') * W)[:,1:N]

            U = chebyshevu((-2)..(-1))
            W = Weighted(U)
            T = chebyshevt(0..2)
            x = axes(T,1)
            t = axes(W,1)
            H = T \ inv.(x .- t') * W
            @test T[0.5,1:N]'*(H * (W \ @.(sqrt(-1-t)*sqrt(t+2)*exp(t))))[1:N] ≈ 0.047390454610749054
        end
    end

    @testset "two-interval" begin
        T1,T2 = chebyshevt((-2)..(-1)), chebyshevt(0..2)
        U1,U2 = chebyshevu((-2)..(-1)), chebyshevu(0..2)
        W = PiecewiseInterlace(Weighted(U1), Weighted(U2))
        T = PiecewiseInterlace(T1, T2)
        U = PiecewiseInterlace(U1, U2)
        x = axes(W,1)
        H = T \ inv.(x .- x') * W;

        @test iszero(H[1,1])
        @test H[3,1] ≈ π
        @test maximum(BlockArrays.blockcolsupport(H,Block(5))) ≤ Block(50)
        @test blockbandwidths(H) == (25,26)

        c = W \ broadcast(x -> exp(x)* (0 ≤ x ≤ 2 ? sqrt(2-x)*sqrt(x) : sqrt(-1-x)*sqrt(x+2)), x)
        f = W * c
        @test T[0.5,1:200]'*(H*c)[1:200] ≈ -6.064426633490422

        @testset "inversion" begin
            H̃ = BlockHcat(Eye((axes(H,1),))[:,Block(1)], H)
            @test BlockArrays.blockcolsupport(H̃,Block(1)) == Block.(1:1)
            @test last(BlockArrays.blockcolsupport(H̃,Block(2))) ≤ Block(30)

            UT = U \ T
            D = U \ Derivative(x) * T
            V = x -> x^4 - 10x^2
            Vp = x -> 4x^3 - 20x
            V_cfs = T \ V.(x)
            Vp_cfs_U = D * V_cfs
            Vp_cfs_T = T \ Vp.(x);

            @test (UT \ Vp_cfs_U)[Block.(1:10)] ≈ Vp_cfs_T[Block.(1:10)]

            @time c = H̃ \ Vp_cfs_T;

            @test c[Block.(1:100)] ≈ H̃[Block.(1:100),Block.(1:100)] \ Vp_cfs_T[Block.(1:100)]

            E1,E2 = c[Block(1)]
            @test [E1,E2] ≈  [12.939686758642496,-10.360345667126758]
            c1 = [paddeddata(c)[3:2:end]; Zeros(∞)]
            c2 = [paddeddata(c)[4:2:end]; Zeros(∞)]

            u1 = Weighted(U1) * c1
            u2 = Weighted(U2) * c2
            x1 = axes(u1,1)
            x2 = axes(u2,1)

            @test inv.(-1.3 .- x1') * u1 + inv.(-1.3 .- x2') * u2 + E1 ≈ Vp(-1.3)
            @test inv.(1.3 .- x1') * u1 + inv.(1.3 .- x2') * u2 + E2 ≈ Vp(1.3)
        end

        @testset "Stieltjes" begin
            z = 5.0
            @test inv.(z .- x')*f ≈ 1.317290060427562
            @test log.(abs.(z .- x'))*f ≈ 6.523123127595374
            @test log.(abs.((-z) .- x'))*f ≈ 8.93744698863906

            t = 1.2
            @test inv.(t .- x')*f ≈ -2.797995066227555
            @test log.(abs.(t .- x'))*f ≈ -5.9907385495482821485
        end
    end

    @testset "three-interval" begin
        d = (-2..(-1), 0..1, 2..3)
        T = PiecewiseInterlace(chebyshevt.(d)...)
        U = PiecewiseInterlace(chebyshevu.(d)...)
        W = PiecewiseInterlace(Weighted.(U.args)...)
        x = axes(W,1)
        H = T \ inv.(x .- x') * W
        c = W \ broadcast(x -> exp(x) *
            if -2 ≤ x ≤ -1
                sqrt(x+2)sqrt(-1-x)
            elseif 0 ≤ x ≤ 1
                sqrt(1-x)sqrt(x)
            else
                sqrt(x-2)sqrt(3-x)
            end, x)
        f = W * c
        @test T[0.5,1:200]'*(H*c)[1:200] ≈ -3.0366466972156143
    end

    #################################################
    # ∫f(x)g(x)(t-x)^a dx evaluation where f and g in Legendre
    #################################################

    @testset "Pow kernel" begin
        @testset "Multiplication methods" begin
            P = Normalized(Legendre())
            x = axes(P,1)
            for (a,t) in ((0.1,1.2), (0.5,1.5))
                @test (t.-x).^a isa PowKernelPoint
                w = (t.-x).^a
                @test w .* P isa typeof(P*PowerLawMatrix(P,a,t))
            end
            # some functions
            f = P \ exp.(x.^2)
            g = P \ (sin.(x).*exp.(x.^(2)))
            # some parameters for (t-x)^a
            a = BigFloat("1.23")
            t = BigFloat("1.00001")
            # define powerlaw multiplication
            w = (t.-x).^a

            # check if it can compute the integral correctly
            @test g'*(P'*(w.*P)*f) ≈ -2.656108697646584 # Mathematica
        end
        @testset "Equivalence to multiplication in integer case" begin
            # TODO: overload integer input to make this work
            P = Normalized(Legendre())
            x = axes(P,1)
            a = 1
            t = 1.2
            @test_broken PowerLawMatrix(P,Float64(a),t)[1:20,1:20] ≈ ((t*I-jacobimatrix(P))^a)[1:20,1:20]
            a = 2
            t = 1.0001
            J = ((t*I-jacobimatrix(P)))[1:80,1:80]
            @test_broken PowerLawMatrix(P,BigFloat("$a"),BigFloat("$t"))[1:60,1:60] ≈ (J^2)[1:60,1:60]
        end
        @testset "Cached Legendre power law integral operator" begin
            P = Normalized(Legendre())
            a = 2*rand(1)[1]
            t = 1.0000000001
            Acached = PowerLawMatrix(P,BigFloat("$a"),BigFloat("$t"))
            @test size(Acached) == (∞,∞)
        end
        @testset "PowKernelPoint dot evaluation" begin
            @testset "Set 1" begin
                    P = Normalized(Legendre())
                    x = axes(P,1)
                    f = P \ abs.(π*x.^7)
                    g = P \ (cosh.(x.^3).*exp.(x.^(2)))
                    a = 1.9127
                    t = 1.211
                    w = (BigFloat("$t") .- x).^BigFloat("$a")
                    Pw = P'*(w .* P)
                    @test w isa PowKernelPoint
                    @test Pw[1:20,1:20] ≈ PowerLawMatrix(P,a,t)[1:20,1:20]
                    # this is slower than directly using PowerLawMatrix but it works
                    @test dot(f[1:20],Pw[1:20,1:20],g[1:20]) ≈ 5.082145576355614 # Mathematica
                end
            @testset "Set 2" begin
                P = Normalized(Legendre())
                x = axes(P,1)
                f = P \ exp.(x.^2)
                g = P \ (sin.(x).*exp.(x.^(2)))
                a = 1.23
                t = 1.00001
                W = PowerLawMatrix(P,a,t)
                @test dot(f,W,g) ≈ -2.656108697646584 # Mathematica
            end
            @testset "Set 3" begin
                P = Normalized(Legendre())
                x = axes(P,1)
                t = 1.2
                a = 1.1
                W = PowerLawMatrix(P,a,t)
                f = P \ exp.(x)
                g = P \ exp.(x.^2)
                @test dot(f,W,g) ≈ 2.916955525390389 # Mathematica
            end
            @testset "Set 4" begin
                P = Normalized(Legendre())
                x = axes(P,1)
                t = 1.001
                a = 1.001
                W = PowerLawMatrix(P,a,t)
                f = P \ (sinh.(x).*exp.(x))
                g = P \ cos.(x.^3)
                @test dot(f,W,g) ≈ -0.1249375144525209 # Mathematica
            end
            @testset "More explicit evaluation tests" begin
                # basis
                a = 2.9184
                t = 1.000001
                P = Normalized(Legendre())
                x = axes(P,1)
                # operator
                W = PowerLawMatrix(P,a,t)
                # functions
                f = P \ exp.(x)
                g = P \ sin.(x)
                const1(x) = 1
                onevec = P \ const1.(x)
                # dot() and * methods tests, explicit values via Mathematica
                @test -2.062500116206712 ≈ dot(onevec,W,g)
                @test 2.266485452423447 ≈ dot(onevec,W,f)
                @test -0.954305839543464 ≈ dot(g,W,f)
                @test 1.544769699288028 ≈ dot(f,W,f)
                @test 1.420460011606107 ≈ dot(g,W,g)
            end
        end
        @testset "Tests for -1 < a < 0" begin
            P = Normalized(Legendre())
            x = axes(P,1)
            a = -0.7
            t = 1.271
            # operator
            W = PowerLawMatrix(P,a,t)
            WB = PowerLawMatrix(P,BigFloat("$a"),BigFloat("$t"))
            # functions
            f0 = P \ exp.(2 .*x.^2)
            g0 = P \ sin.(x)
            @test dot(f0,W,g0) ≈  dot(f0,WB,g0) ≈ 1.670106472636101 # Mathematica
            f1 = P \ ((x.^2)./3 .+(x.^3)./3)
            g1 = P \ (x.*exp.(x.^3))
            @test dot(f1,W,g1) ≈ dot(f1,WB,g1) ≈ 0.5362428541997497 # Mathematica
        end
        @testset "Lanczos" begin
            P = Normalized(Legendre())
            x = axes(P,1)
            @time D = ClassicalOrthogonalPolynomials.LanczosData((1.001 .- x).^0.5, P);
            @time ClassicalOrthogonalPolynomials.resizedata!(D,100);
        end
    end
end


import LazyArrays: resizedata!
import ClassicalOrthogonalPolynomials: sqrtx2

T = ChebyshevT()
wT = Weighted(T)
x = axes(T,1)
z = range(2, 3; length=10); S = inv.(z .- x'); @time S*wT;

P = wT
# since we build column-by-column its better to construct the transpose of the returned result
z = S.args[1].args[1] # vector of points to eval at
x,ns = axes(P)
m = length(z)
n = length(ns)
ret = zeros(n, m) # transpose as we fill column-by-column

# TODO: estimate number of entries based on exact 
r = minimum(abs, z)




r = 100.0
ξ = inv(r + sqrtx2(r))
k = ceil(Int,log(eps())/log(ξ))
(inv.(r .- x') *P)[k]/(inv.(r .- x') *P)[1]

k*log(ξ) ≤ log(ε)

import ClassicalOrthogonalPolynomials: sqrtx2
z = 2.0
ξ = inv(z + sqrtx2(z))
ξ

plot(abs.((inv.(2 .- x') * P)[1:100]); yscale=:log10)

recurrencecoefficients(T)


b = (5:2:∞) ./ (6:2:∞)

function f(b, N)
    ret = 0.0
    @inbounds for k = 1:N
        ret += b[k]
    end
    ret
end

function g(N)
    ret = 0.0
    @inbounds for k = 1:N
        ret += (2k+3)/(2k+4)
    end
    ret
end
