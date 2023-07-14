using ClassicalOrthogonalPolynomials, Test
import ClassicalOrthogonalPolynomials: orthogonalityweight

@testset "Laguerre" begin
    @testset "Laguerre weight" begin
        w = LaguerreWeight()
        @test sum(w) ≡ 1.0
        @test w[0.1] ≈ orthogonalityweight(Laguerre())[0.1] ≈ exp(-0.1)
        @test LaguerreWeight(1/2)[0.1] ≈ sqrt(0.1)exp(-0.1)

        @test AbstractQuasiArray{Float32}(w) ≡ AbstractQuasiVector{Float32}(w) ≡ LaguerreWeight{Float32}()
    end

    @testset "Laguerre L" begin
        L = Laguerre()
        @test AbstractQuasiArray{Float32}(L) ≡ AbstractQuasiMatrix{Float32}(L) ≡ Laguerre{Float32}()
        
        x = 0.1
        @test laguerrel.(0:3, 0, x) ≈ laguerrel.(0:3, x) ≈ [1, 1-x, (2 - 4x + x^2)/2, (6 - 18x + 9x^2 - x^3)/6]
        @test laguerrel.(0:3, 1/2, x) ≈ [1, 3/2 - x, (15 - 20x + 4x^2)/8, (105 - 210x + 84x^2 - 8x^3)/48]
    end
    @testset "Derivatives" begin
        L = Laguerre()
        D = Derivative(axes(L,1))
        x = 0.1
        @test (D*L)[x,1:4] ≈ [0,-1,x-2,-x^2/2 + 3x - 3]

        L = Laguerre(1/2)
        @test (D*L)[x,1:4] ≈ [0,-1,x-20/8,-x^2/2 + 7/2*x - 35/8]

        x = axes(L,1)
        X = L \ (x .* L)
        @test 0.1*L[0.1,1:10]' ≈ L[0.1,1:11]'*X[1:11,1:10]
    end
end