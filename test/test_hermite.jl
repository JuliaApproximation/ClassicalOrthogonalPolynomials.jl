using ClassicalOrthogonalPolynomials, ContinuumArrays, FillArrays, Test
import ClassicalOrthogonalPolynomials: jacobimatrix, oneto, OrthonormalWeighted
import DomainSets: ℝ

@testset "Hermite" begin    
    @testset "Basics" begin
        H = Hermite()
        w = HermiteWeight()
        @test axes(H) == (Inclusion(ℝ), oneto(∞))
        x = axes(H,1)
        @test H[0.1,1:4] ≈ hermiteh.(0:3,0.1) ≈ [1,2*0.1,4*0.1^2-2,8*0.1^3-12*0.1]
        @test w[0.1] ≈ exp(-0.1^2)

        X = jacobimatrix(H)
        @test 0.1 * H[0.1,1:10]' ≈ H[0.1,1:11]'*X[1:11,1:10]

        @test (H'*(w .* H))[1,1] ≈ sqrt(π)
        @test (H'*(w .* H))[2,2] ≈ 2sqrt(π)
        @test (H'*(w .* H))[3,3] ≈ 8sqrt(π)

        D = Derivative(x)
        @test (D*H)[0.1,1:4] ≈ [0,2,8*0.1,24*0.1^2-12]
    end

    @testset "Weighted" begin
        H = Hermite()
        W = Weighted(H)
        x = axes(W,1)
        D = Derivative(x)
        @test (D*W)[0.1,1:4] ≈ [-2*0.1, 2-4*0.1^2, 12*0.1 - 8*0.1^3, -4*(3 - 12*0.1^2 + 4*0.1^4)]*exp(-0.1^2)
    end

    @testset "OrthonormalWeighted" begin
        H = Hermite()
        Q = OrthonormalWeighted(H)
        @testset "evaluation" begin
            x = 0.1
            @test Q[x,1] ≈ exp(-x^2/2)/π^(1/4)
            @test Q[x,2] ≈ 2x*exp(-x^2/2)/(sqrt(2)π^(1/4))
            # Trap test of L^2 orthonormality
            x = range(-20,20; length=1_000)
            @test Q[x,1:5]'Q[x,1:5] * step(x) ≈ I
        end
        
        @testset "Differentiation" begin
            x = axes(Q,1)
            D = Derivative(x)
            D¹ = Q \ (D * Q)
            @test D¹[1:10,1:10] ≈ -D¹[1:10,1:10]'
            D² = Q \ (D^2 * Q)
            X = Q \ (x .* Q)
            @test (D² - X^2)[1:10,1:10] ≈ -Diagonal(1:2:19)
        end
    end
end