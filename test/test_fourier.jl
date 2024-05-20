using ClassicalOrthogonalPolynomials, QuasiArrays, InfiniteArrays
using BlockBandedMatrices, BlockArrays, LazyArrays, Test, FillArrays, InfiniteLinearAlgebra, ArrayLayouts, LazyBandedMatrices
import BlockBandedMatrices: _BlockBandedMatrix
import ClassicalOrthogonalPolynomials: Inclusion, ShuffledR2HC, ShuffledFFT
import QuasiArrays: MulQuasiArray

@testset "Fourier" begin
    @testset "ShuffledR2HC" begin
        n = 5
        θ = range(0,2π; length=n+1)[1:end-1]
        p = ShuffledR2HC{Float64}(n)
        @test p * one.(θ) ≈ [1; zeros(4)]
        @test p * sin.(θ) ≈ [0; 1; zeros(3)]
        @test p * cos.(θ) ≈ [zeros(2); 1; zeros(2)]

        @test p \ [1; zeros(4)] ≈ ones(5)
        @test p \ [0; 1; zeros(3)] ≈ sin.(θ)
        @test p \ [zeros(2); 1; zeros(2)] ≈ cos.(θ)

        ret = randn(5)
        @test p \ (p * ret) ≈ p * (p \ ret) ≈ ret

        ret = randn(3,5)
        p = ShuffledR2HC{Float64}(size(ret,1))
        P = ShuffledR2HC{Float64}(size(ret),1)
        @test size(p) == (3,)
        @test size(P) == (3,5)
        for k = 1:size(ret,2)
            @test (P * ret)[:,k] ≈ p * ret[:,k]
        end
        @test P \ (P*ret) ≈ P*(P\ret) ≈ ret

        p = ShuffledR2HC{Float64}(size(ret,2))
        P = ShuffledR2HC{Float64}(size(ret),2)
        for k = 1:size(ret,1)
            @test (P * ret)[k,:] ≈ p * ret[k,:]
        end
        @test P \ (P*ret) ≈ P*(P\ret) ≈ ret
    end
    
    @testset "Evaluation" begin
        F = Fourier()

        @test F == F

        @test axes(F,2) isa BlockedOneTo{Int,InfiniteArrays.InfStepRange{Int,Int}}
        @test axes(F) isa Tuple{<:Inclusion,BlockedOneTo{Int,InfiniteArrays.InfStepRange{Int,Int}}}
        @test axes(F,2)[Block(2)] == 2:3
        @test F[0.1,1] == 1.0
        @test F[0.1,2] == sin(0.1)
        @test F[0.1,1:4] == [1,sin(0.1),cos(0.1),sin(2*0.1)]
        @test F[0.1,Block(4)] == [sin(3*0.1),cos(3*0.1)]
        @test F[0.1,Block.(1:3)] == [1,sin(0.1),cos(0.1),sin(2*0.1),cos(2*0.1)]

        u = F * BlockedVector([[1,2,3]; zeros(∞)], (axes(F,2),));
        @test u[0.1] == 1 + 2sin(0.1) + 3cos(0.1)
    end

    @testset "Transform" begin
        F = Fourier()
        θ = axes(F,1)
        @test F[:,Base.OneTo(5)] \ cos.(θ) ≈ [0,0,1,0,0]
        @test F[:,Block.(Base.OneTo(5))] \ cos.(θ) ≈ [0,0,1,0,0,0,0,0,0]

        @test (F \ cos.(θ))[Block(2)] ≈ [0,1]
        u = F * (F \ exp.(cos.(θ)))
        @test u[0.1] ≈ exp(cos(0.1))
        @test F[:,Base.OneTo(5)] \ [cos.(θ) sin.(θ)] ≈ [0 0; 0 1; 1 0; 0 0; 0 0]
        @test F[:,Block.(Base.OneTo(3))] \ [cos.(θ) sin.(θ)] ≈ [0 0; 0 1; 1 0; 0 0; 0 0]
        U = F / F \ [exp.(cos.(θ)) cos.(cos.(θ))]
        @test U[0.1,:] ≈ [exp(cos(0.1)),cos(cos(0.1))]
        @test U[[0.1,0.2],:] ≈ [exp(cos(0.1)) cos(cos(0.1)); exp(cos(0.2)) cos(cos(0.2))]
        @test U[0.1,1] ≈ exp(cos(0.1))
        @test U[[0.1,0.2],1] ≈ exp.(cos.([0.1,0.2]))
    end

    @testset "Derivative" begin
        F = Fourier()
        @test F\F === Eye((axes(F,2),))
        @test (F'F)[1:10,1:10] == Diagonal(Vcat(2π,Fill(1.0π,9)))
        @test blockisequal(axes(F'F), (axes(F,2),axes(F,2)))
        D = Derivative(axes(F,1))
        D̃ = (D*F).args[2]
        @test (F\F)*D̃ isa BlockArray
        @test (F \ (D*F))[Block.(1:3),Block.(1:3)] == [0 0 0 0 0; 0 0.0 -1 0 0; 0 1 0 0 0; 0 0 0 0 -2; 0 0 0 2 0]

        u = F * BlockedVector([[1,2,3,4,5]; zeros(∞)], (axes(F,2),));
        @test blockisequal(axes(D̃,2),axes(u.args[2],1))
        @test (D*u)[0.1] ≈ 2cos(0.1) - 3sin(0.1) + 8cos(2*0.1) - 10sin(2*0.1)
    end

    @testset "cos(θ)" begin
        F = Fourier()
        θ = axes(F,1)
        c,s = cos.(θ),sin.(θ)
        @test c .* F isa MulQuasiArray
        @test s .* F isa MulQuasiArray
        @test cos.(θ) .* F isa MulQuasiArray
        @test sin.(θ) .* F isa MulQuasiArray
        X,Y = F \ (c .* F),F \ (s .* F)
        N = 10
        XY = X*Y
        @test XY isa MulMatrix
        @test MemoryLayout(XY) isa LazyBandedMatrices.ApplyBlockBandedLayout{typeof(*)}
        @test MemoryLayout(view(XY,Block.(1:N),Block.(1:N))) isa LazyBandedMatrices.ApplyBlockBandedLayout{typeof(*)}
        @test XY[Block.(1:N),Block.(1:N)] isa BlockSkylineMatrix
        @test blockbandwidths(XY[Block.(1:N),Block.(1:N)]) == (2,2)
        @test XY[Block.(1:N),Block.(1:N)] ≈ ApplyMatrix(*,Y,X)[Block.(1:N),Block.(1:N)]
        @test ApplyMatrix(*,X,X)[Block.(1:N),Block.(1:N)] + ApplyMatrix(*,Y,Y)[Block.(1:N),Block.(1:N)] ≈ Eye(19)
    end
end


@testset "Laurent" begin
    @testset "ShuffledFFT" begin
        ret = randn(3,5)
        p = ShuffledFFT{ComplexF64}(size(ret,1))
        P = ShuffledFFT{ComplexF64}(size(ret),1)
        @test size(p) == (3,)
        @test size(P) == (3,5)
        for k = 1:size(ret,2)
            @test (P * ret)[:,k] ≈ p * ret[:,k]
        end

        p = ShuffledFFT{ComplexF64}(size(ret,2))
        P = ShuffledFFT{ComplexF64}(size(ret),2)
        for k = 1:size(ret,1)
            @test (P * ret)[k,:] ≈ p * ret[k,:]
        end
    end
    
    @testset "Evaluation" begin
        F = Laurent()

        @test F == F

        @test axes(F,2) isa BlockedOneTo{Int,InfiniteArrays.InfStepRange{Int,Int}}
        @test axes(F) isa Tuple{<:Inclusion,BlockedOneTo{Int,InfiniteArrays.InfStepRange{Int,Int}}}
        @test axes(F,2)[Block(2)] == 2:3
        @test F[0.1,1] ≡ complex(1.0)
        @test F[0.1,2] == exp(-im*0.1)
        @test F[0.1,1:4] == [1,exp(-im*0.1),exp(im*0.1),exp(-2im*0.1)]
        @test F[0.1,Block(4)] == [exp(-im*3*0.1),exp(im*3*0.1)]
        @test F[0.1,Block.(1:3)] == [1,exp(-im*0.1),exp(im*0.1),exp(-2im*0.1),exp(2im*0.1)]

        u = F * BlockedVector([[1,2,3]; zeros(∞)], (axes(F,2),));
        @test u[0.1] == 1 + 2exp(-im*0.1) + 3exp(im*0.1)
    end

    @testset "Transform" begin
        F = Laurent()
        θ = axes(F,1)
        @test F[:,Base.OneTo(5)] \ cos.(θ) ≈ [0,0.5,0.5,0,0]
        @test F[:,Block.(Base.OneTo(5))] \ cos.(θ) ≈ [0,0.5,0.5,0,0,0,0,0,0]

        @test (F \ cos.(θ))[Block(2)] ≈ [0.5,0.5]
        u = F * (F \ exp.(cos.(θ)))
        @test u[0.1] ≈ exp(cos(0.1))
        @test F[:,Base.OneTo(5)] \ [cos.(θ) sin.(θ)] ≈ [0 0; 0.5 0.5im; 0.5 -0.5im; 0 0; 0 0]
        @test F[:,Block.(Base.OneTo(3))] \ [cos.(θ) sin.(θ)] ≈ [0 0; 0.5 0.5im; 0.5 -0.5im; 0 0; 0 0]
        U = F / F \ [exp.(cos.(θ)) cos.(cos.(θ))]
        @test U[0.1,:] ≈ [exp(cos(0.1)),cos(cos(0.1))]
        @test U[[0.1,0.2],:] ≈ [exp(cos(0.1)) cos(cos(0.1)); exp(cos(0.2)) cos(cos(0.2))]
        @test U[0.1,1] ≈ exp(cos(0.1))
        @test U[[0.1,0.2],1] ≈ exp.(cos.([0.1,0.2]))
    end

    @testset "Derivative" begin
        F = Laurent()
        @test F\F === Eye{ComplexF64}((axes(F,2),))
        @test (F'F)[1:10,1:10] == Diagonal(Fill(2π,10))
        @test blockisequal(axes(F'F), (axes(F,2),axes(F,2)))
        D = Derivative(axes(F,1))
        D̃ = (D*F).args[2]
        @test (F\F)*D̃ isa Diagonal
        @test (F \ (D*F))[Block.(1:3),Block.(1:3)] == Diagonal([0,-im,im,-2im,2im])

        u = F * BlockedVector([[1,2,3,4,5]; zeros(∞)], (axes(F,2),));
        @test blockisequal(axes(D̃,2),axes(u.args[2],1))
        @test (D*u)[0.1] ≈ -2im*exp(-im*0.1) + 3im*exp(im*0.1) - 8im*exp(-im*2*0.1) + 10im*exp(im*2*0.1)
    end
end
