abstract type AbstractFourier{T} <: Basis{T} end

struct Fourier{T} <: AbstractFourier{T} end
struct Laurent{T} <: AbstractFourier{T} end

Fourier() = Fourier{Float64}()
Laurent() = Laurent{ComplexF64}()

==(::Fourier, ::Fourier) = true
==(::Laurent, ::Laurent) = true

axes(F::AbstractFourier) = (Inclusion(ℝ), _BlockedUnitRange(1:2:∞))

function getindex(F::Fourier{T}, x::Real, j::Int)::T where T
    isodd(j) && return cos((j÷2)*x)
    sin((j÷2)*x)
end

function getindex(F::Laurent{T}, x::Real, j::Int)::T where T
    s = 1-2iseven(j)
    exp(im*s*(j÷2)*x)
end

### transform
checkpoints(F::AbstractFourier) = eltype(axes(F,1))[1.223972,3.14,5.83273484]

fouriergrid(T, n) = convert(T,π)*collect(0:2:2n-2)/n

function grid(Pn::SubQuasiArray{T,2,<:AbstractFourier,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) where T
    kr,jr = parentindices(Pn)
    n = maximum(jr)
    fouriergrid(eltype(axes(Pn,1)), n)
end


abstract type AbstractShuffledPlan{T} <: Plan{T} end

"""
Gives a shuffled version of the real FFT, with order
1,sin(θ),cos(θ),sin(2θ)…
"""
struct ShuffledR2HC{T,Pl<:Plan} <: AbstractShuffledPlan{T}
    plan::Pl
end

"""
Gives a shuffled version of the FFT, with order
1,sin(θ),cos(θ),sin(2θ)…
"""
struct ShuffledFFT{T,Pl<:Plan} <: AbstractShuffledPlan{T}
    plan::Pl
end

size(F::AbstractShuffledPlan, k) = size(F.plan,k)
size(F::AbstractShuffledPlan) = size(F.plan)

ShuffledR2HC{T}(p::Pl) where {T,Pl<:Plan} = ShuffledR2HC{T,Pl}(p)
ShuffledR2HC{T}(n, d...) where T = ShuffledR2HC{T}(FFTW.plan_r2r(Array{T}(undef, n), FFTW.R2HC, d...))

ShuffledFFT{T}(p::Pl) where {T,Pl<:Plan} = ShuffledFFT{T,Pl}(p)
ShuffledFFT{T}(n, d...) where T = ShuffledFFT{T}(FFTW.plan_fft(Array{T}(undef, n), d...))


function _shuffledR2HC_postscale!(_, ret::AbstractVector{T}) where T
    n = length(ret)
    lmul!(convert(T,2)/n, ret)
    ret[1] /= 2
    iseven(n) && (ret[n÷2+1] /= 2)
    negateeven!(reverseeven!(interlace!(ret,1)))
end

function _shuffledR2HC_postscale!(d::Number, ret::AbstractMatrix{T}) where T
    if isone(d)
        n = size(ret,1)
        lmul!(convert(T,2)/n, ret)
        ldiv!(2, view(ret,1,:))
        iseven(n) && ldiv!(2, view(ret,n÷2+1,:))
        for j in axes(ret,2)
            negateeven!(reverseeven!(interlace!(view(ret,:,j),1)))
        end
    else
        n = size(ret,2)
        lmul!(convert(T,2)/n, ret)
        ldiv!(2, view(ret,:,1))
        iseven(n) && ldiv!(2, view(ret,:,n÷2+1))
        for k in axes(ret,1)
            negateeven!(reverseeven!(interlace!(view(ret,k,:),1)))
        end
    end
    ret
end

function _shuffledFFT_postscale!(_, ret::AbstractVector{T}) where T
    n = length(ret)
    cfs = lmul!(inv(convert(T,n)), ret)
    reverseeven!(interlace!(cfs,1))
end

function _shuffledFFT_postscale!(d::Number, ret::AbstractMatrix{T}) where T
    if isone(d)
        n = size(ret,1)
        lmul!(inv(convert(T,n)), ret)
        for j in axes(ret,2)
            reverseeven!(interlace!(view(ret,:,j),1))
        end
    else
        n = size(ret,2)
        lmul!(inv(convert(T,n)), ret)
        for k in axes(ret,1)
            reverseeven!(interlace!(view(ret,k,:),1))
        end
    end
    ret
end

function mul!(ret::AbstractArray{T}, F::ShuffledR2HC{T}, b::AbstractArray) where T
    mul!(ret, F.plan, convert(Array{T}, b))
    _shuffledR2HC_postscale!(F.plan.region, ret)
end

function mul!(ret::AbstractArray{T}, F::ShuffledFFT{T}, b::AbstractArray) where T
    mul!(ret, F.plan, convert(Array{T}, b))
    _shuffledFFT_postscale!(F.plan.region, ret)
end

*(F::AbstractShuffledPlan{T}, b::AbstractVecOrMat) where T = mul!(similar(b, T), F, b)

factorize(L::SubQuasiArray{T,2,<:Fourier,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(L), ShuffledR2HC{T}(size(L,2)))
factorize(L::SubQuasiArray{T,2,<:Fourier,<:Tuple{<:Inclusion,<:OneTo}}, d) where T =
    TransformFactorization(grid(L), ShuffledR2HC{T}((size(L,2),d),1))

factorize(L::SubQuasiArray{T,2,<:Laurent,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(L), ShuffledFFT{T}(size(L,2)))
factorize(L::SubQuasiArray{T,2,<:Laurent,<:Tuple{<:Inclusion,<:OneTo}}, d) where T =
    TransformFactorization(grid(L), ShuffledFFT{T}((size(L,2),d),1))


factorize(L::SubQuasiArray{T,2,<:AbstractFourier,<:Tuple{<:Inclusion,<:BlockSlice}},d...) where T =
    ProjectionFactorization(factorize(parent(L)[:,OneTo(size(L,2))],d...),parentindices(L)[2])

import BlockBandedMatrices: _BlockSkylineMatrix

@simplify function *(A::QuasiAdjoint{<:Any,<:Fourier}, B::Fourier)
    TV = promote_type(eltype(A),eltype(B))
    PseudoBlockArray(Diagonal(Vcat(2convert(TV,π),Fill(convert(TV,π),∞))), (axes(A,1),axes(B,2)))
end

@simplify function *(A::QuasiAdjoint{<:Any,<:Laurent}, B::Laurent)
    TV = promote_type(eltype(A),eltype(B))
    Diagonal(Fill(2convert(TV,π),(axes(B,2),)))
end

@simplify function *(D::Derivative, F::Fourier)
    TV = promote_type(eltype(D),eltype(F))
    Fourier{TV}()*_BlockArray(Diagonal(Vcat([reshape([0.0],1,1)], (1.0:∞) .* Fill([0 -one(TV); one(TV) 0], ∞))), (axes(F,2),axes(F,2)))
end

@simplify function *(D::Derivative, F::Laurent)
    TV = promote_type(eltype(D),eltype(F))
    Laurent{TV}() * Diagonal(PseudoBlockVector((((1:∞) .÷ 2) .* (1 .- 2 .* iseven.(1:∞))) * convert(TV,im), (axes(F,2),)))
end


function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), c::BroadcastQuasiVector{<:Any,typeof(cos),<:Tuple{<:Inclusion{<:Any,RealNumbers}}}, F::Fourier)
    axes(c,1) == axes(F,1) || throw(DimensionMismatch())
    T = promote_type(eltype(c), eltype(F))
    # Use LinearAlgebra.Tridiagonal for now since broadcasting support not complete for LazyBandedMatrices.Tridiagonal
    F*mortar(LinearAlgebra.Tridiagonal(Vcat([reshape([0; one(T)],2,1)], Fill(Matrix(one(T)/2*I,2,2),∞)),
                        Vcat([zeros(T,1,1)], Fill(Matrix(zero(T)I,2,2),∞)),
                        Vcat([[0 one(T)/2]], Fill(Matrix(one(T)/2*I,2,2),∞))))
end


function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), s::BroadcastQuasiVector{<:Any,typeof(sin),<:Tuple{<:Inclusion{<:Any,RealNumbers}}}, F::Fourier)
    axes(s,1) == axes(F,1) || throw(DimensionMismatch())
    T = promote_type(eltype(s), eltype(F))
    # Use LinearAlgebra.Tridiagonal for now since broadcasting support not complete for LazyBandedMatrices.Tridiagonal
    F*mortar(LinearAlgebra.Tridiagonal(Vcat([reshape([one(T); 0],2,1)], Fill([0 one(T)/2; -one(T)/2 0],∞)),
                        Vcat([zeros(T,1,1)], Fill(Matrix(zero(T)*I,2,2),∞)),
                        Vcat([[one(T)/2 0]], Fill([0 -one(T)/2; one(T)/2 0],∞))))
end


# support cos.(θ) .* F
Base.broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), c::Broadcasted{<:Any,<:Any,typeof(cos),<:Tuple{<:Inclusion{<:Any,RealNumbers}}}, F::Fourier) = materialize(c) .* F
Base.broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), c::Broadcasted{<:Any,<:Any,typeof(sin),<:Tuple{<:Inclusion{<:Any,RealNumbers}}}, F::Fourier) = materialize(c) .* F