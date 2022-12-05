
struct Fourier{T} <: Basis{T} end
Fourier() = Fourier{Float64}()

==(::Fourier, ::Fourier) = true

axes(F::Fourier) = (Inclusion(ℝ), _BlockedUnitRange(1:2:∞))

function getindex(F::Fourier{T}, x::Real, j::Int)::T where T
    isodd(j) && return cos((j÷2)*x)
    sin((j÷2)*x)
end

### transform
checkpoints(F::Fourier) = eltype(axes(F,1))[1.223972,3.14,5.83273484]

fouriergrid(T, n) = convert(T,π)*collect(0:2:2n-2)/n

function grid(Pn::SubQuasiArray{T,2,<:Fourier,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) where T
    kr,jr = parentindices(Pn)
    n = maximum(jr)
    fouriergrid(T, n)
end

"""
Gives a shuffled version of the real FFT, with order
1,sin(θ),cos(θ),sin(2θ)…
"""
struct ShuffledRFFT{T,Pl<:Plan} <: Factorization{T}
    plan::Pl
end

size(F::ShuffledRFFT, k) = size(F.plan,k)
size(F::ShuffledRFFT) = size(F.plan)

ShuffledRFFT{T}(p::Pl) where {T,Pl<:Plan} = ShuffledRFFT{T,Pl}(p)
ShuffledRFFT{T}(n, d...) where T = ShuffledRFFT{T}(FFTW.plan_r2r(Array{T}(undef, n), FFTW.R2HC, d...))

function _shuffledrfft_postscale!(_, ret::AbstractVector{T}) where T
    n = length(ret)
    lmul!(convert(T,2)/n, ret)
    ret[1] /= 2
    iseven(n) && (ret[n÷2+1] /= 2)
    negateeven!(reverseeven!(interlace!(ret,1)))
end

function _shuffledrfft_postscale!(d::Number, ret::AbstractMatrix{T}) where T
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


function mul!(ret::AbstractArray{T}, F::ShuffledRFFT{T}, b::AbstractArray) where T
    mul!(ret, F.plan, convert(Array{T}, b))
    _shuffledrfft_postscale!(F.plan.region, ret)
end

*(F::ShuffledRFFT{T}, b::AbstractVecOrMat) where T = mul!(similar(b, T), F, b)

factorize(L::SubQuasiArray{T,2,<:Fourier,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(L), ShuffledRFFT{T}(size(L,2)))
factorize(L::SubQuasiArray{T,2,<:Fourier,<:Tuple{<:Inclusion,<:OneTo}}, d) where T =
    TransformFactorization(grid(L), ShuffledRFFT{T}((size(L,2),d),1))

factorize(L::SubQuasiArray{T,2,<:Fourier,<:Tuple{<:Inclusion,<:BlockSlice}},d...) where T =
    ProjectionFactorization(factorize(parent(L)[:,OneTo(size(L,2))],d...),parentindices(L)[2])

import BlockBandedMatrices: _BlockSkylineMatrix

@simplify function *(A::QuasiAdjoint{<:Any,<:Fourier}, B::Fourier)
    TV = promote_type(eltype(A),eltype(B))
    PseudoBlockArray(Diagonal(Vcat(2convert(TV,π),Fill(convert(TV,π),∞))), (axes(A,1),axes(B,2)))
end

@simplify function *(D::Derivative, F::Fourier)
    TV = promote_type(eltype(D),eltype(F))
    Fourier{TV}()*_BlockArray(Diagonal(Vcat([reshape([0.0],1,1)], (1.0:∞) .* Fill([0 -one(TV); one(TV) 0], ∞))), (axes(F,2),axes(F,2)))
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