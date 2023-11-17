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
grid(Pn::AbstractFourier, n::Integer) = fouriergrid(eltype(axes(Pn,1)), n)

abstract type AbstractShuffledPlan{T} <: Plan{T} end

"""
Gives a shuffled version of the real FFT, with order
1,sin(θ),cos(θ),sin(2θ)…
"""
struct ShuffledR2HC{T,Pl<:Plan} <: AbstractShuffledPlan{T}
    plan::Pl
end

"""
Gives a shuffled version of the real IFFT, with order
1,sin(θ),cos(θ),sin(2θ)…
"""
struct ShuffledIR2HC{T,Pl<:Plan} <: AbstractShuffledPlan{T}
    plan::Pl
end

"""
Gives a shuffled version of the FFT, with order
1,sin(θ),cos(θ),sin(2θ)…
"""
struct ShuffledFFT{T,Pl<:Plan} <: AbstractShuffledPlan{T}
    plan::Pl
end

"""
Gives a shuffled version of the IFFT, with order
1,sin(θ),cos(θ),sin(2θ)…
"""
struct ShuffledIFFT{T,Pl<:Plan} <: AbstractShuffledPlan{T}
    plan::Pl
end

size(F::AbstractShuffledPlan, k) = size(F.plan,k)
size(F::AbstractShuffledPlan) = size(F.plan)

ShuffledR2HC{T}(p::Pl) where {T,Pl<:Plan} = ShuffledR2HC{T,Pl}(p)
ShuffledR2HC{T}(n, d...) where T = ShuffledR2HC{T}(FFTW.plan_r2r(Array{T}(undef, n), FFTW.R2HC, d...))

ShuffledFFT{T}(p::Pl) where {T,Pl<:Plan} = ShuffledFFT{T,Pl}(p)
ShuffledFFT{T}(n, d...) where T = ShuffledFFT{T}(FFTW.plan_fft(Array{T}(undef, n), d...))

ShuffledIFFT{T}(p::Pl) where {T,Pl<:Plan} = ShuffledIFFT{T,Pl}(p)
ShuffledIR2HC{T}(p::Pl) where {T,Pl<:Plan} = ShuffledIR2HC{T,Pl}(p)

inv(P::ShuffledR2HC{T}) where T = ShuffledIR2HC{T}(inv(P.plan))
inv(P::ShuffledFFT{T}) where T = ShuffledIFFT{T}(inv(P.plan))


_shuffled_prescale!(_, ret::AbstractVector) = ret
_shuffled_postscale!(_, ret::AbstractVector) = ret

function _shuffled_postscale!(::ShuffledR2HC, ret::AbstractVector{T}) where T
    n = length(ret)
    lmul!(convert(T,2)/n, ret)
    ret[1] /= 2
    iseven(n) && (ret[n÷2+1] /= 2)
    negateeven!(reverseeven!(interlace!(ret,1)))
end

function _shuffled_prescale!(::ShuffledIR2HC, ret::AbstractVector{T}) where T
    n = length(ret)
    reverseeven!(negateeven!(ret))
    ret .= [ret[1:2:end]; ret[2:2:end]] # todo: non-allocating
    iseven(n) && (ret[n÷2+1] *= 2)
    ret[1] *= 2
    lmul!(convert(T,n)/2, ret)
end

function _shuffled_postscale!(::ShuffledFFT, ret::AbstractVector{T}) where T
    n = length(ret)
    cfs = lmul!(inv(convert(T,n)), ret)
    reverseeven!(interlace!(cfs,1))
end

_region(F::Plan) = F.region
_region(F::ScaledPlan) = F.p.region

for func in (:_shuffled_postscale!, :_shuffled_prescale!)
    @eval function $func(F::AbstractShuffledPlan, ret::AbstractMatrix{T}) where T
        d = _region(F.plan)
        if isone(d)
            for j in axes(ret,2)
                $func(F, view(ret,:,j))
            end
        else
            @assert d == 2
            for k in axes(ret,1)
                $func(F, view(ret,k,:))
            end
        end
        ret
    end
end

function mul!(ret::AbstractArray{T}, F::AbstractShuffledPlan{T}, bin::AbstractArray) where T
    b = _shuffled_prescale!(F, Array{T}(bin))
    mul!(ret, F.plan, b)
    _shuffled_postscale!(F, ret)
end


*(F::AbstractShuffledPlan{T}, b::AbstractVecOrMat) where T = mul!(similar(b, T), F, b)

plan_transform(F::Fourier{T}, szs::NTuple{N,Int}, dims...) where {T,N} = ShuffledR2HC{T}(szs, dims...)
plan_transform(F::Laurent{T}, szs::NTuple{N,Int}, dims...) where {T,N} = ShuffledFFT{T}(szs, dims...)

import BlockBandedMatrices: _BlockSkylineMatrix

@simplify function *(A::QuasiAdjoint{<:Any,<:Fourier}, B::Fourier)
    TV = promote_type(eltype(A),eltype(B))
    PseudoBlockArray(Diagonal(Vcat(2convert(TV,π),Fill(convert(TV,π),∞))), (axes(A,1),axes(B,2)))
end

@simplify function *(A::QuasiAdjoint{<:Any,<:Laurent}, B::Laurent)
    TV = promote_type(eltype(A),eltype(B))
    Diagonal(Fill(2convert(TV,π),(axes(B,2),)))
end

function diff(F::Fourier{T}; dims=1) where T
    D = _BlockArray(Diagonal(Vcat([reshape([zero(T)],1,1)], (one(T):∞) .* Fill([0 -one(T); one(T) 0], ∞))), (axes(F,2),axes(F,2)))
    F * D
end

function diff(F::Laurent{T}; dims=1) where T
    D = Diagonal(PseudoBlockVector((((one(real(T)):∞) .÷ 2) .* (1 .- 2 .* iseven.(1:∞))) * convert(T,im), (axes(F,2),)))
    F * D
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