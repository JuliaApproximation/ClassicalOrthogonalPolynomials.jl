
struct Fourier{T} <: Basis{T} end
Fourier() = Fourier{Float64}()

==(::Fourier, ::Fourier) = true

axes(F::Fourier) = (Inclusion(ℝ), _BlockedUnitRange(1:2:∞))

function getindex(F::Fourier{T}, x::Real, j::Int)::T where T
    isodd(j) && return cos((j÷2)*x)
    sin((j÷2)*x)
end

### transform
checkpoints(::Fourier{T}) where T = T[1.223972,3.14,5.83273484]

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
struct ShuffledRFFT{T,Plan} <: Factorization{T}
    plan::Plan
end

size(F::ShuffledRFFT, _) = size(F.plan,1)
size(F::ShuffledRFFT) = (size(F.plan,1),size(F.plan,1))

ShuffledRFFT{T}(p::Plan) where {T,Plan} = ShuffledRFFT{T,Plan}(p)
ShuffledRFFT{T}(n::Int) where T = ShuffledRFFT{T}(FFTW.plan_r2r!(Array{T}(undef, n), FFTW.R2HC))


function *(F::ShuffledRFFT{T}, b::AbstractVector) where T
    n = size(F,1)
    c = lmul!(convert(T,2)/n, F.plan * convert(Array, b))
    c[1] /= 2
    iseven(n) && (c[n÷2+1] /= 2)
    negateeven!(reverseeven!(interlace!(c,1)))
end

factorize(L::SubQuasiArray{T,2,<:Fourier,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(L), ShuffledRFFT{T}(size(L,2)))

import BlockBandedMatrices: _BlockSkylineMatrix

@simplify function *(A::QuasiAdjoint{<:Any,<:Fourier}, B::Fourier)
    TV = promote_type(eltype(A),eltype(B))
    PseudoBlockArray(Diagonal(Vcat(2convert(TV,π),Fill(convert(TV,π),∞))), (axes(A,1),axes(B,2)))
end

@simplify function *(D::Derivative, F::Fourier)
    TV = promote_type(eltype(D),eltype(F))
    Fourier{TV}()*_BlockArray(Diagonal(Vcat([reshape([0.0],1,1)], (1.0:∞) .* Fill([0 -one(TV); one(TV) 0], ∞))), (axes(F,2),axes(F,2)))
end


function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), c::BroadcastQuasiVector{<:Any,typeof(cos),<:Tuple{<:Inclusion{<:Any,<:FullSpace}}}, F::Fourier)
    axes(c,1) == axes(F,1) || throw(DimensionMismatch())
    T = promote_type(eltype(c), eltype(F))
    F*mortar(Tridiagonal(Vcat([reshape([0; one(T)],2,1)], Fill(Matrix(one(T)/2*I,2,2),∞)),
                        Vcat([zeros(T,1,1)], Fill(Matrix(zero(T)I,2,2),∞)),
                        Vcat([[0 one(T)/2]], Fill(Matrix(one(T)/2*I,2,2),∞))))
end


function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), s::BroadcastQuasiVector{<:Any,typeof(sin),<:Tuple{<:Inclusion{<:Any,<:FullSpace}}}, F::Fourier)
    axes(s,1) == axes(F,1) || throw(DimensionMismatch())
    T = promote_type(eltype(s), eltype(F))
    F*mortar(Tridiagonal(Vcat([reshape([one(T); 0],2,1)], Fill([0 one(T)/2; -one(T)/2 0],∞)),
                        Vcat([zeros(T,1,1)], Fill(Matrix(zero(T)*I,2,2),∞)),
                        Vcat([[one(T)/2 0]], Fill([0 -one(T)/2; one(T)/2 0],∞))))
end
