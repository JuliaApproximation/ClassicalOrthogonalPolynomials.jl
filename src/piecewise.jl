struct PiecewisePolynomial{T,Bas,P<:AbstractVector} <: Basis{T}
    basis::Bas
    points::P
end

PiecewisePolynomial{T}(basis::AbstractQuasiMatrix, points::AbstractVector) where T =
    PiecewisePolynomial{T,typeof(basis),typeof(points)}(basis, points)
PiecewisePolynomial(basis::AbstractQuasiMatrix{T}, points::AbstractVector) where T =
    PiecewisePolynomial{T}(basis, points)

struct ContinuousPolynomial{order,T,P<:AbstractVector} <: Basis{T}
    points::P
end


ContinuousPolynomial{o,T}(pts::P) where {o,T,P} = ContinuousPolynomial{o,T,P}(pts)
ContinuousPolynomial{o}(pts) where o = ContinuousPolynomial{o,Float64}(pts)

axes(B::PiecewisePolynomial) =
    (Inclusion(first(B.points)..last(B.points)), blockedrange(Fill(length(B.points)-1, ∞)))

axes(B::ContinuousPolynomial{1}) where o =
    (Inclusion(first(B.points)..last(B.points)), blockedrange(Vcat(length(B.points), Fill(length(B.points)-1, ∞)))) 

==(::PiecewisePolynomial, ::ContinuousPolynomial) = false
==(::ContinuousPolynomial, ::PiecewisePolynomial) = false

function getindex(P::PiecewisePolynomial{T}, x::Number, Kk::BlockIndex{1}) where T
    K,k = block(Kk),blockindex(Kk)
    b = searchsortedlast(P.points,x)
    if b == length(P.points) == k + 1 # last point
        P.basis[affine(P.points[end-1]..P.points[end],axes(P.basis,1))[x],Int(K)]
    elseif b == k
        P.basis[affine(P.points[b]..P.points[b+1],axes(P.basis,1))[x],Int(K)]
    else
        zero(T)
    end
end    

function getindex(P::ContinuousPolynomial{1,T}, x::Number, Kk::BlockIndex{1}) where T
    K,k = block(Kk),blockindex(Kk)
    if K == Block(1)
        LinearSpline(P.points)[x,k]
    else
        b = searchsortedlast(P.points,x)
        if b == k
            Weighted(Jacobi{T}(1,1))[affine(P.points[b]..P.points[b+1],ChebyshevInterval{real(T)}())[x],Int(K)-1]
        else
            zero(T)
        end
    end
end


getindex(P::Union{PiecewisePolynomial,ContinuousPolynomial}, x::Number, k::Int) = P[x,findblockindex(axes(P,2),k)]


function grid(V::SubQuasiArray{T,2,<:PiecewisePolynomial,<:Tuple{Inclusion,BlockSlice}}) where T
    P = parent(V)
    _,JR = parentindices(Q)
    N = Int(last(JR))
    g = grid(P[:,OneTo(N)])
    pts = P.points
    ret = Matrix{T}(undef,length(g),length(pts)-1)
    for j in axes(ret,2)
        ret[:,j] = affine(axes(P.basis,1),pts[j]..pts[j+1])[g]
    end
    ret
end

function grid(V::SubQuasiArray{T,2,<:ContinuousPolynomial{1},<:Tuple{Inclusion,BlockSlice}}) where T
    P = parent(V)
    _,JR = parentindices(Q)
    pts = P.points
    grid(view(PiecewisePolynomial(Weighted(Jacobi{T}(1,1)), pts),:,JR))
end

#######
# Conversion
#######

function \(P::PiecewisePolynomial{T,<:Legendre}, C::ContinuousPolynomial{1,V}) where {T,V}
    @assert P.points == C.points

    v = mortar(Fill.((2:2:∞) ./ (3:2:∞), length(P.points)))

    # L = Legendre{T}() \ Weighted(Jacobi{T}(1,1))

end