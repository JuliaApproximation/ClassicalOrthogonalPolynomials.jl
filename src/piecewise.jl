struct PiecewisePolynomial{T,Bas,P<:AbstractVector} <: Basis{T}
    basis::Bas
    points::P
end

struct ContinuousPolynomial{order,T,P<:AbstractVector} <: Basis{T}
    points::P
end


ContinuousPolynomial{o,T}(pts::P) where {o,T,P} = ContinuousPolynomial{o,T,P}(pts)
ContinuousPolynomial{o}(pts) where o = ContinuousPolynomial{o,Float64}(pts)

axes(B::PiecewisePolynomial) =
    (Inclusion(first(B.points)..last(B.points)), blockedrange(Fill(length(B.points)-1, ∞)))

axes(B::ContinuousPolynomial{1}) where o =
    (Inclusion(first(B.points)..last(B.points)), blockedrange(Vcat(length(B.points), Fill(length(B.points)-1, ∞)))) 

function getindex(P::PiecewisePolynomial{T}, x::Number, Kk::BlockIndex{1}) where T
    K,k = block(Kk),blockindex(Kk)
    b = searchsortedlast(P.points,x)
    if b == length(P.points)
        P.basis[affine(P.points[b-1]..P.points[b],axes(P.basis,1))[x],Int(K)-1]
    elseif b == k
        P.basis[affine(P.points[b]..P.points[b+1],axes(P.basis,1))[x],Int(K)-1]
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
        if b == length(P.points)
            Weighted(Jacobi{T}(1,1))[affine(P.points[b-1]..P.points[b],ChebyshevInterval{real(T)}())[x],Int(K)-1]
        elseif b == k
            Weighted(Jacobi{T}(1,1))[affine(P.points[b]..P.points[b+1],ChebyshevInterval{real(T)}())[x],Int(K)-1]
        else
            zero(T)
        end
    end
end

function getindex(P::ContinuousPolynomial{1,T}, x::Number, Kk::BlockIndex{1}) where T
    K,k = block(Kk),blockindex(Kk)
    if K == Block(1)
        LinearSpline(P.points)[x,k]
    else
        b = searchsortedlast(P.points,x)
        if b == length(P.points)
            Weighted(jacobi(1,1,P.points[b-1]..P.points[b]))[x,Int(K)-1]
        elseif b == k
            Weighted(jacobi(1,1,P.points[b]..P.points[b+1]))[x,Int(K)-1]
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