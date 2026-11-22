
ContinuumArrays.transform_ldiv_size(::Tuple{<:Any,InfiniteCardinal{0}}, A::AbstractQuasiArray{T}, f::AbstractQuasiArray{V}) where {T,V}  =
    adaptivetransform_ldiv(A, f)

# Right division by Q' becomes the adjoint of Q \ (P \ f)'.
function ContinuumArrays.basis_ldiv_size(::NTuple{2,InfiniteCardinal{0}}, Q::OrthogonalPolynomial,
        L::QuasiAdjoint{<:Any,<:ApplyQuasiMatrix{<:Any,typeof(\),<:Tuple{OrthogonalPolynomial,AbstractQuasiMatrix}}})
    P, f = arguments(parent(L))
    adaptivetransform_ldiv(P, f, Q)'
end

function adaptivetransform_ldiv(P::OrthogonalPolynomial, f::AbstractQuasiMatrix, Q::OrthogonalPolynomial)
    r, s = checkpoints(P), checkpoints(Q)
    samples = f[r,s]
    T = promote_type(eltype(P), eltype(Q), typeof(first(samples)))
    P, Q = convert(AbstractQuasiMatrix{T}, P), convert(AbstractQuasiMatrix{T}, Q)
    tol = 20eps(real(T))
    samplemax = norm(samples, Inf)
    m = n = 16

    while true
        x, Px = plan_grid_transform(P, (m,n), 1)
        y, Qy = plan_grid_transform(Q, (n,m), 1)
        gridvalues = Matrix{T}(f[x,y])
        scale = max(samplemax, norm(gridvalues, Inf))
        isfinite(scale) || throw(ArgumentError("non-finite samples in adaptive transform"))
        rowcoeffs = Px * gridvalues
        C = (Qy * Matrix(rowcoeffs'))'

        k = standardchoplength(vec(maximum(abs, C; dims=2)), tol)
        l = standardchoplength(vec(maximum(abs, C; dims=1)), tol)
        rowsresolved, colsresolved = k < m-1, l < n-1
        if rowsresolved && colsresolved
            block = C[1:k,1:l]
            values = P[r,1:k] * block * Q[s,1:l]'
            if norm(values - samples, Inf) ≤ max(m,n)*tol*scale
                return pad(block, axes(P,2), axes(Q,2))
            end
            # Decaying tails can also come from aliasing on the transform grid.
            m *= 2
            n *= 2
        else
            rowsresolved || (m *= 2)
            colsresolved || (n *= 2)
        end
    end
end


padchop!(cfs, tol, ax...) = pad(chop!(cfs, tol), ax...)
padchop(cfs, tol, ax...) = pad(chop(cfs, tol), ax...)

padrows(a::AbstractVector, ax) = pad(a, ax)
padrows(a::AbstractMatrix, ax) = pad(a, ax, :)

# ax will impose block structure for us
padchop!(cfs::BlockedVector, tol, ax...) = padchop!(cfs.blocks, tol, ax...)


padresize!(cfs, m, ax...) = pad(compatible_resize!(cfs, m), ax...)
padresize!(cfs::BlockedVector, m, ax...) = BlockedArray(padresize!(cfs.blocks, m, ax...), ax)


increasingtruncations(::OneToInf) = oneto.(2 .^ (4:∞))
increasingtruncations(::BlockedOneTo) = broadcast(n -> Block.(oneto(n)), (2 .^ (4:∞)))


function adaptivetransform_ldiv(A::AbstractQuasiArray{U}, f::AbstractQuasiVector{V}) where {U,V}
    r = checkpoints(A)
    fr = f[r]
    maxabsfr = norm(fr,Inf)

    # don't use V as eltype might be Any
    T = promote_type(eltype(U), eltype(typeof(first(fr))))
    tol = 20eps(real(T))
    ax = axes(A,2)

    for jr in increasingtruncations(ax)
        An = A[:,jr]
        cfs = transform_ldiv(An, f)
        maxabsc = maximum(abs, cfs)
        if maxabsc ≤ tol && maxabsfr ≤ tol # probably zero
            return pad(similar(cfs,0), ax)
        end

        m = length(cfs)
        m̃ = standardchoplength(cfs, tol)

        
        if m̃ < m-1 # coefficient tail is "zero" based on standard chop
            c = padresize!(cfs, m̃, ax)
            un = A * c
            if all(norm.(un[r] - fr, 1) .<  m*tol*1000*max(maxabsfr, 1))
                return c
            end
        end
    end
    error("Have not converged")
end


function adaptivetransform_ldiv(A::AbstractQuasiArray{U}, f::AbstractQuasiMatrix{V}) where {U,V}
    T = promote_type(eltype(U),eltype(V))

    r = checkpoints(A)
    fr = f[r,:]
    maxabsfr = norm(fr,Inf)

    tol = 20eps(real(T))
    ax = axes(A,2)

    for jr = increasingtruncations(ax)
        An = A[:,jr]
        cfs = An \ f
        maxabsc = maximum(abs, cfs)
        if maxabsc == 0 && maxabsfr == 0
            return pad(similar(cfs,0,size(cfs,2)), ax, :)
        end

        if maximum(abs,@views(cfs[end-2:end,:])) < 10tol*maxabsc
            n = size(cfs,1)
            c = padchop(cfs, tol, ax, :)
            un = A * c # expansion
        # we allow for transformed coefficients being a different size
        ##TODO: how to do scaling for unnormalized bases like Jacobi?
            if all(norm.(un[r,:] - fr, 1) .< tol * n * maxabsfr*1000)
                return c
            end
        end
    end
    error("Have not converged")
end
