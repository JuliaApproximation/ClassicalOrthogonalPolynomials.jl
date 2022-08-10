
transform_ldiv(A::AbstractQuasiArray{T}, f::AbstractQuasiArray{V}, ::Tuple{<:Any,InfiniteCardinal{0}}) where {T,V}  =
    adaptivetransform_ldiv(A, f)

pad(c::AbstractVector{T}, ax::Union{OneTo,OneToInf}) where T = [c; Zeros(length(ax)-length(c))]
pad(c, ax...) = PaddedArray(c, ax)

padchop!(cfs, tol, ax...) = pad(chop!(cfs, tol), ax...)
padchop(cfs, tol, ax...) = pad(chop(cfs, tol), ax...)

# ax will impose block structure for us
padchop!(cfs::PseudoBlockVector, tol, ax...) = padchop!(cfs.blocks, tol, ax...)


padresize!(cfs, m, ax...) = pad(compatible_resize!(cfs, m), ax...)
padresize!(cfs::PseudoBlockVector, m, ax...) = padresize!(cfs.blocks, m, ax...)


increasingtruncations(::OneToInf) = oneto.(2 .^ (4:∞))
increasingtruncations(::BlockedUnitRange) = broadcast(n -> Block.(oneto(n)), (2 .^ (4:∞)))


function adaptivetransform_ldiv(A::AbstractQuasiArray{U}, f::AbstractQuasiVector{V}) where {U,V}
    T = promote_type(eltype(U),eltype(V))

    r = checkpoints(A)
    fr = f[r]
    maxabsfr = norm(fr,Inf)

    tol = 20eps(real(typeof(first(fr))))
    ax = axes(A,2)

    for jr in increasingtruncations(ax)
        An = A[:,jr]
        cfs = An \ f
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

    bx = axes(f,2)
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
            return pad(similar(cfs,0,size(cfs,2)), ax, bx)
        end

        if maximum(abs,@views(cfs[end-2:end,:])) < 10tol*maxabsc
            n = size(cfs,1)
            c = padchop(cfs, tol, ax, bx)
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