## Root finding for Chebyshev expansions
#
# Contains code that is based in part on Chebfun v5's chebfun/@chebteck/roots.m,
# which is distributed with the following license:
#
# Copyright (c) 2015, The Chancellor, Masters and Scholars of the University
# of Oxford, and the Chebfun Developers. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Oxford nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# If f(x) = P[x,1:n+1] * c
# normalized so that c[n+1] == 1 so that
# P[x,n+1] == -P[x,1:n]'*c[1:n]
# then with J = jacobimatrix(P)' we have
# x*P[x,1:n] == J[1:n,1:n+1] * P[x,1:n+1]
#    == J[1:n,1:n] * P[x,1:n] + [zeros(n-1); J[n,n+1]*P[x,n+1]]
#    == J[1:n,1:n] * P[x,1:n] + [zeros(n-1); -J[n,n+1]*c[1:n]']*P[x,1:n]
# I.e. (J[1:n,1:n] + [zeros(n-1,n); -J[n,n+1]*c[1:n]'])*P[x,1:n] = x*P[x,1:n]
#

function colleaguematrix(P, c)
    cₙ = paddeddata(c)
    isempty(cₙ) && return Matrix{eltype(P)}(undef, 0, 0)
    n = findlast(!iszero, cₙ)-1
    J = jacobimatrix(P)'
    C = Matrix(J[1:n,1:n])
    C[end,:] .-= J[n,n+1] .* view(cₙ,1:n) ./ cₙ[n+1]
    C
end

_fromcanonical(ax, x) = (first(ax) + last(ax) + (last(ax) - first(ax)) * x) / 2

function _pruneroots(r, htol)
    rr = real(r[abs.(imag.(r)) .< htol])
    rr = sort(rr[abs.(rr) .<= 1 + htol])
    clamp.(rr, -1, 1)
end

# Chebfun constants: expansions longer than this are split at a point
# slightly left of 0 (to avoid splitting at a root of a symmetric function)
const _ROOTS_MAXDEGREE = 70
const _ROOTS_SPLITPOINT = -0.004849834917525

# roots in [-1,1] of the Chebyshev expansion with coefficients c
function _rootsunit_coeffs(c::AbstractVector{T}, htol) where T<:Number
    RT = real(float(T))
    nrmc = norm(c, 1)
    nrmc > 0 || return RT[]
    c = c[1:findlast(x -> abs(x) > eps(RT) * nrmc, c)]
    n = length(c)
    n ≤ 1 && return RT[]
    n ≤ _ROOTS_MAXDEGREE && return _pruneroots(eigvals(colleaguematrix(ChebyshevT{RT}(), c)), htol)

    s = convert(RT, _ROOTS_SPLITPOINT)
    left(x) = _fromcanonical((-one(RT), s), x)
    right(x) = _fromcanonical((s, one(RT)), x)
    x = chebyshevpoints(RT, n, Val(1))
    r1 = _rootsunit_coeffs(chebyshevtransform(clenshaw(c, left.(x))), 2htol)
    r2 = _rootsunit_coeffs(chebyshevtransform(clenshaw(c, right.(x))), 2htol)
    [left.(r1); right.(r2)]
end

function findall_layout(::ExpansionLayout{<:AbstractOPLayout}, ::typeof(iszero), f)
    ax = axes(f, 1)
    c = paddeddata(chebyshevt(ax) \ f)
    all(iszero, c) && return eltype(ax)[]
    RT = real(float(eltype(c)))
    tol = eps(RT(2000))
    vscale = max(maximum(abs, clenshaw(c, chebyshevpoints(RT, max(length(c), 2), Val(1)))), eps(RT))
    r = _rootsunit_coeffs(c, tol)

    if (isempty(r) || !isapprox(last(r), 1)) && abs(clenshaw(c, one(RT))) < tol * vscale
        push!(r, one(RT))
    end
    if (isempty(r) || !isapprox(first(r), -1)) && abs(clenshaw(c, -one(RT))) < tol * vscale
        pushfirst!(r, -one(RT))
    end

    map(x -> convert(eltype(ax), _fromcanonical(ax, x)), r)
end

####
# min/max/extrema
####
function minimum_layout(::ExpansionLayout{<:AbstractOPLayout}, f::AbstractQuasiVector, dims)
    r = findall(iszero, diff(f))
    if isempty(r)
        min(first(f), last(f))
    else
        min(first(f), minimum(f[r]), last(f))
    end
end

function maximum_layout(::ExpansionLayout{<:AbstractOPLayout}, f::AbstractQuasiVector, dims)
    r = findall(iszero, diff(f))
    if isempty(r)
        max(first(f), last(f))
    else
        max(first(f), maximum(f[r]), last(f))
    end
end

function extrema_layout(::ExpansionLayout{<:AbstractOPLayout}, f::AbstractQuasiVector, dims...)
    r = findall(iszero, diff(f))
    if isempty(r)
        extrema([first(f); last(f)])
    else    
        extrema([first(f); f[r]; last(f)])
    end
end