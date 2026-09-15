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

alternatingsum(c::AbstractVector) = mapreduce(k -> isodd(k) ? c[k] : -c[k], +, eachindex(c); init=zero(eltype(c)))

_fromcanonical(ax, x) = (first(ax) + last(ax) + (last(ax) - first(ax)) * x) / 2

function _chopcoeffs(c::AbstractVector, tol::Real=0)
    n = findlast(x -> abs(x) > tol, c)
    n === nothing ? eltype(c)[] : collect(view(c, 1:n))
end

function _chebyshevcoefficients(f)
    P, _ = arguments(f)
    T = chebyshevt(P)
    paddeddata(T \ f)
end

function _colleague_matrix(c::AbstractVector{T}) where T
    n = length(c) - 1
    A = zeros(T, n, n)
    for k in 1:n-1
        A[k+1,k] = one(T)/2
        A[k,k+1] = one(T)/2
    end
    for k in 1:n
        A[1,end-k+1] -= c[k] / (2c[end])
    end
    n > 1 && (A[n,n-1] = one(T))
    A
end

function _pruneroots(r, htol::Float64)
    rr = real(r[abs.(imag.(r)) .< htol])
    rr = sort(rr[abs.(rr) .<= 1 + htol])
    clamp.(rr, -1, 1)
end

function _rootsunit_coeffs(c::AbstractVector{T}, htol::Float64) where T<:Number
    RT = typeof(float(real(zero(T))))
    splitpoint = convert(RT, -0.004849834917525)
    nrmc = norm(c, 1)
    nrmc > 0 || return RT[]
    c = _chopcoeffs(c, eps(RT) * nrmc)
    n = length(c)

    if n == 0
        return RT[]
    elseif n == 1
        return iszero(c[1]) ? RT[zero(RT)] : RT[]
    elseif n == 2
        r1 = -c[1] / c[2]
        return (abs(imag(r1)) > htol || abs(real(r1)) > 1 + htol) ? RT[] : RT[clamp(real(r1), -1, 1)]
    elseif n <= 70
        return _pruneroots(eigvals(_colleague_matrix(c)), htol)
    end

    x = chebyshevpoints(RT, n, Val(1))
    v1 = clenshaw(c, @. (splitpoint - 1)/2 + (splitpoint + 1)/2 * x)
    v2 = clenshaw(c, @. (splitpoint + 1)/2 + (1 - splitpoint)/2 * x)
    p1 = plan_chebyshevtransform(v1)
    p2 = plan_chebyshevtransform(v2)
    r1 = _rootsunit_coeffs(p1 * v1, 2htol)
    r2 = _rootsunit_coeffs(p2 * v2, 2htol)
    [@. (splitpoint - 1)/2 + (splitpoint + 1)/2 * r1;
     @. (splitpoint + 1)/2 + (1 - splitpoint)/2 * r2]
end

function findall_layout(::ExpansionLayout{<:AbstractOPLayout}, ::typeof(iszero), f)
    c = _chebyshevcoefficients(f)
    ax = axes(f, 1)
    (isempty(c) || all(iszero, c)) && return eltype(ax)[]
    hscale = max(abs(first(ax)), abs(last(ax)))
    htol = eps(2000.0) * max(hscale, 1)
    vscale = max(maximum(abs, clenshaw(c, chebyshevpoints(Float64, max(length(c), 2), Val(1)))), eps(Float64))
    cvscale = c ./ vscale
    r = _rootsunit_coeffs(cvscale, htol)

    if (isempty(r) || !isapprox(last(r), 1)) && abs(sum(cvscale)) < htol
        push!(r, 1.0)
    end
    if (isempty(r) || !isapprox(first(r), -1)) && abs(alternatingsum(cvscale)) < htol
        insert!(r, 1, -1.0)
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