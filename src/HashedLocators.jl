struct HashedLocator{N,T,F<:Function,A<:AbstractArray{T,N}}
    alignment::F
    uv_bounds::NTuple{2,T}
    lower::SVector{N,T}
    step::T
    uv_hash::A
    function HashedLocator(surf,uv_bounds,lower,upper;step=1,t⁰=0.,T=Float64,mem=Array)
        # surf(uv*,t) is closest to x when alignment(uv*,t,x)=0
        dsurf(uv,t) = ForwardDiff.derivative(uv->surf(uv,t),uv)
        alignment(uv,t,x) = (x-surf(uv,t))'*dsurf(uv,t)
    
        # Allocate hash table and struct
        uv_hash = fill(T(uv_bounds[1]),ceil.(Int,(upper-lower)/step .+ 1)...) |> mem
        N,F,A = length(lower),typeof(alignment),typeof(uv_hash)
        l = new{N,T,F,A}(alignment,T.(uv_bounds),SVector{N,T}(lower),T(step),uv_hash)
    
        # Fill uv_hash
        update!(l,surf,t⁰,samples=20)
    end
end

using WaterLily
function update!(l::HashedLocator,surf,t;samples=2) 
    function update(I)
        # Map hash index to physical space
        x = l.step*(SA[I.I...] .- 1)+l.lower
    
        # Grid search for uv within bounds
        @inline dis2(uv) = (q=x-surf(uv,t); q'*q)
        uv = l.uv_hash[I]; d = dis2(uv)
        for uvᵢ in range(l.uv_bounds...,samples)
            dᵢ = dis2(uvᵢ)
            dᵢ<d && (uv=uvᵢ; d=dᵢ)
        end
    
        # Refine with NewtonStep
        NewtonStep!(uv,l,x,t)
    end
    WaterLily.@loop l.uv_hash[I] = update(I) over I ∈ CartesianIndices(l.uv_hash)
    return l
end

function NewtonStep!(uv,l::HashedLocator,x,t) 
    step = l.alignment(uv,t,x)/ForwardDiff.derivative(uv->l.alignment(uv,t,x),uv)
    isnan(step) && return uv
    uv = clamp(uv-step,l.uv_bounds...)
end

function (l::HashedLocator)(x,t)
    # Map location to hash index and clamp to within domain
    hash_index = (x-l.lower)/l.step .+ 1
    clamped = clamp.(hash_index,1,size(l.uv_hash) .- 0.5f0)

    # Interpolate hash and return if index is outside domain
    uv = interp(clamped,l.uv_hash)
    hash_index != clamped && return uv

    # Otherwise, refine uv estimate with NewtonStep
    return NewtonStep!(uv,l,x,t)
end

function interp(x::SVector{D}, arr::AbstractArray{T,D}) where {D,T}
    # Index below the interpolation coordinate and the difference
    i = floor.(Int,x); y = x-i

    # CartesianIndices around x 
    I = CartesianIndex(i...); R = I:I+oneunit(I)

    # Linearly weighted sum over arr[R] (in serial)
    s = zero(T)
    @fastmath @inbounds @simd for J in R
        weight = prod(@. ifelse(J.I==I.I,1-y,y))
        s += arr[J]*weight
    end
    return s
end