struct HashedLocator
    alignment
    uv_bounds
    lower
    step
    uv_hash
    function HashedLocator(surf,uv_bounds,lower,upper,step=1,t⁰=0.,T=Float64,mem=Array)
        # surf(uv*,t) is closest to x when alignment(uv*,t,x)=0
        dsurf(uv,t) = ForwardDiff.derivative(uv->surf(uv,t),uv)
        alignment(uv,t,x) = (x-surf(uv,t))'*dsurf(uv,t)
    
        # Allocate hash table and struct
        uv_hash = zeros(T,ceil.(Int,(upper-lower)/step .+ 1)...) |> mem
        l = new(alignment,uv_bounds,lower,step,uv_hash)
    
        # Fill uv_hash
        @fastmath @inbounds @simd for I in CartesianIndices(l.uv_hash)
            # Map hash index to physical space
            x = step*(SA[I.I...] .- 1)+lower
    
            # Grid search for uv within bounds
            uv = argmin(range(uv_bounds...,20)) do uv
                sum(abs2,x-surf(uv,t⁰))
            end
    
            # Refine with NewtonStep
            l.uv_hash[I] = NewtonStep!(uv,l,x,t⁰)
        end
        return l
    end
end

function NewtonStep!(uv,l::HashedLocator,x,t) 
    step = l.alignment(uv,t,x)/ForwardDiff.derivative(uv->l.alignment(uv,t,x),uv)
    isnan(step) && return uv
    uv = clamp(uv-step,l.uv_bounds...)
end

function (l::HashedLocator)(x::SVector,t)
    # Map location to hash index and clamp to within domain
    hash_index = (x-l.lower)/l.step .+ 1
    clamped = clamp.(hash_index,1.5,size(l.uv_hash) .- 0.5)

    # Interpolate hash and return if index is outside domain
    uv = interp(clamped,l.uv_hash)
    hash_index != clamped && return uv

    # Otherwise, refine uv estimate with NewtonStep
    return NewtonStep!(uv,l::HashedLocator,x::SVector,t)
end

function interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
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