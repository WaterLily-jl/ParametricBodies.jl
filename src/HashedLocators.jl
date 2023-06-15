struct HashedLocator{T,F<:Function,A<:AbstractArray{T,2}}
    refine::F
    lims::NTuple{2,T}
    hash::A
    lower::SVector{2,T}
    step::T
end
Adapt.adapt_structure(to, x::HashedLocator) = HashedLocator(x.refine,x.lims,adapt(to,x.hash),x.lower,x.step)

function HashedLocator(curve,lims;t⁰=0,step=1,T=Float64,mem=Array)
    # Apply type and get refinement function
    lims,t⁰,step = T.(lims),T(t⁰),T(step)
    f = refine(curve,lims)

    # Get curve's bounding box
    samples = range(lims...,20)
    lower = upper = curve(first(samples),t⁰)
    @assert isa(lower,SVector{2,T}) "`curve` is not type stable"
    for uv in samples
        x = curve(uv,t⁰)
        lower = min.(lower,x)
        upper = max.(upper,x)
    end

    # Allocate hash and struct, and update hash
    hash = fill(first(lims),ceil.(Int,(upper-lower)/step .+ 3)...) |> mem
    l=adapt(mem,HashedLocator{T,typeof(f),typeof(hash)}(f,lims,hash,lower.-step,step))
    update!(l,curve,t⁰,samples)
end

function refine(curve,lims)
    # uv⁺ = argmin_uv (X-curve(uv,t))² -> alignment(X,uv⁺,t))=0
    dcurve(uv,t) = ForwardDiff.derivative(uv->curve(uv,t),uv)
    align(X,uv,t) = (X-curve(uv,t))'*dcurve(uv,t)
    dalign(X,uv,t) = ForwardDiff.derivative(uv->align(X,uv,t),uv)
    return function(X,uv,t) # Newton step to alignment root
        step=align(X,uv,t)/dalign(X,uv,t)
        ifelse(isnan(step),uv,clamp(uv-step,lims...))
    end
end
notC¹(l::HashedLocator,uv) = any(uv.≈l.lims)

update!(l::HashedLocator,surf,t,samples=l.lims)=(_update!(get_backend(l.hash),64)(l,surf,samples,t,ndrange=size(l.hash));l)
@kernel function _update!(l::HashedLocator{T},surf,@Const(samples),@Const(t)) where T
    # Map index to physical space
    I = @index(Global,Cartesian)
    x = l.step*(SVector{2,T}(I.I...) .-1)+l.lower

    # Grid search for uv within bounds
    @inline dis2(uv) = (q=x-surf(uv,t); q'*q)
    uv = l.hash[I]; d = dis2(uv)
    for uvᵢ in samples
        dᵢ = dis2(uvᵢ)
        dᵢ<d && (uv=uvᵢ; d=dᵢ)
    end
    
    # Refine estimate with clamped Newton step
    l.hash[I] = l.refine(x,uv,t)
end

function (l::HashedLocator)(x,t)
    # Map location to hash index and clamp to within domain
    hash_index = (x-l.lower)/l.step .+ 1
    clamped = clamp.(hash_index,1,size(l.hash) .- 0.5f0)

    # Interpolate hash and return if index is outside domain
    uv = interp(clamped,l.hash)
    hash_index != clamped && return uv

    # Otherwise, refine estimate with Newton step
    return l.refine(x,uv,t)
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