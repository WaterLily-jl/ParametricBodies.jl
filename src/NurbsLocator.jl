"""
    NurbsLocator

    - `refine<:Function`` Performs bounded Newton root-finding step
    - `lims::NTuple{2,T}:` limits of the `uv` parameter
    - `hash<:AbstractArray{T,2}:` Hash to supply good IC to `refine`
    - `lower::SVector{2,T}:` bottom corner of the hash in ξ-space
    - `step::T:` ξ-resolution of the hash

Type to preform efficient and fairly stable `locate`ing on parametric curves. Newton's method is fast, 
but can be very unstable for general parametric curves. This is mitigated by supplying a close initial 
`uv` guess by interpolating `hash``, and by bounding the derivative adjustment in the Newton `refine`ment.

----

    NurbsLocator(curve,lims;t⁰=0,step=1,buffer=2,T=Float64,mem=Array)

Creates NurbsLocator by sampling the curve and finding the bounding box. This box is expanded by the amount `buffer`. 
The hash array is allocated to span the box with resolution `step` and initialized using `update!(::,curve,t⁰,samples)`.
"""
struct NurbsLocator{T,F<:Function,F2<:Function,A<:AbstractArray{T,2}}
    refine::F
    surf::F2
    lims::NTuple{2,T}
    hash::A
    lower::MVector{2,T}
    upper::MVector{2,T}
    step::T
end
Adapt.adapt_structure(to, x::NurbsLocator) = NurbsLocator(x.refine,x.surf,x.lims,adapt(to,x.hash),x.lower,x.upper,x.step)

function NurbsLocator(curve,lims;t⁰=0,step=1,buffer=2,T=Float64,mem=Array)
    # Apply type and get refinement function
    lims,t⁰,step = T.(lims),T(t⁰),T(step)
    f = refine(curve,lims,curve(first(lims),t⁰)≈curve(last(lims),t⁰))

    # Get curve's bounding box
    samples = range(lims...,64)
    lower = upper = curve(first(samples),t⁰)
    @assert eltype(lower)==T "`curve` is not type stable"
    @assert isa(curve(first(samples),t⁰),SVector{2,T}) "`curve` doesn't return a 2D SVector"

    # Allocate hash and struct, and update hash
    hash = fill(first(lims),ceil.(Int,(2,2))...) |> mem
    l=adapt(mem,NurbsLocator{T,typeof(f),typeof(curve),typeof(hash)}(f,curve,lims,hash,lower,upper,step))
    update!(l,curve,t⁰,samples)
end

notC¹(l::NurbsLocator,uv) = any(uv.≈l.lims)

"""
    update!(l::NurbsLocator,surf,t,samples=l.lims)

Updates `l.hash` for `surf` at time `t` by searching through `samples` and refining.
"""
update!(l::NurbsLocator,surf,t,samples=l.lims)=(_update!(get_backend(l.hash),64)(l,surf,samples,t,ndrange=size(l.hash));l)
@kernel function _update!(l::NurbsLocator{T},surf,@Const(samples),@Const(t)) where T
    # update bounding box
    l.lower .= l.upper .= surf(first(samples),t)
    for uv in samples
        x = surf(uv,t)
        l.lower .= min.(l.lower,x)
        l.upper .= max.(l.upper,x)
    end
    l.lower .= l.lower.-2*l.step
    l.upper .= l.upper.+2*l.step
end

"""
    (l::NurbsLocator)(x,t)

Estimate the parameter value `uv⁺ = argmin_uv (X-curve(uv,t))²` in two steps:
1. Interploate an initial guess  `uv=l.hash(x)`. Return `uv⁺~uv` if `x` is outside the hash domain.
2. Apply a bounded Newton step `uv⁺≈l.refine(x,uv,t)` to refine the estimate.
"""
function (l::NurbsLocator)(x,t)
    # check if the point is in bounding box
    inside = all(x.>l.lower) && all(x.<l.upper)

    # Grid search for uv within bounds
    @inline dis2(uv) = (q=x-l.surf(uv,t); q'*q)
    uv = 0.0; d = dis2(uv)
    for uvᵢ in range(l.lims...,64)
        dᵢ = dis2(uvᵢ)
        dᵢ<d && (uv=uvᵢ; d=dᵢ)
    end
    # if we are outside, this is sufficient
    !inside && return uv

    # Otherwise, refine estimate with two Newton steps
    uv = l.refine(x,uv,t)
    return l.refine(x,uv,t)
end