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
The hash array is allocated to span the box with resolution `step` and initialized using `update!(::,curve,t⁰)`.
"""
struct NurbsLocator{T,F<:Function,G<:Function,B<:AbstractVector{T}}
    refine::F
    surf::G
    lims::NTuple{2,T}
    lower::B
    upper::B
    step::T
end
Adapt.adapt_structure(to, x::NurbsLocator) = NurbsLocator(x.refine,x.surf,x.lims,adapt(to,x.lower),adapt(to,x.upper),x.step)

function NurbsLocator(curve::NurbsCurve{n},lims;t⁰=0,step=1,buffer=2,mem=Array) where n
    # Apply type and get refinement function
    T = eltype(curve.pnts); lims,t⁰,step = T.(lims),T(t⁰),T(step)
    f = refine(curve,lims,curve(first(lims),t⁰)≈curve(last(lims),t⁰))

    # Get curve's bounding box
    lower = curve(first(lims),t⁰) |> mem; upper = curve(last(lims),t⁰) |> mem
    @assert eltype(lower)==T "`curve` is not type stable"
    @assert isa(curve(first(lims),t⁰),SVector{n,T}) "`curve` doesn't return a 2D SVector"

    # Allocate struct, and update
    l=adapt(mem,NurbsLocator{T,typeof(f),typeof(curve),typeof(lower)}(f,curve,lims,lower,upper,step))
    update!(l,curve,t⁰)
end

# if it's open, we need to check that we are not at the endpoints
notC¹(l::NurbsLocator,uv) = !(l.surf(first(l.lims),0)≈l.surf(last(l.lims),0)) && any(uv.≈l.lims)

"""
    update!(l::NurbsLocator,surf,t)

Updates `l` for `surf` at time `t` by searching through `samples` and refining.
"""
update!(l::NurbsLocator,surf,t)=(_update!(get_backend(l.lower),64)(l,surf,t,ndrange=size(l.lower));l)
@kernel function _update!(l::NurbsLocator{T},surf,@Const(t)) where T
    # update bounding box
    l.lower .= l.upper .= surf(zero(T),t)
    # the cps net in a convex hull of the curve
    for x in eachcol(surf.pnts)
        l.lower .= min.(l.lower,x)
        l.upper .= max.(l.upper,x)
    end
    l.lower .= l.lower.-2*l.step
    l.upper .= l.upper.+2*l.step
end

"""
    (l::NurbsLocator)(x,t)

Estimate the parameter value `uv⁺ = argmin_uv (X-curve(uv,t))²` in two steps:
1. Interploate an initial guess  `uv=l(x)`. Return `uv⁺~uv` if `x` is outside the bounding box.
2. Apply a bounded Newton step `uv⁺≈l.refine(x,uv,t)` to refine the estimate.
"""
function (l::NurbsLocator{T})(x,t) where T
    # check if the point is in bounding box
    inside = all(x.>l.lower) && all(x.<l.upper)
    # if we are outside, this is sufficient
    !inside && return T(0.5)

    # Grid search for uv within bounds
    @inline dis2(uv) = (q=x-l.surf(uv,t); q'*q)
    uv = zero(T); d = dis2(uv)
    for uvᵢ in range(l.lims...,64)
        dᵢ = dis2(uvᵢ)
        dᵢ<d && (uv=uvᵢ; d=dᵢ)
    end

    # Otherwise, refine estimate with two Newton steps
    uv = l.refine(x,uv,t)
    return l.refine(x,uv,t)
end