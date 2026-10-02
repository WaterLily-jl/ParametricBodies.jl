"""
    NurbsLocator(curve::NurbsCurve)

NURBS-specific locator function. Loops through the spline sections, to find an inital guess
based on the `degree=1` (straight-line) version, and then refine. Unlike `HashedLocator`
this locator doesn't need to be initialized.
"""
struct NurbsLocator{C<:NurbsCurve,F<:Function,V<:SVector,T} <: AbstractLocator
    curve::C
    C¹end::Bool
    refine::F
    lo::V      # bounding box of the control points. A NURBS with positive weights lies in
    hi::V      # their convex hull, so the distance to this box lower-bounds `d`
    thkmax::T  # largest half-thickness, the allowance that bound needs for a thick body
end

"""
    NurbsLocator(curve::NurbsCurve;thkmax=Inf)

`thkmax` is the largest half-thickness of the body this locator will serve, and is what lets
[`far_field`](@ref) skip the search for points nowhere near the curve. It defaults to `Inf`,
which disables that shortcut -- a locator built without knowing the thickness cannot tell a
far point from one inside a very thick body, so it must not guess.
"""
function NurbsLocator(curve::NurbsCurve{N};thkmax=eltype(curve.pnts)(Inf)) where N
    low,high = first(curve.knots),last(curve.knots)
    C¹end = curve(low)≈curve(high) && tangent(curve,low,0)≈tangent(curve,high,0) # closed C¹ curve?
    lo = SVector{N}(minimum(curve.pnts[i,:]) for i in 1:N)
    hi = SVector{N}(maximum(curve.pnts[i,:]) for i in 1:N)
    NurbsLocator(curve,C¹end,refine(curve,(low,high),C¹end),lo,hi,eltype(curve.pnts)(thkmax))
end
Adapt.adapt_structure(to, x::NurbsLocator) = NurbsLocator(x.curve,x.C¹end,x.refine,x.lo,x.hi,x.thkmax)

update!(l::NurbsLocator,curve,t) = l=NurbsLocator(curve;thkmax=l.thkmax) # just make a new locator

"""
    max_half_thk(half_thk,curve,n=65)

Largest half-thickness over the curve's parameter range, sampled. Sampling is enough for the
tapered and piecewise thickness functions in use; a pathological `half_thk` would need its
own bound.
"""
max_half_thk(half_thk,curve,n=65) = maximum(half_thk(u) for u in range(first(curve.knots),last(curve.knots),n))

"""
    far_field(body,ξ,fastd²)

The `(d,n,V)` of a point that provably cannot lie within `fastd²` of the body, or `nothing`
when it might. `scale`×(distance from `ξ` to the control-point box) − `thkmax` lower-bounds
the signed distance, so a point beyond the band by that measure stays beyond it however the
search refines -- and the whole locate, the curve evaluations and the normal logic can be
skipped. `d` is that bound rather than the exact distance, the same capping `MeshBody` does
with `√fastd²`, so a field of `sdf` values is only exact inside the band. `fastd²=0` means
the caller wants no shortcut.
"""
function far_field(body::ParametricBody{T,L},ξ,fastd²) where {T,L<:NurbsLocator}
    d = body.scale*boxd(ξ,body.locate.lo,body.locate.hi)-body.locate.thkmax
    (fastd²>0 && d>0 && d^2≥fastd²) ? (d,zero(ξ),zero(ξ)) : nothing
end
@inline boxd(ξ,lo,hi) = sqrt(sum(abs2,max.(lo.-ξ,ξ.-hi,zero(ξ))))

function notC¹(l::NurbsLocator{C},uv) where C<:NurbsCurve{n,d} where {n,d}
    d==1 && return any(uv.≈l.curve.knots) # straight line spline is not C¹ at any knot
    # Assuming we don't have repeated knots, ends are the only remaining potential not C¹ locations
    low,high = first(l.curve.knots),last(l.curve.knots)
    (uv≈low || uv≈high) ? !l.C¹end : false 
end
function eachside(l::NurbsLocator,uv,s=√eps(typeof(uv)))
    low,high = first(l.curve.knots),last(l.curve.knots)
    l.curve.pnts[:,1] ≈ l.curve.pnts[:,end] ? mymod.(uv .+(-s,s),low,high) : clamp.(uv .+(-s,s),low,high)
end

lims(b::ParametricBody{T,L}) where {T,L<:NurbsLocator} = (first(b.curve.knots),last(b.curve.knots))
"""
    (l::NurbsLocator)(x,t;fastd²=Inf)

Estimate the parameter value `u⁺ = argmin_u (x-l.curve(u))²` for a NURBS in two steps
1. The nearest point `u` on the `degree=1` version of the curve is found. Return this if degree==1.
2. Otherwse `refine` this guess until converged or the square distance ≥ `fastd²`. 
"""
function (l::NurbsLocator{C})(x,t;fastd²=Inf) where C<:NurbsCurve{N,degree} where {N,degree}
    # Closest parameter on linear NURBS
    pnts = l.curve.pnts; n = size(pnts,2)-1
    b = pnts[:,1]; u,d² = zero(eltype(pnts)),sum(abs2,x-b)
    for i in 1:n                        # Loop through segments
        a = b; b = pnts[:,i+1]               # segment a-to-b
        a==b && continue                     # skip zero length segments
        s = b-a                              # tangent vector
        p = clamp(((x-a)'*s)/(s'*s),0,1)     # perp distance along s
        uᵢ,d²ᵢ = (i-1+p)/n,sum(abs2,x-a-s*p) # segment minimizer
        d²ᵢ<d² && (u=uᵢ;d²=d²ᵢ)              # update if uᵢ is closests
    end
    # Return if degree=1, otherwise refine
    degree == 1 ? u : l.refine(u,x,t;fastd²)
end
"""
    ParametricBody(curve::NurbsCurve;kwargs...)

Creates a `ParametricBody` with `locate=NurbsLocator`.
"""
function ParametricBody(curve::NurbsCurve{N};T=eltype(curve.pnts),ndims=N,kwargs...) where N
    # the locator is told the thickness here, the one place it is known, so that `far_field`
    # can reject points off the body without mistaking a thick body's surface for far away
    thkmax = max_half_thk(make_func(get(kwargs,:thk,zero(T))),curve)
    ParametricBody(curve,NurbsLocator(curve;thkmax);T,ndims,kwargs...)
end

"""
    DynamicNurbsBody(curve::NurbsCurve;kwargs...)

Creates a `ParametricBody` with `locate=NurbsLocator`, and `dotS` defined by a second spline curve.
"""
function DynamicNurbsBody(curve::NurbsCurve;kwargs...)
    # Make a zero velocity spline
    dotS = NurbsCurve(zeros(typeof(curve.pnts)),curve.knots,curve.wgts)
    # Make body
    ParametricBody(curve;dotS,kwargs...)
end
function update!(body::ParametricBody{T,L,S},uⁿ::AbstractArray{T},vⁿ::AbstractArray{T}) where {T,L<:NurbsLocator,S<:NurbsCurve}
    curve = NurbsCurve(uⁿ,body.curve.knots,body.curve.wgts)
    dotS = NurbsCurve(vⁿ,body.curve.knots,body.curve.wgts)
    ParametricBody(curve,dotS,NurbsLocator(curve;thkmax=body.locate.thkmax),body.map,body.scale,body.half_thk,body.boundary)
end
update!(body::ParametricBody,uⁿ::AbstractArray,Δt::Number) = update!(body,uⁿ,(uⁿ-copy(body.curve.pnts))/Δt)
