"""
    NurbsLocator(curve::NurbsCurve)

NURBS-specific locator function. Loops through the spline sections, to find an inital guess
based on the `degree=1` (straight-line) version, and then refine. Unlike `HashedLocator`
this locator doesn't need to be initialized.
"""
struct NurbsLocator{C<:NurbsCurve,F<:Function} <: AbstractLocator
    curve::C
    C¹end::Bool
    refine::F
end

function NurbsLocator(curve::NurbsCurve)
    low,high = first(curve.knots),last(curve.knots)
    C¹end = curve(low)≈curve(high) && tangent(curve,low,0)≈tangent(curve,high,0) # closed C¹ curve?
    NurbsLocator(curve,C¹end,refine(curve,(low,high),C¹end))
end
Adapt.adapt_structure(to, x::NurbsLocator) = NurbsLocator(x.curve,x.C¹end,x.refine)

update!(l::NurbsLocator,curve,t) = l=NurbsLocator(curve) # just make a new locator

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
ParametricBody(curve::NurbsCurve{N};T=eltype(curve.pnts),ndims=N,kwargs...) where N = ParametricBody(curve,NurbsLocator(curve);T,ndims,kwargs...)

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
    ParametricBody(curve,dotS,NurbsLocator(curve),body.map,body.scale,body.half_thk,body.boundary)
end
update!(body::ParametricBody,uⁿ::AbstractArray,Δt::Number) = update!(body,uⁿ,(uⁿ-copy(body.curve.pnts))/Δt)
