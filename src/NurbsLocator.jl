"""
    NurbsLocator

    - `curve<:NurbsCurve` NURBS defined curve
    - `refine<:Function` Performs bounded Newton root-finding step
    - `step<:Real` buffer size around control points
    - `C¹end::Bool` check if the curve is closed and C¹

NURBS-specific locator function. Defines local bounding boxes using the control point locations instead of a
hash function. If point is within a section, Newton `refine`ment is used.
"""
struct NurbsLocator{C<:NurbsCurve,F<:Function,T<:Number} <: AbstractLocator
    curve::C
    refine::F
    step::T
    C¹end::Bool
end

function NurbsLocator(curve::NurbsCurve;step=1,t=0.)
    # Check ends
    low,high = first(curve.knots),last(curve.knots)
    c(u) = curve(u,t); dc(u) = ForwardDiff.derivative(c,u)
    C¹end = c(low)≈c(high) && dc(low)≈dc(high)
    f = refine(curve,(low,high),C¹end)
    NurbsLocator(curve,f,step,C¹end)
end

update!(l::NurbsLocator,curve,t) = l=NurbsLocator(curve,step=l.step;t) # just make a new locator

function notC¹(l::NurbsLocator{C},uv) where C<:NurbsCurve{n,d} where {n,d}
    d==1 && return any(uv.≈l.curve.knots) # straight line spline is not C¹ at any knot
    # Assuming we don't have repeated knots, ends are the only remaining potential not C¹ locations
    low,high = first(l.curve.knots),last(l.curve.knots)
    (uv≈low || uv≈high) ? !l.C¹end : false 
end
include("Davidon.jl")
"""
    (l::NurbsLocator)(x,t)

Estimate the parameter value `u⁺ = argmin_u (x-curve(u,t))²` for a NURBS by looping through the 
spline segments.
"""
function (l::NurbsLocator{C})(x,t) where C<:NurbsCurve{n,degree} where {n,degree}
    dis2(u) = sum(abs2,x-l.curve(u,t))
    S = length(l.curve.wgts)-degree
    uv(i) = 0.5f0(l.curve.knots[degree+1+i÷2]+l.curve.knots[degree+1+(i+1)÷2])

    # Find closest segement
    s,d2 = 1,dis2(uv(1))
    for i in 2:2S-1
        d2ᵢ = dis2(uv(i))
        d2ᵢ<d2 && (d2=d2ᵢ; s=i)
    end; u = uv(s)

    # If well outside the bounding box, this is sufficient
    box(x,l.curve.pnts,(s+1)÷2,(s+1)÷2+degree)>9l.step^2 && return u

    # Otherwise use Davidon
    f = fdual(dis2,u); a = f(u)
    b = f(uv(s-copysign(1,a.∂)))
    _davidon(f,a,b,∂tol=l.step).x
end
function box(x,pnts,s,e)
    low = high = pnts[:,s]
    for j in s+1:e
        pj = pnts[:,j]
        low = min.(low,pj); high = max.(high,pj)
    end
    c,p = 0.5f0(high+low),0.5f0(high-low)
    sum(abs2,max.(abs.(x-c)-p,0))
end

"""
    ParametricBody(curve::NurbsCurve;kwargs...)

Creates a `ParametricBody` with `locate=NurbsLocator`.
"""
ParametricBody(curve::NurbsCurve;step=1,kwargs...) = ParametricBody(curve,NurbsLocator(curve;step);kwargs...)

"""
    DynamicNurbsBody(curve::NurbsCurve;kwargs...)

Creates a `ParametricBody` with `locate=NurbsLocator`, and `dotS` defined by a second spline curve.
"""
function DynamicNurbsBody(curve::NurbsCurve;step=1,kwargs...)
    # Make a zero velocity spline
    dotS = NurbsCurve(zeros(typeof(curve.pnts)),curve.knots,curve.wgts)
    # Make body
    ParametricBody(curve,NurbsLocator(curve;step);dotS,kwargs...)
end
function update!(body::ParametricBody{T,L,S},uⁿ::AbstractArray{T},vⁿ::AbstractArray{T}) where {T,L<:NurbsLocator,S<:NurbsCurve}
    curve = NurbsCurve(uⁿ,body.curve.knots,body.curve.wgts)
    dotS = NurbsCurve(vⁿ,body.curve.knots,body.curve.wgts)
    ParametricBody(curve,dotS,NurbsLocator(curve,step=body.locate.step),body.map,body.scale,body.thk,body.boundary)
end
update!(body::ParametricBody,uⁿ::AbstractArray,Δt) = update!(body,uⁿ,(uⁿ-copy(body.curve.pnts))/Δt)
