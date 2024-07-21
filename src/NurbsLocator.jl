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

function notC¹(l::NurbsLocator{NurbsCurve{n,d}},uv) where {n,d}
    d==1 && return !any(uv.≈l.curve.knots) # straight line spline is not C¹ at any knot
    # Assuming we don't have repeated knots, ends are the only remaining potential not C¹ locations
    low,high = first(l.curve.knots),last(l.curve.knots)
    (uv≈low || uv≈high) ? !l.C¹end : false 
end

"""
    (l::NurbsLocator)(x,t)

Estimate the parameter value `u⁺ = argmin_u (x-curve(u,t))²` for a NURBS by looping through the 
spline segments.
"""
function (l::NurbsLocator{C})(x,t) where C<:NurbsCurve{n,degree} where {n,degree}
    @inline dis2(uv) = sum(abs2,x-l.curve(uv,t))
    function check_segment(s)
        # uv at center of segment
        u = 0.5f0(l.curve.knots[degree+s]+l.curve.knots[degree+s+1])

        # squared distance outside the bounding box
        q2 = box(x,@view(l.curve.pnts[:,s:s+degree]))

        # if we are outside, this is sufficient
        q2>9l.step^2 && return q2,u

        # otherwise refine twice
        u = l.refine(x,u,t)
        u = l.refine(x,u,t)
        dis2(u),u
    end

    # Return uv of closest segment
    d2,uv = check_segment(1)
    for s ∈ 2:length(l.curve.wgts)-degree
        d2ᵢ,uvᵢ = check_segment(s)
        d2ᵢ<d2 && (uv=uvᵢ; d2=d2ᵢ)
    end; uv
end
function box(x,pnts)
    ex = extrema(pnts,dims=2)
    low,high = SA[first.(ex)...],SA[last.(ex)...]
    c,p = 0.5f0(high+low),0.5f0(high-low)
    sum(abs2,max.(abs.(x-c)-p,0))
end

"""
    DynamicNurbsBody(curve;kwargs...)

Creates a `ParametricBody` with `locate=NurbsLocator`, mutable curve points,
and `dotS` defined by a second spline curve.
"""
function DynamicNurbsBody(curve::NurbsCurve;step=1,kwargs...)
    # Make a mutable version of the curve
    mcurve = NurbsCurve(MMatrix(curve.pnts),curve.knots,curve.wgts)
    # Make a velocity curve (init with 0)
    dotS = copy(mcurve); dotS.pnts .= 0 
    # Make body
    ParametricBody(mcurve,NurbsLocator(mcurve;step);dotS,kwargs...)
end
function update!(body::ParametricBody{T,L,S,dS},uⁿ::AbstractArray,vⁿ::AbstractArray) where {T,L<:NurbsLocator,S<:NurbsCurve,dS<:NurbsCurve}
    body.surf.pnts .= uⁿ
    body.dotS.pnts .= vⁿ
    update!(body.locate,body.surf,0.0)
end
update!(body::ParametricBody,uⁿ::AbstractArray,Δt) = update!(body,uⁿ,(uⁿ-copy(body.surf.pnts))/Δt)
