"""
    NurbsLocator

    - `curve<:NurbsCurve` NURBS defined curve
    - `step<:Real` buffer size around control points
    - `C¹end::Bool` check if the curve is closed and C¹

NURBS-specific locator function. Loops through the spline sections, locating points accurately 
(using inverse cubic interpolation) only if it's possible for that to be the closest section.
"""
struct NurbsLocator{C<:NurbsCurve,T<:Number,S<:SVector} <: AbstractLocator
    curve::C
    step::T
    C¹end::Bool
    C::S
    R::S
end

function NurbsLocator(curve::NurbsCurve;step=1,t=0.)
    # Check ends
    low,high = first(curve.knots),last(curve.knots)
    c(u) = curve(u,t); dc(u) = ForwardDiff.derivative(c,u)
    C¹end = c(low)≈c(high) && dc(low)≈dc(high)
    # Control-point bounding box 
    ex = extrema(curve.pnts,dims=2)
    low,high = SVector(first.(ex)),SVector(last.(ex))
    NurbsLocator(curve,step,C¹end,0.5f0*(low+high),0.5f0(high-low))
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
function (l::NurbsLocator{C})(x,t,fast=false) where C<:NurbsCurve{n,d} where {n,d}
    fast && return √sum(abs2,max.(0,abs.(x-l.C)-l.R))
    loc_nurbs(l,x,t,Val(d))
end
function loc_nurbs(l,x,t,::Val{degree}) where degree # C¹⁺ NURBS locator
    # location, Dual distance and Davidon kwargs
    uv(i) = l.curve.knots[degree+i+1]
    dis2 = fdual(u->sum(abs2,x-l.curve(u,t)),uv(0))
    kwargs = (tol=5f-3,∂tol=5l.step,itmx=2degree)

    # Locate closest segment
    u = b = dis2(uv(0))
    for i in 1:length(l.curve.wgts)-degree
        a=b; b = dis2(uv(i))
        a==b && continue
        uᵢ,vᵢ = inv_cubic(dis2,a,b) # quick check
        uᵢ.f<2u.f && (uᵢ=davidon(dis2,uᵢ,vᵢ;kwargs...))
        uᵢ.f<u.f && (u=uᵢ)
    end; u.x # Best location
end
function loc_nurbs(l,x,t,::Val{1}) # Linear NURBS locator
    # location & function value pair
    uv(i) = 0.5f0(l.curve.knots[2+i÷2]+l.curve.knots[2+(i+1)÷2])
    dis2(u) = (x=u,f=sum(abs2,x-l.curve(u,t)))

    # Locate closest segment
    u = c = dis2(uv(0))
    for i in 1:length(l.curve.wgts)-1
        a=c; b,c = dis2.(uv.((2i-1:2i)))
        uᵢ = inv_quad(dis2,a,b,c)
        uᵢ.f<u.f && (u=uᵢ)
    end; u.x # Exact since d² is quadratic
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
    ParametricBody(curve,dotS,NurbsLocator(curve,step=body.locate.step),body.map,body.scale,body.half_thk,body.boundary)
end
update!(body::ParametricBody,uⁿ::AbstractArray,Δt) = update!(body,uⁿ,(uⁿ-copy(body.curve.pnts))/Δt)
