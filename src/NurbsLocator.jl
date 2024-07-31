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
Adapt.adapt_structure(to, x::NurbsLocator) = NurbsLocator(x.curve,x.step,x.C¹end,x.C,x.R)

update!(l::NurbsLocator,curve,t) = l=NurbsLocator(curve,step=l.step;t) # just make a new locator

function notC¹(l::NurbsLocator{C},uv) where C<:NurbsCurve{n,d} where {n,d}
    d==1 && return any(uv.≈l.curve.knots) # straight line spline is not C¹ at any knot
    # Assuming we don't have repeated knots, ends are the only remaining potential not C¹ locations
    low,high = first(l.curve.knots),last(l.curve.knots)
    (uv≈low || uv≈high) ? !l.C¹end : false 
end
"""
    (l::NurbsLocator)(x,t)

Estimate the parameter value `u⁺ = argmin_u (x-curve(u,t))²` for a NURBS by looping through the 
spline segments.
"""
function (l::NurbsLocator{C})(x,t,fast=false;tol=5f-3,∂tol=5l.step,itmx=2degree) where C<:NurbsCurve{n,degree} where {n,degree}
    fast && return √sum(abs2,max.(0,abs.(x-l.C)-l.R))
    degree==1 && return lin_loc(l,x,t)

    # location and Dual distance function
    uv(i) = l.curve.knots[degree+i+1]
    dis2(u) = fdual(u->sum(abs2,x-l.curve(u,t)),u)  

    # Locate closest segment
    u = b = dis2(uv(0))
    for i in 1:length(l.curve.wgts)-degree
        a = b; b = dis2(uv(i))
        a==b && continue
        (aᵢ,bᵢ) = a.f<b.f ? (a,b) : (b,a)  # aᵢ is current minimizer
        uᵢ,vᵢ = inv_cubic(dis2,aᵢ,bᵢ;tol)  # first refinement
        uᵢ.f<2u.f && for _ in 1:itmx       # requires accurate search
            (abs(uᵢ.x-vᵢ.x) ≤ 2tol || abs(uᵢ.∂) < ∂tol ||(uᵢ,vᵢ)==(aᵢ,bᵢ)) && break
            aᵢ,bᵢ = uᵢ,vᵢ
            uᵢ,vᵢ = inv_cubic(dis2,aᵢ,bᵢ;tol)
        end
        uᵢ.f<u.f && (u=uᵢ) # Replace current best
    end; u.x               # Return location
end
# Inversed Cubic Interpolation minimizer
using ForwardDiff: Dual,Tag,value,partials
function fdual(f::F,x::R) where {F<:Function,R<:AbstractFloat}
    T = typeof(Tag(f,R))
    fx = f(Dual{T}(x,one(R)))
    (x=x,f=value(fx),∂=partials(T,fx,1))
end
function inv_cubic(f,a,b;tol=√eps(a.x))
    Δ = b.x-a.x
    v = a.∂+b.∂-3(b.f-a.f)/Δ; w = v^2-a.∂*b.∂
    w < 0 && return a,b      # bust!
    w = copysign(√w,Δ); q = b.∂-a.∂+2w
    !(0<(b.∂+w-v)/q<1) && return a,b # bust!
    margin = max(0.1f0,tol/abs(Δ))
    c = f(b.x-Δ*clamp((b.∂+w-v)/q,margin,1-margin))
    c.f > b.f && return a,b  # bust!
    c.f > a.f && return a,c  # save minimizer
    c,(c.∂*Δ<0 ? b : a)      # pick "downhill" bracket
end
function lin_loc(l::NurbsLocator,x,t)
    uv(i) = l.curve.knots[1+i]
    dis2(u) = (x=u,f=sum(abs2,x-l.curve(u,t)))

    # Locate closest segment
    u = dis2(uv(1)); b = l.curve(uv(1),t)
    for i in 1:length(l.curve.wgts)-1
        a = b; b = l.curve(uv(i+1),t)
        a==b && continue
        s = b-a                            # tangent
        p = clamp(((x-a)'*s)/(s'*s),0,1)   # projected distance
        uᵢ = dis2(uv(i)+(uv(i+1)-uv(i))*p) # segment min
        uᵢ.f<u.f && (u=uᵢ) # Replace current best
    end; u.x               # Return location
end
"""
    ParametricBody(curve::NurbsCurve;kwargs...)

Creates a `ParametricBody` with `locate=NurbsLocator`.
"""
ParametricBody(curve::NurbsCurve;step=1,T=eltype(curve.pnts),kwargs...) = ParametricBody(curve,NurbsLocator(curve;step);T,kwargs...)

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
    ParametricBody(curve,dotS,NurbsLocator(curve,step=body.locate.step),body.map,body.scale,body.half_thk,body.boundary)
end
update!(body::ParametricBody,uⁿ::AbstractArray,Δt::Number) = update!(body,uⁿ,(uⁿ-copy(body.curve.pnts))/Δt)
