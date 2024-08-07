"""
    NurbsLocator(curve::NurbsCurve)

NURBS-specific locator function. Loops through the spline sections, locating points accurately 
using inverse cubic interpolation if it's possible for that to be the closest section.
"""
struct NurbsLocator{C<:NurbsCurve,S<:SVector} <: AbstractLocator
    curve::C
    C¹end::Bool
    C::S
    R::S
end

function NurbsLocator(curve::NurbsCurve)
    # Check ends
    low,high = first(curve.knots),last(curve.knots)
    dc(u) = ForwardDiff.derivative(curve,u)
    C¹end = curve(low)≈curve(high) && dc(low)≈dc(high)
    # Control-point bounding box 
    ex = extrema(curve.pnts,dims=2)
    low,high = SVector(first.(ex)),SVector(last.(ex))
    NurbsLocator(curve,C¹end,0.5f0*(low+high),0.5f0(high-low))
end
Adapt.adapt_structure(to, x::NurbsLocator) = NurbsLocator(x.curve,x.C¹end,x.C,x.R)

update!(l::NurbsLocator,curve,t) = l=NurbsLocator(curve) # just make a new locator

function notC¹(l::NurbsLocator{C},uv) where C<:NurbsCurve{n,d} where {n,d}
    d==1 && return any(uv.≈l.curve.knots) # straight line spline is not C¹ at any knot
    # Assuming we don't have repeated knots, ends are the only remaining potential not C¹ locations
    low,high = first(l.curve.knots),last(l.curve.knots)
    (uv≈low || uv≈high) ? !l.C¹end : false 
end
"""
    (l::NurbsLocator)(x,t)

Estimate the parameter value `u⁺ = argmin_u (x-l.curve(u))²` for a NURBS by looping through the 
spline segments. `t` is unused.
"""
function (l::NurbsLocator{C})(x,t,fast=false;tol=5f-3,itmx=2degree) where C<:NurbsCurve{n,degree} where {n,degree}
    fast && return √sum(abs2,max.(0,abs.(x-l.C)-l.R))
    degree==1 && return lin_loc(l,x)

    # location and Dual distance function
    uv(i) = l.curve.knots[degree+i+1]
    dis2(u) = fdual(u->sum(abs2,x-l.curve(u)),u)  

    # Locate closest segment
    u = b = dis2(uv(0))
    for i in 1:length(l.curve.wgts)-degree
        a = b; b = dis2(uv(i))
        a==b && continue
        uᵢ = davidon(dis2,a,b;fmax=2u.f,tol,itmx)
        uᵢ.f<u.f && (u=uᵢ) # Replace current best
    end; u.x               # Return location
end
# Returns x=argument, f=function value(x) and ∂=df/dx(x) as a named tuple
using ForwardDiff: Dual,Tag,value,partials
function fdual(f::F,x::R) where {F<:Function,R<:AbstractFloat}
    T = typeof(Tag(f,R))
    fx = f(Dual{T}(x,one(R)))
    (x=x,f=value(fx),∂=partials(T,fx,1))
end
# Inversed Cubic Interpolation minimizer
davidon(f,a::Number,b::Number;kwargs...) = (ff(x)=fdual(f,x); davidon(ff,ff(a),ff(b);kwargs...).x)
@inline function davidon(f,a,b;tol=√eps(typeof(a.x)),∂tol=0,fmax=Inf,itmx=1000)
    (a,b) = a.f<b.f ? (a,b) : (b,a) # a is current minimizer
    u,v = inv_cubic(f,a,b,tol)      # first refinement
    u.f<fmax && for _ in 1:itmx     # requires accurate search
        (abs(u.x-v.x) ≤ 2tol || abs(u.∂) ≤ ∂tol ||(u,v)==(a,b)) && break
        a,b = u,v
        u,v = inv_cubic(f,a,b,tol)  # keep refining bracket
    end; u # return minimizer
end
function inv_cubic(f::F,a,b,tol) where F
    Δ = b.x-a.x
    v = a.∂+b.∂-3(b.f-a.f)/Δ; w = v^2-a.∂*b.∂
    w < tol && return a,b    # done: co-linear!
    w = copysign(√w,Δ); q = (b.∂+w-v)/(b.∂-a.∂+2w)
    !(0<q<1) && return a,b   # done: outside the bracket!
    margin = max(0.1f0,tol/abs(Δ))
    c = f(b.x-Δ*clamp(q,margin,1-margin))
    c.f > b.f && return a,b  # fail: regression
    c.f > a.f && return a,c  # save minimizer
    c,(c.∂*Δ<0 ? b : a)      # pick "downhill" bracket
end
function lin_loc(l::NurbsLocator,x)
    uv(i) = l.curve.knots[1+i]
    dis2(u) = (x=u,f=sum(abs2,x-l.curve(u)))

    # Locate closest segment
    u = dis2(uv(1)); b = l.curve(uv(1))
    for i in 1:length(l.curve.wgts)-1
        a = b; b = l.curve(uv(i+1))
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
ParametricBody(curve::NurbsCurve;T=eltype(curve.pnts),kwargs...) = ParametricBody(curve,NurbsLocator(curve);T,kwargs...)

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
