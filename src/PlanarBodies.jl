"""
    PlanarBody(curve::NurbsCurve;map)       # creates NurbsLocator
    PlanarBody(curve,lims::Tuple;T,map,mem) # creates HashedLocator

Embeds a ParametricBody onto the `ζ₃=0` plane defined by `map`. The `curve` 
defines the planform of a planar body with minimial thickness `thk=ϵ+√3/2`. 
See Lauber & Weymouth 2022. Note, the velocity is defined soley from `dot(map)`.
"""
struct PlanarBody{T,P<:ParametricBody,F<:Function} <: AbstractParametricBody
    planform::P
    map::F
    scale::T
    half_thk::T
end
function PlanarBody(curve,lims::Tuple;T=Float32,map=dmap,thk=T(√3+2),kwargs...)
    # Wrap in type safe functions (GPUs are picky)
    wcurve(u::U,t::T) where {U,T} = SVector{2,promote_type(U,T)}(curve(u,t))
    wmap(x::SVector{n,X},t::T) where {n,X,T} = SVector{n,promote_type(X,T)}(map(x,t))

    scale = T(ParametricBodies.get_scale(map,SA{T}[0,0,0]))
    locate = HashedLocator(wcurve,T.(lims);T,step=inv(scale),kwargs...)
    planform = ParametricBody(wcurve,locate)
    PlanarBody(planform,wmap,scale,T(thk/2))
end
Adapt.adapt_structure(to, x::PlanarBody) = PlanarBody(adapt(to,x.planform),x.map,x.scale,x.half_thk)

function PlanarBody(curve::NurbsCurve;map=dmap,T=eltype(curve.pnts),thk=T(√3+2))
    # Wrap in type safe function (GPUs are picky)
    wmap(x::SVector{n,X},t::T) where {n,X,T} = SVector{n,promote_type(X,T)}(map(x,t))

    scale = T(ParametricBodies.get_scale(map,SA{T}[0,0,0]))
    PlanarBody(ParametricBody(curve),wmap,scale,T(thk/2))
end

function curve_props(body::PlanarBody{T},x::SVector{3},t;fastd²=Inf) where T
    # Get vector to point
    ξ = body.map(x,t)
    if body.scale*abs(ξ[3])<2body.half_thk # might be close to planar body
        d,n,_ = curve_props(body.planform,SA[ξ[1],ξ[2]],t;fastd²)
        d^2>fastd² && return d,zero(x),zero(x) # can't trust n
        p = SA[max(d,0)*n[1],max(d,0)*n[2],ξ[3]]
    else
        p = SA[0,0,ξ[3]] # simple planar approximation
    end

    # return scaled distance, normal, and dot(S)=zero
    n = hat(p)
    return body.scale*p'*n-body.half_thk,n,zero(x)
end
