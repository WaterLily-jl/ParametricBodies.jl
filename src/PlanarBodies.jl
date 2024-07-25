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
    thk::T
end
function PlanarBody(curve,lims::Tuple;T=Float32,map=dmap,thk=T(√3/2+1),kwargs...)
    # Wrap in type safe functions (GPUs are picky)
    wcurve(u::U,t::T) where {U,T} = SVector{2,promote_type(U,T)}(curve(u,t))
    wmap(x::SVector{n,X},t::T) where {n,X,T} = SVector{n,promote_type(X,T)}(map(x,t))

    scale = T(ParametricBodies.get_scale(map,SA{T}[0,0,0]))
    locate = HashedLocator(wcurve,T.(lims);T,step=inv(scale),kwargs...)
    planform = ParametricBody(wcurve,locate)
    PlanarBody(planform,wmap,scale,thk)
end
Adapt.adapt_structure(to, x::PlanarBody) = PlanarBody(adapt(to,x.body),x.map,x.scale,x.thk)

function PlanarBody(curve::NurbsCurve;map=dmap,T=eltype(curve.pnts),thk=T(√3/2+1))
    # Wrap in type safe function (GPUs are picky)
    wmap(x::SVector{n,X},t::T) where {n,X,T} = SVector{n,promote_type(X,T)}(map(x,t))

    scale = T(ParametricBodies.get_scale(map,SA{T}[0,0,0]))
    locate = NurbsLocator(curve;step=inv(scale))
    planform = ParametricBody(curve,locate)
    PlanarBody(planform,wmap,scale,thk)
end

function curve_props(body::PlanarBody{T},x::SVector{3},t) where T
    # Get vector to point
    ξ = body.map(x,t)
    if body.scale*abs(ξ[3])<2body.thk # might be close to planar body
        d,n,_ = curve_props(body.planform,SA[ξ[1],ξ[2]],t)
        p = SA[max(d,0)*n[1],max(d,0)*n[2],ξ[3]]
    else
        p = SA[0,0,ξ[3]] # simple planar approximation
    end

    # return scaled distance, normal, and dot(S)=zero
    n = p/(eps(T)+√(p'*p))
    return body.scale*p'*n-body.thk,n,zero(x)
end
