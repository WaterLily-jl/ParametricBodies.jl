module ParametricBodies

using StaticArrays,ForwardDiff
include("HashedLocators.jl")
export HashedLocator

import WaterLily: AbstractBody,measure,sdf
struct ParametricBody{T,S<:Function,L<:Union{Function,HashedLocator},M<:Function} <: AbstractBody
    surf::S    #ξ = surf(uv,t)
    locate::L  #uv = locate(ξ,t)
    map::M     #ξ = map(x,t)
    scale::T   #|dx/dξ| = scale
end
function ParametricBody(surf,locate;map=(x,t)->x,T=Float64)
    # Check input functions
    x,t = SVector{2,T}(0,0),T(0); ξ = map(x,t); uv = locate(ξ,t); p = ξ-surf(uv,t)
    @assert isa(ξ,SVector{2,T}) "map is not type stable"
    @assert isa(uv,T) "locate is not type stable"
    @assert isa(p,SVector{2,T}) "surf is not type stable"
  
    ParametricBody(surf,locate,map,T(get_scale(map,x)))
end

import LinearAlgebra: det
get_scale(map,x::SVector{D},t=0.) where D = (dξdx=ForwardDiff.jacobian(x->map(x,t),x); abs(det(dξdx))^(-1/D))

function measure(body::ParametricBody{T},x::SVector{N,T},t::T) where {N,T}
    # Compute n=∇sdf. This must include ∇uv, so uv can't be input
    n = ForwardDiff.gradient(x->sdf(body,x,t), x) 

    # Precompute uv(x,t) and compute distance 
    uv = body.locate(body.map(x,t),t)
    d = sdf(body,x,t;uv)

    # Expand V = dx/dt = (dξ/dx)\(dξ/dt) to avoid defining map⁻¹
    dξdx = ForwardDiff.jacobian(x->body.map(x,t),x)
    dξdt = ForwardDiff.derivative(t->body.surf(uv,t)-body.map(x,t),t)
    return (d,n,dξdx\dξdt)
end

function sdf(body::ParametricBody{T},x::SVector{N,xT},t::T;uv=body.locate(body.map(x,t),t)) where {N,T,xT<:Union{T,ForwardDiff.Dual}}
    p = body.map(x,t)-body.surf(uv,t)
    return body.scale*copysign(√(p'*p),norm_dir(body.surf,uv,t)'*p)
end

function norm_dir(surf,uv::Number,t)
    s = ForwardDiff.derivative(uv->surf(uv,t),uv)
    return SA[s[2],-s[1]]
end

Adapt.adapt_structure(to, x::ParametricBody{T,F,L}) where {T,F,L<:HashedLocator} =
    ParametricBody(x.surf,adapt(to,x.locate),x.map,x.scale)

ParametricBody(surf,uv_bounds::Tuple;step=1,t⁰=0.,T=Float64,mem=Array,map=(x,t)->x) = 
    adapt(mem,ParametricBody(surf,HashedLocator(surf,uv_bounds;step,t⁰,T,mem);map,T))

update!(body::ParametricBody{T,F,L},t) where {T,F,L<:HashedLocator} = 
    update!(body.locate,body.surf,t)

export ParametricBody

end
