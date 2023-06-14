module ParametricBodies

using StaticArrays,ForwardDiff
include("HashedLocators.jl")
export HashedLocator

import WaterLily: AbstractBody,measure,sdf
struct ParametricBody{S<:Function,L<:Union{Function,HashedLocator},I<:Function,N<:Number} <: AbstractBody
    surf::S    #ξ = surf(uv,t)
    locate::L  #uv = locate(surf,ξ,t)
    map::I     #ξ = map(x,t)
    scale::N   #|dx/dξ| = scale
end
ParametricBody(surf,locate,map=(x,t)->x,x::SVector=SA[0.,0.]) = ParametricBody(surf,locate,map,get_scale(map,x))

import LinearAlgebra: det
get_scale(map,x::SVector{D},t=0.) where D = (dξdx=ForwardDiff.jacobian(x->map(x,t),x); abs(det(dξdx))^(-1/D))

function measure(body::ParametricBody,x::SVector,t)
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

function sdf(body::ParametricBody,x::SVector,t;uv=body.locate(body.map(x,t),t))
    p = body.map(x,t)-body.surf(uv,t)
    return body.scale*copysign(√(p'*p),norm_dir(body.surf,uv,t)'*p)
end

function norm_dir(surf,uv::Number,t)
    s = ForwardDiff.derivative(uv->surf(uv,t),uv)
    return SA[s[2],-s[1]]
end

ParametricBody(surf,uv_bounds::Tuple;step=1,t⁰=0.,T=Float64,mem=Array,map=(x,t)->x) = 
    ParametricBody(surf,HashedLocator(surf,uv_bounds;step,t⁰,T,mem),map)

update!(body::ParametricBody{F,L},t) where {F<:Function,L<:HashedLocator} = 
    update!(body.locate,body.surf,t)

export ParametricBody

end
