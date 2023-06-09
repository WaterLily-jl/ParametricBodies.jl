module ParametricBodies

using StaticArrays,ForwardDiff
include("HashedLocators.jl")
export HashedLocator

import WaterLily: AbstractBody
struct ParametricBody{S<:Function,L<:Union{Function,HashedLocator}} <: AbstractBody
    surf::S    #x = surf(uv,t)
    locate::L  #uv = locate(surf,x,t)
end

function measure(body::ParametricBody,x::SVector,t)
    n = ForwardDiff.gradient(x->sdf(body,x,t), x)    # don't precompute uv!
    uv = body.locate(x,t)                            # precompute uv 
    d = sdf(body,x,t,uv)                             # use here
    V = ForwardDiff.derivative(t->body.surf(uv,t),t) # here too!
    return (d,n,V)
end

function sdf(body::ParametricBody,x::SVector,t,uv=body.locate(x,t))
    p = x-body.surf(uv,t)
    return copysign(√(p'*p),norm_dir(body.surf,uv,t)'*p)
end

function norm_dir(surf,uv::Number,t)
    s = ForwardDiff.derivative(uv->surf(uv,t),uv)
    return SA[s[2],-s[1]]
end

update!(body::ParametricBody{F,L},t) where {F<:Function,L<:HashedLocator} = 
    update!(body.locate,body.surf,t)

export ParametricBody,measure,sdf,update!

end
