module ParametricBodies

import WaterLily: AbstractBody
using StaticArrays,ForwardDiff

struct ParametricBody <: AbstractBody
    surf    #x = surf(uv,t)
    locate  #uv = locate(surf,x,t)
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

export ParametricBody,measure,sdf

include("HashedLocators.jl")
export HashedLocator

end
