module ParametricBodies

using StaticArrays,ForwardDiff
using CUDA,Adapt,KernelAbstractions

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
    x,t = SVector{2,T}(0,0),T(0); ξ = map(x,t)
    @CUDA.allowscalar uv = locate(ξ,t); p = ξ-surf(uv,t)
    @assert isa(ξ,SVector{2,T}) "map is not type stable"
    @assert isa(uv,T) "locate is not type stable"
    @assert isa(p,SVector{2,T}) "surf is not type stable"
  
    ParametricBody(surf,locate,map,T(get_scale(map,x)))
end

import LinearAlgebra: det
get_scale(map,x::SVector{D},t=0.) where D = (dξdx=ForwardDiff.jacobian(x->map(x,t),x); abs(det(dξdx))^(-1/D))

function measure(body::ParametricBody,x,t)
    # Surf props and velocity in ξ-frame
    d,n,uv = surf_props(body,x,t)
    dξdt = ForwardDiff.derivative(t->body.surf(uv,t)-body.map(x,t),t)

    # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
    dξdx = ForwardDiff.jacobian(x->body.map(x,t),x)
    return (d,dξdx\n/body.scale,dξdx\dξdt)
end
sdf(body::ParametricBody,x,t) = surf_props(body,x,t)[1]

function surf_props(body::ParametricBody,x,t)
    # Map to ξ and locate nearest uv
    ξ = body.map(x,t)
    uv = body.locate(ξ,t)

    # Get normal direction and vector from surf to ξ
    n = norm_dir(body.surf,uv,t)
    p = ξ-body.surf(uv,t)

    # Fix direction for C⁰ points, normalize, and get distance
    notC¹(body.locate,uv) && (n = p)
    n /=  √(n'*n)
    return (body.scale*n'*p,n,uv)
end
notC¹(::Function,uv) = false

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

export ParametricBody,measure,sdf

end
