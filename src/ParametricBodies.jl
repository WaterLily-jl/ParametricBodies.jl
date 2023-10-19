module ParametricBodies

using StaticArrays,ForwardDiff
using CUDA,Adapt,KernelAbstractions

include("HashedLocators.jl")
export HashedLocator, refine, mymod

include("NurbsCurves.jl")
export NurbsCurve, BSplineCurve, f

import WaterLily: AbstractBody,measure,sdf,interp
"""
    ParametricBody{T::Real}(surf,locate,map=(x,t)->x,scale=|∇map|⁻¹) <: AbstractBody

    - `surf(uv::T,t::T)::SVector{2,T}:` parametrically define curve
    - `locate(ξ::SVector{2,T},t::T):` method to find nearest parameter `uv` to `ξ`
    - `map(x::SVector{2,T},t::T)::SVector{2,T}:` mapping from `x` to `ξ`
    - `scale::T`: distance scaling from `ξ` to `x`.

Explicitly defines a geometries by an unsteady parametric curve and optional coordinate `map`.
The curve is currently limited to be 2D, and must wind counter-clockwise. Any distance scaling
induced by the map needs to be uniform and `scale` is computed automatically unless supplied.

Example:

    surf(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locate(x::SVector{2},t) = atan(x[2],x[1])-t
    body = ParametricBody(surf,locate)

    @test body.surf(body.locate(SA[4.,3.],1),1) == SA[4/5,3/5]

    d,n,V = measure(body,SA[-.75,1],4.)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5]
    @test V ≈ SA[-4/5,-3/5]
"""
abstract type AbstractParametricBody <: AbstractBody end
struct ParametricBody{T,S<:Function,L<:Union{Function,HashedLocator},M<:Function} <: AbstractParametricBody
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

"""
    d,n,V = measure(body::ParametricBody,x,t)

Determine the geometric properties of body.surf at time `t` closest to 
point `x`. Both `dot(surf)` and `dot(map)` contribute to `V`.
"""
function measure(body::ParametricBody,x,t)
    # Surf props and velocity in ξ-frame
    d,n,uv = surf_props(body,x,t)
    dξdt = ForwardDiff.derivative(t->body.surf(uv,t)-body.map(x,t),t)

    # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
    dξdx = ForwardDiff.jacobian(x->body.map(x,t),x)
    return (d,dξdx\n/body.scale,dξdx\dξdt)
end
"""
    d = sdf(body::AbstractParametricBody,x,t)

Signed distance from `x` to closest point on `body.surf` at time `t`. Sign depends on the
winding direction of the parametric curve.
"""
sdf(body::AbstractParametricBody,x,t) = surf_props(body,x,t)[1]

function surf_props(body::ParametricBody,x,t)
    # Map x to ξ and locate nearest uv
    ξ = body.map(x,t)
    uv = body.locate(ξ,t)

    # Get normal direction and vector from surf to ξ
    n = norm_dir(body.surf,uv,t)
    p = ξ-body.surf(uv,t)

    # Fix direction for C⁰ points, normalize, and get distance
    notC¹(body.locate,uv) && p'*p>0 && (n = p)
    n /=  √(n'*n)
    return (body.scale*dis(p,n),n,uv)
end
dis(p,n) = n'*p
notC¹(::Function,uv) = false

function norm_dir(surf,uv::Number,t)
    s = ForwardDiff.derivative(uv->surf(uv,t),uv)
    return SA[s[2],-s[1]]
end

Adapt.adapt_structure(to, x::ParametricBody{T,F,L}) where {T,F,L<:HashedLocator} =
    ParametricBody(x.surf,adapt(to,x.locate),x.map,x.scale)

"""
    ParametricBody(surf,uv_bounds;step,t⁰,T,mem,map) <: AbstractBody

Creates a `ParametricBody` with `locate=HashedLocator(surf,uv_bounds...)`.
"""
ParametricBody(surf,uv_bounds::Tuple;step=1,t⁰=0.,T=Float64,mem=Array,map=(x,t)->x) = 
    adapt(mem,ParametricBody(surf,HashedLocator(surf,uv_bounds;step,t⁰,T,mem);map,T))

update!(body::ParametricBody{T,F,L},t) where {T,F,L<:HashedLocator} = 
    update!(body.locate,body.surf,t)

export AbstractParametricBody,ParametricBody,measure,sdf

include("NurbsLocator.jl")
export NurbsLocator

include("DynamicBodies.jl")
export DynamicBody,measure,sdf

end
