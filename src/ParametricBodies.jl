module ParametricBodies

using StaticArrays,ForwardDiff
import WaterLily: AbstractBody,measure,sdf,interp

abstract type AbstractParametricBody <: AbstractBody end
"""
    d,n,V = measure(body::AbstractParametricBody,x,t)

Determine the geometric properties of the body at time `t` closest to 
point `x`. Both `dot(surf)` and `dot(map)` contribute to `V` if defined.
"""
function measure(body::AbstractParametricBody,x,t)
    # Surf props and velocity in ξ-frame
    d,n,dotS = surf_props(body,x,t)
    dξdt = dotS-ForwardDiff.derivative(t->body.map(x,t),t)

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

"""
    ParametricBody{T::Real}(surf,locate) <: AbstractBody

    - `surf(u,t)` parametrically defined curve
    - `dotS(u,t)=derivative(t->surf(u,t),t)` time derivative of curve 
    - `locate(ξ,t)` method to find nearest parameter `u` to `ξ`
    - `map(x,t)=x` mapping from `x` to `ξ`
    - `scale=|∇map|⁻¹` distance scaling from `ξ` to `x`
    - `dis(p,n)=p'*n` distance function of position `p=ξ-S` and normal `n`

Explicitly defines a geometry by an unsteady parametric curve. The curve is currently limited 
to be univariate, and must wind counter-clockwise if closed. The optional `dotS`, `map`, and 
`dis` functions allow for more general geometry embeddings.

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
struct ParametricBody{T,L<:Function,S<:Function,dS<:Function,M<:Function,D<:Function} <: AbstractParametricBody
    surf::S    #ξ = surf(v,t)
    dotS::dS   #dξ/dt
    locate::L  #u = locate(ξ,t)
    map::M     #ξ = map(x,t)
    scale::T   #|dx/dξ| = scale
    dis::D     #d = dis(p,n)
end
# Default functions
import LinearAlgebra: det
dmap(x,t) = x; ddis(p,n) = p'*n
get_dotS(surf) = (u,t)->ForwardDiff.derivative(t->surf(u,t),t)
get_scale(map,x::SVector{D,T}) where {D,T} = (dξdx=ForwardDiff.jacobian(x->map(x,zero(T)),x); T(abs(det(dξdx))^(-1/D)))
ParametricBody(surf,locate;map=dmap,dis=ddis,x₀=SA_F32[0,0],dotS=get_dotS(surf)) = ParametricBody(surf,dotS,locate,map,get_scale(map,x₀),dis)

function surf_props(body::ParametricBody,x,t)
    # Map x to ξ and locate nearest u
    ξ = body.map(x,t)
    u = body.locate(ξ,t)

    # Get normal direction and vector from surf to ξ
    n = norm_dir(body.surf,u,t)
    p = ξ-body.surf(u,t)

    # Fix direction for C⁰ points, normalize, and get distance
    notC¹(body.locate,u) && p'*p>0 && (n = p)
    n /=  √(n'*n)
    return (body.scale*body.dis(p,n),n,body.dotS(u,t))
end
notC¹(::Function,u) = false

function norm_dir(surf,u::Number,t)
    s = ForwardDiff.derivative(u->surf(u,t),u)
    return SA[s[2],-s[1]]
end

export AbstractParametricBody,ParametricBody,sdf,measure

abstract type AbstractLocator <:Function end
export AbstractLocator

include("HashedLocators.jl")
export HashedBody, HashedLocator, refine, mymod, update!

include("NurbsCurves.jl")
export NurbsCurve,BSplineCurve,interpNurbs

include("NurbsLocator.jl")
export NurbsLocator,DynamicNurbsBody,update

include("PlanarBodies.jl")
export PlanarBody

include("Recipes.jl")
export f
include("integrals.jl")

# Backward compatibility for extensions
if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/ParametricBodiesAMDGPUExt.jl")
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/ParametricBodiesCUDAExt.jl")
    end
end

end
