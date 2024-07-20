module ParametricBodies

using StaticArrays,ForwardDiff
import WaterLily: AbstractBody,measure,sdf,interp
import WaterLily
# Force loc to return Float32 SVector by default
WaterLily.loc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(I.I .- 1.5 .- 0.5 .* δ(i,I).I)

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
struct ParametricBody{T,L<:Function,S<:Function,dS<:Function,M<:Function,D<:Function} <: AbstractParametricBody
    surf::S    #ξ = surf(uv,t)
    dotS::dS   #dξ/dt
    locate::L  #uv = locate(ξ,t)
    map::M     #ξ = map(x,t)
    scale::T   #|dx/dξ| = scale
    dis::D     #d = dis(p,n)
end
# Default functions
import LinearAlgebra: det
dmap(x,t) = x; ddis(p,n) = p'*n
get_dotS(surf) = (uv,t)->ForwardDiff.derivative(t->surf(uv,t),t)
get_scale(map,x::SVector{D,T}) where {D,T} = (dξdx=ForwardDiff.jacobian(x->map(x,zero(T)),x); T(abs(det(dξdx))^(-1/D)))
ParametricBody(surf,locate;map=dmap,dis=ddis,x₀=SA_F32[0,0],dotS=get_dotS(surf)) = ParametricBody(surf,dotS,locate,map,get_scale(map,x₀),dis)

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
    return (body.scale*body.dis(p,n),n,body.dotS(uv,t))
end
notC¹(::Function,uv) = false

function norm_dir(surf,uv::Number,t)
    s = ForwardDiff.derivative(uv->surf(uv,t),uv)
    return SA[s[2],-s[1]]
end

# include("integrals.jl")

export AbstractParametricBody,ParametricBody

abstract type AbstractLocator <:Function end
export AbstractLocator

include("HashedLocators.jl")
export HashedBody, HashedLocator, refine, mymod, update!

include("NurbsCurves.jl")
export NurbsCurve,BSplineCurve,interpNurbs

include("NurbsLocator.jl")
export NurbsLocator,DynamicNurbsBody, update!

include("PlanarBodies.jl")
export PlanarBody
# include("Recipes.jl")
# export f

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
