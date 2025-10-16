module ParametricBodies

using StaticArrays,ForwardDiff
# Non-allocating method for 2x3 SMatrix solve
using LinearAlgebra
import Base: \
@inline function (\)(a::SMatrix{2,3}, b::SVector{2})
    # columns of a'
    a1,a2 = a[1,:],a[2,:]

    # Q,R decomposition
    r11 = norm(a1)
    q1 = a1/r11
    r12 = q1'*a2
    p = a2-r12*q1
    r22 = norm(p) < eps(r11) ? one(r11) : norm(p)
    q2 = p/r22

    # forward substitution to solve v = R'\b
    v1 = b[1]/r11
    v2 = (b[2]-r12*v1)/r22

    # return solution x = Qv
    return q1*v1+q2*v2
end

import WaterLily: AbstractBody,measure,sdf,interp

abstract type AbstractParametricBody <: AbstractBody end
"""
    d,n,V = measure(body::AbstractParametricBody,x,t)

Determine the geometric properties of the body at time `t` closest to
point `x`. Both `dot(curve)` and `dot(map)` contribute to `V` if defined.
"""
function measure(body::AbstractParametricBody,x,t;fastd²=Inf)
    # curve props and velocity in ξ-frame
    d,n,dotS = curve_props(body,x,t;fastd²)
    d^2 > fastd² && return d,zero(x),zero(x)
    dξdt = dotS-ForwardDiff.derivative(t->body.map(x,t),t)

    # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
    dξdx = ForwardDiff.jacobian(x->body.map(x,t),x)
    return (d,dξdx\n/body.scale,dξdx\dξdt)
end
"""
    d = sdf(body::AbstractParametricBody,x,t)

Signed distance from `x` to closest point on `body.curve` at time `t`. Sign depends on the
winding direction of the parametric curve.
"""
sdf(body::AbstractParametricBody,x,t;kwargs...) = curve_props(body,x,t;kwargs...)[1]

"""
    ParametricBody{T::Real}(curve,locate) <: AbstractBody

    - `curve(u,t)` parametrically defined curve
    - `dotS(u,t)=derivative(t->curve(u,t),t)` time derivative of curve
    - `locate(ξ,t)` method to find nearest parameter `u` to `ξ`
    - `map(x,t)=x` mapping from `x` to `ξ`
    - `thk=0` thickness offset for the signed distance
    - `boundary=true` if the curve represent a body boundary, not a space-curve

Explicitly defines a geometry by an unsteady parametric curve. The curve is currently limited
to be univariate, and must wind counter-clockwise if closed. The optional `dotS`, `map`,
`thk` and `boundary` parameters allow for more general geometry embeddings.

Example:

    curve(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locate(x::SVector{2},t) = atan(x[2],x[1])-t
    body = ParametricBody(curve,locate)

    @test body.curve(body.locate(SA[4.,3.],1),1) == SA[4/5,3/5]

    d,n,V = measure(body,SA[-.75,1],4.)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5]
    @test V ≈ SA[-4/5,-3/5]
"""
struct ParametricBody{T,L<:Function,S<:Function,dS<:Function,M<:Function,dT<:Function} <: AbstractParametricBody
    curve::S    #ξ = curve(v,t)
    dotS::dS    #dξ/dt
    locate::L   #u = locate(ξ,t)
    map::M      #ξ = map(x,t)
    scale::T    #|dx/dξ| = scale
    half_thk::dT #half thickness
    boundary::Bool
end
# Default functions
import LinearAlgebra: det
dmap(x,t) = x
get_dotS(curve) = (u,t)->ForwardDiff.derivative(t->curve(u,t),t)
x_hat(ndims) = SVector(ntuple(i->√inv(ndims),ndims))
get_scale(map,x,t=0) = norm(ForwardDiff.jacobian(x->map(x,t),x)\x_hat(length(map(x,t))))
ParametricBody(curve,locate;dotS=get_dotS(curve),thk=(u)->0f0,boundary=true,map=dmap,ndims=2,x₀=x_hat(ndims),
               scale=get_scale(map,x₀),T=Float32,kwargs...) = ParametricBody(curve,dotS,locate,map,T(scale),make_func(thk),boundary)
make_func(a::Function) = (s)->a(s)/2
make_func(a::Number) = (s)->a/2
function curve_props(body::ParametricBody,x,t;fastd²=Inf)
    # Map x to ξ and do fast bounding box check
    ξ = body.map(x,t)
    # if isfinite(fastd²) && applicable(body.locate,ξ,t,true)
    #     # d = body.scale*body.locate(ξ,t,true)-maximum(body.half_thk.(range(lims(body)...,10)))
    #     d = body.scale*body.locate(ξ,t,true)-body.half_thk(ξ)
    #     d^2>fastd² && return d,zero(x),zero(x)
    # end

    # Locate nearest u, and get vector
    u = body.locate(ξ,t)
    p = ξ-body.curve(u,t)

    # Get unit normal
    n = notC¹(body.locate,u) ? hat(p) : (s=tangent(body.curve,u,t); body.boundary ? perp(s) : align(p,s))

    # Get scaled & thinkess adjusted distance and dot(S)
    return (body.scale*p'*n-body.half_thk(u),n,body.dotS(u,t))
end
notC¹(::Function,u) = false

hat(p) = p/√(eps(eltype(p))+p'*p)
tangent(curve,u,t) = hat(ForwardDiff.derivative(u->curve(u,t),u))
align(p,s) = hat(p-(p'*s)*s)
perp(s::SVector{2}) = SA[s[2],-s[1]]
perp(s) = s # should never be used!

export AbstractParametricBody,ParametricBody,sdf,measure

abstract type AbstractLocator <:Function end
export AbstractLocator

include("HashedLocators.jl")
export HashedBody, HashedLocator, refine, mymod, update!

include("NurbsCurves.jl")
export NurbsCurve,BSplineCurve,interpNurbs

include("NurbsLocator.jl")
export NurbsLocator,davidon,DynamicNurbsBody

include("PlanarBodies.jl")
export PlanarBody

include("Recipes.jl")
export f

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