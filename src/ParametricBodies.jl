module ParametricBodies

using StaticArrays
import WaterLily: AbstractBody,measure,sdf,interp,derivative,jacobian

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
    dξdt = dotS-derivative(t->body.map(x,t),t)

    # Convert to x-frame with dξ/dx⁻¹ (d has already been scaled)
    dξdx = jacobian(x->body.map(x,t),x)
    return (d,hat(dξdx'n),dξdx\dξdt)
end
"""
    d = sdf(body::AbstractParametricBody,x,t)

Signed distance from `x` to closest point on `body.curve` at time `t`. Sign depends on the
winding direction of the parametric curve.
"""
sdf(body::AbstractParametricBody,x,t;fastd²=0) = curve_props(body,x,t;fastd²)[1]

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
get_dotS(curve) = (u,t)->derivative(t->curve(u,t),t)
x_hat(ndims) = SVector(ntuple(i->√inv(ndims),ndims))
get_scale(map,x,t=0) = norm(jacobian(x->map(x,t),x)\x_hat(length(map(x,t))))
ParametricBody(curve,locate;dotS=get_dotS(curve),thk=(u)->0f0,boundary=true,map=dmap,ndims=2,x₀=x_hat(ndims),
               scale=get_scale(map,x₀),T=Float32,kwargs...) = ParametricBody(curve,dotS,locate,map,T(scale),make_func(thk),boundary)
make_func(a::Function) = (s)->a(s)/2
make_func(a::Number) = (s)->a/2
function curve_props(body::ParametricBody,x,t;fastd²=Inf)
    # Map x to ξ, locate nearest u (quickly if applicable), and get vector
    ξ = body.map(x,t)
    u = applicable(body.locate,ξ,t;fastd²) ? body.locate(ξ,t,fastd²=fastd²/body.scale^2) : body.locate(ξ,t)
    p = ξ-body.curve(u,t)

    # Get outward unit normal
    n = if body.boundary # outward = RHS of the tangent vector
        if C¹(body.locate,u)
            perp(hat(tangent(body.curve,u,t))) # easy peasy
        else # Set n s.t d=n'p even on corners/end-points...
            t₁,t₂ = tangent.(body.curve,eachside(body.locate,u),t) # segment tangents
            s = sum(perp.(hat.((t₁,t₂))))                          # sum of outward normals
            sign(s'p) * hat(p)
        end
    else # outward = towards p
        notC¹(body.locate,u) ? hat(p) : align(p,hat(tangent(body.curve,u,t)))
    end

    # Get scaled & thinkess adjusted distance and dot(S)
    return (body.scale*p'*n-body.half_thk(u),n,body.dotS(u,t))
end
notC¹(::Function,u) = false; C¹(f,u) = !notC¹(f,u)

hat(p) = p/√(eps(eltype(p))+p'*p)
tangent(curve,u,t) = derivative(u->curve(u,t),u)
align(p,s) = hat(p-(p'*s)*s)
perp(s::SVector{2}) = SA[s[2],-s[1]]
perp(s) = s # should never be used!

export AbstractParametricBody,ParametricBody,sdf,measure

abstract type AbstractLocator <:Function end
export AbstractLocator

include("LinAlg.jl")
include("Refine.jl")
export mymod,refine

include("HashedLocators.jl")
export HashedBody, HashedLocator, update!

include("NurbsCurves.jl")
export NurbsCurve,BSplineCurve,interpNurbs

include("NurbsLocator.jl")
export NurbsLocator,DynamicNurbsBody

include("PlanarBodies.jl")
export PlanarBody

include("Recipes.jl")
export f

end