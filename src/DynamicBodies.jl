"""
    DynamicBody{T::Real}(surf,locate,scale=|∇map|⁻¹) <: AbstractParametricBody

    - `surf(uv::T,t::T)::SVector{2,T}:` parametrically define curve
    - `locate(ξ::SVector{2,T},t::T):` method to find nearest parameter `uv` to `ξ`
    - `scale::T`: distance scaling from `ξ` to `x`.

Explicitly defines a geometries by an unsteady parametric curve. The curve is currently limited 
to be 2D, and must wind counter-clockwise. Any distance scaling induced by the map needs to be 
uniform and `scale` is computed automatically unless supplied.
"""
struct DynamicBody{T,S<:Function,L<:Union{Function,NurbsLocator}} <: AbstractParametricBody
    surf::S    #ξ = surf(uv,t)
    locate::L  #uv = locate(ξ,t)
    scale::T   #|dx/dξ| = scale
end
function DynamicBody(surf,locate;T=Float64)
    # Check input functions
    x,t = SVector{2,T}(0,0),T(0);
    @CUDA.allowscalar uv = locate(x,t); p = x-surf(uv,t)
    @assert isa(uv,T) "locate is not type stable"
    @assert isa(p,SVector{2,T}) "surf is not type stable"
  
    DynamicBody(surf,locate,T(1.0))
end

"""
    d,n,V = measure(body::DynamicBody,x,t)

Determine the geometric properties of body.surf at time `t` closest to 
point `x`. Only `dot(surf)` contributes to `V`.
"""
function measure(body::DynamicBody,x,t)
    # Surf props and velocity in ξ-frame
    d,n,uv = surf_props(body,x,t)
  
    return (d,n,zero(n)) #(d,dξdx\n/body.scale,dξdx\dξdt)
end

function surf_props(body::DynamicBody,ξ,t)
    # locate nearest uv
    uv = body.locate(ξ,t)

    # Get normal direction and vector from surf to ξ
    n = norm_dir(body.surf,uv,t)
    p = ξ-body.surf(uv,t)

    # Fix direction for C⁰ points, normalize, and get distance
    notC¹(body.locate,uv) && p'*p>0 && (n = p)
    n /=  √(n'*n)
    return (dis(p,n),n,uv)
end

Adapt.adapt_structure(to, x::DynamicBody{T,F,L}) where {T,F,L<:NurbsLocator} =
    DynamicBody(x.surf,adapt(to,x.locate))
"""
    DynamicBody(surf,uv_bounds;step,t⁰,T,mem,map) <: AbstractBody

Creates a `DynamicBody` with `locate=NurbsLocator(surf,uv_bounds...)`.
"""
DynamicBody(surf,uv_bounds::Tuple;step=1,t⁰=0.,T=Float64,mem=Array) = 
    adapt(mem,DynamicBody(surf,NurbsLocator(surf,uv_bounds;step,t⁰,T,mem),T))

update!(body::DynamicBody{T,F,L},t) where {T,F,L<:NurbsLocator} = 
    update!(body.locate,body.surf,t)