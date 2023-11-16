"""
    DynamicBody{T::Real}(surf,locate,scale=|∇map|⁻¹) <: AbstractParametricBody

    - `surf(uv::T,t::T)::SVector{2,T}:` parametrically define curve
    - `locate(ξ::SVector{2,T},t::T):` method to find nearest parameter `uv` to `ξ`
    - `scale::T`: distance scaling from `ξ` to `x`.

Explicitly defines a geometries by an unsteady parametric curve. The curve is currently limited 
to be 2D, and must wind counter-clockwise. Any distance scaling induced by the map needs to be 
uniform and `scale` is computed automatically unless supplied.
"""
struct DynamicBody{T,S<:Function,L<:Union{Function,NurbsLocator},V<:Function,D<:Function} <: AbstractParametricBody
    surf::S    #ξ = surf(uv,t)
    locate::L  #uv = locate(ξ,t)
    velocity::V # v = velocity(uv), defaults to v=0
    scale::T   #|dx/dξ| = scale
    dist::D
    store::SVector{2,<:AbstractArray}
end
function DynamicBody(surf,locate;dist=dis,T=Float64)
    # Check input functions
    x,t = SVector{2,T}(0,0),T(0);
    @CUDA.allowscalar uv = locate(x,t); p = x-surf(uv,t)
    @assert isa(uv,T) "locate is not type stable"
    @assert isa(p,SVector{2,T}) "surf is not type stable"
    @assert isa(dist(x,x),T) "dist is not type stable"
    dsurf = copy(surf); dsurf.pnts .= 0.0 # zero velocity
    store = SVector{2,typeof(surf.pnts)}(copy(surf.pnts),copy(dsurf.pnts))
    DynamicBody(surf,locate,dsurf,T(1.0),dist,store)
end
"""
    DynamicBody(surf,uv_bounds;step,t⁰,T,mem,map) <: AbstractBody

Creates a `DynamicBody` with `locate=NurbsLocator(surf,uv_bounds...)`.
"""
DynamicBody(surf,uv_bounds::Tuple;dist=dis,step=1,t⁰=0.,T=Float64,mem=Array) =
    adapt(mem,DynamicBody(surf,NurbsLocator(surf,uv_bounds;step,t⁰,T,mem);dist,T))

Adapt.adapt_structure(to, x::DynamicBody{T,F,L,V,D}) where {T,F,L<:NurbsLocator,V,D} =
                      DynamicBody(x.surf,adapt(to,x.locate),x.velocity,x.scale,x.dist,x.store)
"""
    d,n,V = measure(body::DynamicBody,x,t)

Determine the geometric properties of body.surf at time `t` closest to 
point `x`. Only `dot(surf)` contributes to `V`.
"""
function measure(body::DynamicBody,x,t)
    # Surf props and velocity in ξ-frame
    d,n,uv = surf_props(body,x,t)
    
    # for now zero velocity
    v = body.velocity(uv,t)
    return (d,n,v) #(d,dξdx\n/body.scale,dξdx\dξdt)
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
    return (body.dist(p,n),n,uv)
end

"""
    update
"""
update!(body::DynamicBody{T,F,L,V},t) where {T,F,L<:NurbsLocator,V} = 
    update!(body.locate,body.surf,t)
function update!(body::DynamicBody{T,F,L,V},uⁿ::AbstractArray,Δt) where {T,F,L<:NurbsLocator,V}
    body.velocity.pnts .= (uⁿ.-copy(body.surf.pnts))./Δt
    body.surf.pnts .= uⁿ
    update!(body.locate,body.surf,0.0)
end
function update!(body::DynamicBody{T,F,L,V},uⁿ::AbstractArray,vⁿ::AbstractArray) where {T,F,L<:NurbsLocator,V}
    body.surf.pnts .= uⁿ
    body.velocity.pnts .= vⁿ
    update!(body.locate,body.surf,0.0)
end
# store and revert methods for FSI
function store!(body::DynamicBody)
    body.store[1] .= body.surf.pnts
    body.store[2] .= body.velocity.pnts
end
function revert!(body::DynamicBody)
    body.surf.pnts .= body.store[1]
    body.velocity.pnts .= body.store[2]
end
