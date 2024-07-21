using FastGaussQuadrature: gausslegendre
"""
    integrate(f(uv),curve;N=64)

integrate a function f(uv) along the curve
"""
function _gausslegendre(N,T)
    x,w = gausslegendre(N)
    convert.(T,x),convert.(T,w)
end
integrate(curve::Function,lims=(0.,1.)) = integrate(ξ->1.0,curve,0.0,lims;N=N)
function integrate(f::Function,curve::Function,t,lims::NTuple{2,T};N=64) where T
    # integrate NURBS curve to compute integral
    uv_, w_ = _gausslegendre(N,T)
    # map onto the (uv) interval, need a weight scalling
    scale=(lims[2]-lims[1])/2; uv_=scale*(uv_.+1); w_=scale*w_ 
    sum([f(uv)*norm(ForwardDiff.derivative(uv->curve(uv,t),uv))*w for (uv,w) in zip(uv_,w_)])
end
"""
    ∮nds(p,body::AbstractParametricBody,t=0)

Surface normal pressure integral along the parametric curve(s)
"""
function ∮nds(p::AbstractArray{T},body::ParametricBody,t=0;N=64) where T
    curve(ξ,τ) = -body.map(-body.surf(ξ,τ),τ)
    open = !all(body.surf(body.locate.lims[1],t).≈body.surf(body.locate.lims[2],t))
    integrate(s->_pforce(curve,p,s,t,Val{open}()),curve,t,body.locate.lims;N)
end
"""
    ∮τnds(u,body::AbstractParametricBody,t=0)

Surface normal pressure integral along the parametric curve(s)
"""
function ∮τnds(u::AbstractArray{T},body::ParametricBody,t=0;N=64) where T
    curve(ξ,τ) = -body.map(-body.surf(ξ,τ),τ) # inverse maping
    vel(ξ) = ForwardDiff.derivative(t->curve(ξ,t),t) # get velocity at coordinate ξ
    open = !all(body.surf(body.locate.lims[1],t).≈body.surf(body.locate.lims[2],t))
    integrate(s->_vforce(curve,u,s,t,vel(s),Val{open}()),curve,t,body.locate.lims;N)
end

# pressure force on a parametric `surf` (closed) at parametric coordinate `s` and time `t`.
function _pforce(surf,p::AbstractArray{T},s::T,t,::Val{false},δ=1) where T
    xᵢ = surf(s,t); nᵢ = norm_dir(surf,s,t); nᵢ /= √(nᵢ'*nᵢ)
    return interp(xᵢ+δ*nᵢ,p).*nᵢ
end
function _pforce(surf,p::AbstractArray{T},s::T,t,::Val{true},δ=1) where T
    xᵢ = surf(s,t); nᵢ = ParametricBodies.norm_dir(surf,s,t); nᵢ /= √(nᵢ'*nᵢ)
    return (interp(xᵢ+δ*nᵢ,p)-interp(xᵢ-δ*nᵢ,p))*nᵢ
end
# viscous force on a parametric `surf` (closed) at parametric coordinate `s` and time `t`.
function _vforce(surf,u::AbstractArray{T},s::T,t,vᵢ,::Val{false},δ=1) where T
    xᵢ = surf(s,t); nᵢ = norm_dir(surf,s,t); nᵢ /= √(nᵢ'*nᵢ)
    vᵢ = vᵢ .- sum(vᵢ.*nᵢ)*nᵢ # tangential comp
    uᵢ = interp(xᵢ+δ*nᵢ,u)  # prop in the field
    uᵢ = uᵢ .- sum(uᵢ.*nᵢ)*nᵢ # tangential comp
    return (uᵢ.-vᵢ)./δ # FD
end
function _vforce(surf,u::AbstractArray{T,N},s::T,t,vᵢ,::Val{true},δ=1) where {T,N}
    xᵢ = surf(s,t); nᵢ = ParametricBodies.norm_dir(surf,s,t); nᵢ /= √(nᵢ'*nᵢ)
    τ = zeros(SVector{N-1,T})
    vᵢ = vᵢ .- sum(vᵢ.*nᵢ)*nᵢ
    for j ∈ [-1,1]
        uᵢ = interp(xᵢ+j*δ*nᵢ,u)
        uᵢ = uᵢ .- sum(uᵢ.*nᵢ)*nᵢ
        τ = τ + (uᵢ.-vᵢ)./δ
    end
    return τ
end

# using LinearAlgebra: dot
# function norm_dir(nurbs::NurbsCurve{3},uv::Number,p::SVector{3},t)
#     s = ForwardDiff.derivative(uv->nurbs(uv,t),uv); s/=√(s'*s)
#     return p-dot(p,s)*s
# end

# """
#     ∮nds(p,body::DynamicBody,t=0)

# Surface normal pressure integral along the parametric curve(s)
# """
# function ∮nds(p::AbstractArray{T},body::DynamicBody,t=0;N=64) where T
#     open = !all(body.surf(body.locate.lims[1],t).≈body.surf(body.locate.lims[2],t))
#     integrate(s->_pforce(body.surf,p,s,t,Val{open}()),body.surf,t,body.locate.lims;N)
# end
# """
#     ∮τnds(u,body::DynamicBody,t=0)

# Surface normal pressure integral along the parametric curve(s)
# """
# function ∮τnds(u::AbstractArray{T},body::DynamicBody,t=0;N=64) where T
#     open = !all(body.surf(body.locate.lims[1],t).≈body.surf(body.locate.lims[2],t))
#     integrate(s->_vforce(body.surf,u,s,t,body.velocity(s,0),Val{open}()),body.surf,t,body.locate.lims;N)
# end