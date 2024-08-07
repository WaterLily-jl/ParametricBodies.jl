using FastGaussQuadrature: gausslegendre
function _gausslegendre(N,T)
    x,w = gausslegendre(N)
    convert.(T,x),convert.(T,w)
end
"""
    integrate(f(uv),curve;N=64)

integrate a function f(uv) along the curve
"""
integrate(crv::Function,lims;N=16) = integrate(ξ->1.0,crv,0,lims;N)
function integrate(f::Function,crv::Function,t,lims;N=64)
    @assert length(crv(first(lims),t))==2 "integrate(..) can only be used for 2D curves"
    # integrate NURBS curve to compute integral
    uv_, w_ = _gausslegendre(N,typeof(first(lims)))
    # map onto the (uv) interval, need a weight scalling
    scale=(last(lims)-first(lims))/2; uv_=scale*(uv_.+1); w_=scale*w_ 
    sum([f(uv)*norm(ForwardDiff.derivative(uv->crv(uv,t),uv))*w for (uv,w) in zip(uv_,w_)])
end
import WaterLily: pressure_force,viscous_force,pressure_moment
"""
    pressure_force(p,df,body::AbstractParametricBody,t=0,T;N)

Surface normal pressure integral along the parametric curve(s)
"""
open(b::ParametricBody{T,L};t=0) where {T,L<:NurbsLocator} = Val{(!all(b.curve(first(b.curve.knots),t).≈b.curve(last(b.curve.knots),t)))}()
open(b::ParametricBody{T,L};t=0) where {T,L<:HashedLocator} = Val{(all(b.curve(first(b.locate.lims),t).≈b.curve(last(b.locate.lims),t)))}()
lims(b::ParametricBody{T,L};t=0) where {T,L<:NurbsLocator} = (first(b.curve.knots),last(b.curve.knots))
lims(b::ParametricBody{T,L};t=0) where {T,L<:HashedLocator} = b.locate.lims
function pressure_force(p,df,body::ParametricBody,t=0,T=promote_type(Float64,eltype(p));N=64)
    curve(ξ,τ) = -body.map(-body.curve(ξ,τ),τ) # inverse maping
    -one(T)*integrate(s->_pforce(curve,p,s,t,open(body)),curve,t,lims(body);N)
end
"""
    viscous_force(u,ν,df,body::AbstractParametricBody,t=0,T;N)

Surface normal pressure integral along the parametric curve(s)
"""
function viscous_force(u,ν,df,body::ParametricBody,t=0,T=promote_type(Float64,eltype(u));N=64)
    curve(ξ,τ) = -body.map(-body.curve(ξ,τ),τ) # inverse maping
    ν*integrate(s->_vforce(curve,u,s,t,body.dotS(s,t),open(body)),curve,t,lims(body);N)
end
using LinearAlgebra: cross
"""
    pressure_moment(x₀,p,df,body::AbstractParametricBody,t=0,T;N)

Surface normal pressure moment integral along the parametric curve(s)
"""
function pressure_moment(x₀,p,df,body::ParametricBody,t=0,T=promote_type(Float64,eltype(p));N=64)
    curve(ξ,τ) = -body.map(-body.curve(ξ,τ),τ) # inverse maping
    -one(T)*integrate(s->cross(curve(s,t)-x₀,_pforce(curve,p,s,t,open(body))),curve,t,lims(body);N)
end

perp(curve,u,t) = perp(tangent(curve,u,t))
# pressure force on a parametric `curve` (closed) at parametric coordinate `s` and time `t`.
function _pforce(crv,p::AbstractArray,s,t,::Val{false};δ=1)
    xᵢ = crv(s,t); nᵢ = perp(crv,s,t); nᵢ /= √(nᵢ'*nᵢ)
    return interp(xᵢ+δ*nᵢ,p).*nᵢ
end
function _pforce(crv,p::AbstractArray,s,t,::Val{true};δ=1)
    xᵢ = crv(s,t); nᵢ = ParametricBodies.perp(crv,s,t); nᵢ /= √(nᵢ'*nᵢ)
    return (interp(xᵢ+δ*nᵢ,p)-interp(xᵢ-δ*nᵢ,p))*nᵢ
end
# viscous force on a parametric `curve` (closed) at parametric coordinate `s` and time `t`.
function _vforce(crv,u::AbstractArray,s,t,vᵢ,::Val{false};δ=1)
    xᵢ = crv(s,t); nᵢ = perp(crv,s,t); nᵢ /= √(nᵢ'*nᵢ)
    vᵢ = vᵢ .- sum(vᵢ.*nᵢ)*nᵢ # tangential comp
    uᵢ = interp(xᵢ+δ*nᵢ,u)  # prop in the field
    uᵢ = uᵢ .- sum(uᵢ.*nᵢ)*nᵢ # tangential comp
    return (uᵢ.-vᵢ)./δ # FD
end
function _vforce(crv,u::AbstractArray{T,N},s,t,vᵢ,::Val{true};δ=1) where {T,N}
    xᵢ = crv(s,t); nᵢ = perp(crv,s,t); nᵢ /= √(nᵢ'*nᵢ)
    τ = zeros(SVector{N-1,T})
    vᵢ = vᵢ .- sum(vᵢ.*nᵢ)*nᵢ
    for j ∈ [-1,1]
        uᵢ = interp(xᵢ+j*δ*nᵢ,u)
        uᵢ = uᵢ .- sum(uᵢ.*nᵢ)*nᵢ
        τ = τ + (uᵢ.-vᵢ)./δ
    end
    return τ
end
