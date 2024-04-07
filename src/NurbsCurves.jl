using StaticArrays
using ForwardDiff: derivative
using FastGaussQuadrature: gausslegendre
using LinearAlgebra: norm
using RecipesBase: @recipe, @series

"""
    NurbsCurve(pnts,knos,weights)

Define a non-uniform rational B-spline curve.
- `pnts`: A 2D array representing the control points of the NURBS curve
- `knots`: A 1D array of th knot vector of the NURBS curve
- `wgts`: A 1D array of the wight of the pnts of the NURBS curve 
- `d`: The degree of the NURBS curve
- `n`: the spacial dimension of the NURBS curve, n ∈ {2,3}
- `O`: Open or closed NURBS curve
"""
struct NurbsCurve{n,d,O,A<:AbstractArray,V<:AbstractVector,W<:AbstractVector} <: Function
    pnts::A
    knots::V
    wgts::W
end
function NurbsCurve(pnts,knots,weights)
    (dim,count),T = size(pnts),promote_type(eltype(pnts),Float32)
    @assert count == length(weights) "Invalid NURBS: each control point should have a corresponding weights."
    @assert count < length(knots) "Invalid NURBS: the number of knots should be greater than the number of control points."
    degree = length(knots) - count - 1 # the one in the input is not used
    knots = SA{T}[knots...]; weights = SA{T}[weights...]
    open = !all(pnts[:,1].≈pnts[:,end]) ? Open : Closed
    NurbsCurve{dim,degree,open,typeof(pnts),typeof(knots),typeof(weights)}(pnts,knots,weights)
end
Base.copy(n::NurbsCurve) = NurbsCurve(copy(n.pnts),copy(n.knots),copy(n.wgts))

"""
    BSplineCurve(pnts; degree=3)

Define a uniform B-spline curve.
- `pnts`: A 2D array representing the control points of the B-spline curve
- `degree`: The degree of the B-spline curve
Note: An open, uniform knot vector for a degree `degree` B-spline is constructed by default.
"""
function BSplineCurve(pnts;degree=1)
    (dim,count),T = size(pnts),promote_type(eltype(pnts),Float32)
    @assert degree <= count - 1 "Invalid B-Spline: the degree should be less than the number of control points minus 1."
    knots = SA{T}[[zeros(degree); collect(range(0, count-degree) / (count-degree)); ones(degree)]...]
    weights = SA{T}[ones(count)...]
    open = !all(pnts[:,1].≈pnts[:,end]) ? Open : Closed
    NurbsCurve{dim,degree,open,typeof(pnts),typeof(knots),typeof(weights)}(pnts,knots,weights)
end

"""
    (::NurbsCurve)(s,t)

Evaluate the NURBS curve
- `s` : A float, representing the position along the spline where we want to compute the value of that NURBS
- `t` time is currently unused but needed for ParametricBodies
"""
function (l::NurbsCurve{n,d})(u::T,t)::SVector where {T,d,n}
    pt = zeros(SVector{n,T}); wsum=T(0.0)
    for k in 1:size(l.pnts, 2)
        l.knots[k]>u && break
        l.knots[k+d+1]≥u && (prod = Bd(l.knots,u,k,Val(d))*l.wgts[k];
                             pt +=prod*l.pnts[:,k]; wsum+=prod)
    end
    pt/wsum
end

"""
    Bd(knot, u, k, ::Val{d}) where d

Compute the Cox-De Boor recursion for B-spline basis functions.
- `knot`: A Vector containing the knots of the B-Spline, with the knot value `k ∈ [0,1]`.
- `u` : A Float representing the value of the parameter on the curve at which the basis function is computed, `u ∈ [0,1]`
- `k` : An Integer representing which basis function is computed.
- `d`: An Integer representing the order of the basis function to be computed.
"""
Bd(knots, u::T, k, ::Val{0}) where T = Int(knots[k]≤u<knots[k+1] || u==knots[k+1]==1)
function Bd(knots, u::T, k, ::Val{d}) where {T,d}
    ((u-knots[k])/max(eps(T),knots[k+d]-knots[k])*Bd(knots,u,k,Val(d-1))
    +(knots[k+d+1]-u)/max(eps(T),knots[k+d+1]-knots[k+1])*Bd(knots,u,k+1,Val(d-1)))
end
"""
    NurbsForce(surf::NurbsCurve,p::AbstractArray{T},s,δ=2.0) where T

Compute the normal (Pressure) force on the NurbsCurve curve from a pressure field `p`
at the parametric coordinate `s`. Useful to compute the force at an integration point
along the NurbsCurve
"""
function NurbsForce(surf::NurbsCurve,p::AbstractArray{T},s,δ=2.0) where T
    xᵢ = surf(s,0.0)
    δnᵢ = δ*ParametricBodies.norm_dir(surf,s,0.0); δnᵢ/=√(δnᵢ'*δnᵢ)
    Δpₓ = interp(xᵢ+δnᵢ,p)-interp(xᵢ-δnᵢ,p)
    return -Δpₓ.*δnᵢ
end
"""
    NurbsForce(surf::NurbsCurve,p::AbstractArray{T}) where T

Compute the total force acting on a NurbsCurve from a pressure field `p`.
"""
force(surf::NurbsCurve,p::AbstractArray{T}) where {T} = 
        integrate(s->NurbsForce(surf,p,s),surf;N=64)
"""
    integrate(f(uv),curve;N=64)

"""
    interpNurbs(pnts{D,n};p=n-1)

Given a `SMatrix{D ∈ [2,3],n}` of points, fits a `NurbsCurve{D,n}` of degree `p` to 
this set of points. By default the highest degrees NURBS is constructed.
"""
function interpNurbs(pnts::SMatrix{D,n,T};p=n-1) where {D,n,T}
    @assert p <= n - 1 "Invalid interpolation: the degree should be less than the number of control points minus 1."    
    # construct the parameter and the knot vector
    s = _u(pnts)
    knot = SA[[zeros(p+1); [sum(s[j:j+p-1])/p for j ∈ p-1:n-p]; ones(p+1)]...]
    # construct system and solve
    A = zeros(n,n);
    for i ∈ 1:n, k ∈ 1:n
        A[i,k] = ifelse(abs(i-k)≥p,0.0,ParametricBodies.Bd(knot,s[i],k,Val(p)))
    end
    cpns = SMatrix{2,n}((A\pnts')') # bit ugly, but it works
    # build NurbsCurve and return it
    NurbsCurve(cpns,knot,ones(size(cpns,2)))
end
function _u(pnts::SMatrix{D,n,T}) where {D,n,T}
    d = sum([norm(pnts[:,k]-pnts[:,k-1]) for k ∈ 2:n])
    vcat(zero(T),cumsum([norm(pnts[:,k]-pnts[:,k-1])/d for k ∈ 2:n]))
end