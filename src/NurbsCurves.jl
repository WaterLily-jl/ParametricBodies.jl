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
"""
struct NurbsCurve{d,A<:AbstractArray,V<:AbstractVector,W<:AbstractVector} <: Function
    pnts::A
    knots::V
    wgts::W
end
function NurbsCurve(pnts,knots,weights;degree=3)
    count,T = size(pnts, 2),promote_type(eltype(pnts),Float32)
    @assert count == length(weights) "Invalid NURBS: each control point should have a corresponding weights."
    @assert count < length(knots) "Invalid NURBS: the number of knots should be greater than the number of control points."
    degree = length(knots) - count - 1 # the one in the input is not used
    knots = SA{T}[knots...]; weights = SA{T}[weights...]
    NurbsCurve{degree,typeof(pnts),typeof(knots),typeof(weights)}(copy(pnts),knots,weights)
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
    count,T = size(pnts, 2),promote_type(eltype(pnts),Float32)
    @assert degree <= count - 1 "Invalid B-Spline: the degree should be less than the number of control points minus 1."
    knots = SA{T}[[zeros(degree); collect(range(0, count-degree) / (count-degree)); ones(degree)]...]
    weights = SA{T}[ones(count)...]
    NurbsCurve{degree,typeof(pnts),typeof(knots),typeof(weights)}(copy(pnts),knots,weights)
end

"""
    (::NurbsCurve)(s,t)

Evaluate the NURBS curve
- `s` : A float, representing the position along the spline where we want to compute the value of that NURBS
- `t` time is currently unused but needed for ParametricBodies
"""
function (l::NurbsCurve{d})(u::T,t)::SVector where {T,d}
    pt = SA{T}[0, 0]; wsum=T(0.0)
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
Bd(knots, u, k, ::Val{0}) = Int(knots[k]≤u<knots[k+1] || u==knots[k+1]==1)
function Bd(knots, u, k, ::Val{d}) where d
    ((u-knots[k])/max(eps(Float32),knots[k+d]-knots[k])*Bd(knots,u,k,Val(d-1))
    +(knots[k+d+1]-u)/max(eps(Float32),knots[k+d+1]-knots[k+1])*Bd(knots,u,k+1,Val(d-1)))
end
"""
    NurbsForce(surf::NurbsCurve,p::AbstractArray{T},s,δ=2.0) where T

Compute the normal (Pressure) force on the NurbsCurve curve from a pressure field `p`
at the parametric coordinate `s`. Useful to compuet the force at an integration point
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
        sum(reduce(hcat, [NurbsForce(surf,p,s) for s=0:0.01:1]), dims=2)
"""
    integrate(curve;N=64)

integrate the nurbs curve to give it's length
"""
function integrate(curve::NurbsCurve;N=64)
    # integrate NURBS curve to compute its length
    x, w = gausslegendre(N)
    # map onto the (0,1) interval, need a weight scalling
    uv_ = (x.+1)/2; w/=2 
    sum([norm(derivative(uv->curve(uv,0.),uv))*w[i] for (i,uv) in enumerate(uv_)])
end
"""
    f(C::NurbsCurve, N::Integer=100)

Plot `recipe`` for `NurbsCurve``, plot the `NurbsCurve` and the control points.
"""
@recipe function f(C::NurbsCurve, N::Integer=100; add_cp=true)
    seriestype := :path
    primary := false
    @series begin
        linecolor := :black
        linewidth := 2
        markershape := :none
        c = [C(s,0.0).+0.5 for s ∈ 0:1/N:1]
        getindex.(c,1),getindex.(c,2)
    end
    @series begin
        linewidth  --> (add_cp ? 1 : 0)
        markershape --> (add_cp ? :circle : :none)
        markersize --> (add_cp ? 4 : 0)
        delete!(plotattributes, :add_cp)
        C.pnts[1,:].+0.5,C.pnts[2,:].+0.5
    end
end

# end module
