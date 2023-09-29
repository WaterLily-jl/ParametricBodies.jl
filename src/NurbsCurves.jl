using StaticArrays

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
    NurbsCurve{degree,typeof(pnts),typeof(knots),typeof(weights)}(pnts,knots,weights)
end
"""
    BSplineCurve(pnts; degree=3)

Define a uniform B-spline curve.
- `pnts`: A 2D array representing the control points of the B-spline curve
- `degree`: The degree of the B-spline curve
Note: An open, unifirm knot vector for a degree `degree` B-spline is constructed by default.
"""
function BSplineCurve(pnts;degree=3)
    count,T = size(pnts, 2),promote_type(eltype(pnts),Float32)
    knots = SA{T}[[zeros(degree); collect(range(0, count-degree) / (count-degree)); ones(degree)]...]
    weights = SA{T}[ones(count)...]
    NurbsCurve{degree,typeof(pnts),typeof(knots),typeof(weights)}(pnts,knots,weights)
end
"""
    (::NurbsCurve)(s,t)

Evaluate the NURBS curve
- `s` position along the spline
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
    Compute the Cox-De Boor recursion for B-spline basis functions.
"""
Bd(knots, u, k, ::Val{0}) = Int(knots[k]≤u<knots[k+1] || u==knots[k+1]==1)
function Bd(knots, u, k, ::Val{d}) where d
    ((u-knots[k])/max(eps(Float32),knots[k+d]-knots[k])*Bd(knots,u,k,Val(d-1))
    +(knots[k+d+1]-u)/max(eps(Float32),knots[k+d+1]-knots[k+1])*Bd(knots,u,k+1,Val(d-1)))
end