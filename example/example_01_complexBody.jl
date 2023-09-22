function coxDeBoor(knots, u, k, d, count)
    """
        coxDeBoor(knots, u, k, d, count)

    Compute the Cox-De Boor recursion for B-spline basis functions.

    The `coxDeBoor` function computes the Cox-De Boor recursion for B-spline basis functions,
    used in the evaluation of B-spline curves and surfaces.

    Arguments:
    - `knots`: An array of knot values.
    - `u`: The parameter value at which to evaluate the B-spline basis function.
    - `k`: The index of the current knot interval.
    - `d`: The degree of the B-spline basis function.
    - `count`: The number of control points.

    Returns:
    The value of the B-spline basis function at parameter `u` and knot interval `k`.
    """
    if (d == 0)
        return Int(((knots[k+1] <= u) && (u < knots[k+2])) || ((u >= (1.0-1e-12)) && (k == (count-1))))
    end
    return (((u-knots[k+1])/max(√eps(u), knots[k+d+1]-knots[k+1]))*coxDeBoor(knots, u, k, (d-1), count)
        + ((knots[k+d+2]-u)/max(√eps(u), knots[k+d+2]-knots[k+2]))*coxDeBoor(knots, u, (k+1), (d-1), count))
end

"""
    BSplineCurve(cps; degree=3, mem=Array)

Define a B-spline curve.
- `cps`: A 2D array representing the control points of the B-spline curve
- `degree`: The degree of the B-spline curve
- `mem`: Array memory type
"""
struct BSplineCurve{A<:AbstractArray,V<:AbstractVector} <: Function
    cps::A
    knots::V
    count::Int
    degree::Int
end
function BSplineCurve(cps;degree=3) 
    count,T = size(cps, 2),promote_type(eltype(cps),Float32)
    knots = [zeros(T,degree); collect(T,range(0, count-degree) / (count-degree)); ones(T,degree)]
    BSplineCurve(cps,SA[knots...],count,degree)
end
function (l::BSplineCurve)(s,t) 
    ```
    Evaluate the spline function
    - `s` position along the spline
    - `t` time is currently unused but needed for ParametricBodies
    ```
    pt = zero(l.cps[:,1])
    for k in range(0, l.count-1)
        pt += coxDeBoor(l.knots, s, k, l.degree, l.count) * l.cps[:, k+1]
    end
    return pt
end
using Adapt
Adapt.adapt_structure(to, x::BSplineCurve) = BSplineCurve(adapt(to,x.cps),adapt(to,x.knots),x.count,x.degree)

# Define square using degree=1 BSpline.
using StaticArrays
cps = SA[5 5 0 -5 -5 -5  0  5 5
         0 5 5  5  0 -5 -5 -5 0]
square = BSplineCurve(cps,degree=1)

# Check bspline is type stable. Note that using Float64 for cps will break this!
@assert isa(square(0.,0),SVector)
@assert all([eltype(square(zero(T),0))==T for T in (Float32,Float64)])

# Create curve and heck winding direction
using LinearAlgebra
cross(a,b) = det([a;;b])
@assert all([cross(square(s,0.),square(s+0.1,0.))>0 for s in range(0,.9,10)])

# Wrap the shape function inside the parametric body class and check measurements
using ParametricBodies
body = ParametricBody(square, (0,1));
@assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
@assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]])

# Check that the locator works for closed splines
@assert [body.locate([5,s],0) for s ∈ (-2,-1,-0.1)]≈[0.95,0.975,0.9975]

# Use mem=CUDA
using CUDA; @assert CUDA.functional()
body = ParametricBody(square, (0,1); T=Float32, mem=CUDA.CuArray) # doesn't work.
# @assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
# @assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]])