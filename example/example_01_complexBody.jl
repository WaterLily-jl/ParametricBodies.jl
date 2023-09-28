"""
    BSplineCurve(cps; degree=3)

Define a uniform B-spline curve.
- `cps`: A 2D array representing the control points of the B-spline curve
- `degree`: The degree of the B-spline curve
"""
struct BSplineCurve{d,A<:AbstractArray,V<:AbstractVector} <: Function
    pnts::A
    knots::V
end
using StaticArrays
function BSplineCurve(pnts;degree=3)
    count,T = size(pnts, 2),promote_type(eltype(pnts),Float32)
    knots = SA{T}[[zeros(degree); collect(range(0, count-degree) / (count-degree)); ones(degree)]...]
    BSplineCurve{degree,typeof(pnts),typeof(knots)}(pnts,knots)
end
"""
    (::BSplineCurve)(s,t)

Evaluate the spline function
- `s` position along the spline
- `t` time is currently unused but needed for ParametricBodies
"""
function (l::BSplineCurve{d})(u::T,t) where {T,d}
    u==1 && return l.pnts[:,end]
    pt = SA{T}[0, 0]
    for k in 1:size(l.pnts, 2)
        pt += Bd(l.knots,u,k,Val(d))*l.pnts[:,k]
    end
    pt
end
Bd(knots, u, k, ::Val{0}) = 1
function Bd(knots, u, k, ::Val{d}) where d
    B = 0
    knots[k]<u<knots[k+d] && (B+=(u-knots[k])/(knots[k+d]-knots[k])*Bd(knots,u,k,Val(d-1)))
    knots[k+1]≤u<knots[k+d+1] && (B+=(knots[k+d+1]-u)/(knots[k+d+1]-knots[k+1])*Bd(knots,u,k+1,Val(d-1)))
    return B
end

square = BSplineCurve(SA[5 5 0 -5 -5 -5  0  5 5
                         0 5 5  5  0 -5 -5 -5 0],degree=1)
@assert square(0.,0) ≈ [5,0]
@assert square(.5,0) ≈ [-5,0]
@assert square(1.,0) ≈ [5,0]

# Does it work with KernelAbstractions?
using CUDA; @assert CUDA.functional()
using KernelAbstractions
@kernel function _test!(a::AbstractArray,l::BSplineCurve)
    # Map index to physical space
    I = @index(Global)
    s = (I-1)/(length(a)-1)
    q = l(s,0)
    a[I] = q'*q
end
test!(a,l)=_test!(get_backend(a),64)(a,l,ndrange=length(a))
a = CUDA.zeros(64)
test!(a,square)
a|>Array # yes!

# Check bspline is type stable. Note that using Float64 for cps will break this!
@assert isa(square(0.,0),SVector)
@assert all([eltype(square(zero(T),0))==T for T in (Float32,Float64)])

# check winding direction
using LinearAlgebra
cross(a,b) = det([a;;b])
@assert all([cross(square(s,0.),square(s+0.1,0.))>0 for s in range(0,.9,10)])

# check derivatives
using ForwardDiff
dcurve(u) = ForwardDiff.derivative(u->square(u,0),u)
dcurve(0f0)
dcurve(0.5f0)
dcurve(1)

# Wrap the shape function inside the parametric body class and check measurements
using ParametricBodies
body = ParametricBody(square, (0,1));
@assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
@assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]])

# Check that the locator works for closed splines
@assert [body.locate([5,s],0) for s ∈ (-2,-1,-0.1)]≈[0.95,0.975,0.9975]

# Does it work with ParametricBodies on CUDA?
body = ParametricBody(square, (0,1); T=Float32);
CUDA.@allowscalar @assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
CUDA.@allowscalar @assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]]) # yes!