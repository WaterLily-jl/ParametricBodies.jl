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

function bspline(cv, s::T; d=3) where T
    """
        bspline(cv, s; d=3)

    Evaluate a B-spline curve at a given parameter value.

    The `bspline` function evaluates a B-spline curve at the specified parameter `s`.

    Arguments:
    - `cv`: A 2D array representing the control points of the B-spline curve.
    - `s`: The parameter value at which the B-spline curve should be evaluated.
    - `d`: The degree of the B-spline curve (default is 3).

    Returns:
    A vector representing the point on the B-spline curve at parameter `s`.

    Note:
    - This function assumes a column-major orientation of points as Julia gods intended.
    """
    count = size(cv, 2)
    knots = [zeros(T,d); collect(T,range(0, count-d) / (count-d)); ones(T,d)]
    pt = zeros(T,size(cv, 1))
    for k in range(0, count-1)
        pt += coxDeBoor(knots, s, k, d, count) * cv[:, k+1]
    end
    return pt
end

function evaluate_spline(cps, s; d=3)
    """
        evaluate_spline(cps, s; d=3)

    Evaluate a B-spline curve at multiple parameter values.

    The `evaluate_spline` function evaluates a B-spline curve at the specified parameter values `s`.

    Arguments:
    - `cps`: A 2D array representing the control points of the B-spline curve.
    - `s`: An array of parameter values at which the B-spline curve should be evaluated.
    - `d`: The degree of the B-spline curve (default is 3).

    Returns:
    A 2D array where each column corresponds to a point on the B-spline curve at the parameter values in `s`.

    Note:
    - This function assumes a column-major orientation of points as Julia gods intended.
    """
    return hcat([bspline(cps, u, d=d) for u in s]...)
end

# Define spline control points. Square using d=1.
using StaticArrays
cps = SA[5 5 0 -5 -5 -5  0  5 5
         0 5 5  5  0 -5 -5 -5 0]

# Check bspline is type stable. Note that using Float64 for cps will break this!
@assert all([eltype(bspline(cps,zero(T),d=1))==T for T in (Float32,Float64)])

# Create curve and heck winding direction
curve(s,t) = bspline(cps,s,d=1)
using LinearAlgebra
cross(a,b) = det([a;;b])
@assert all([cross(curve(s,0.),curve(s+0.1,0.))>0 for s in range(.9,10)])

# Wrap the shape function inside the parametric body class and check measurements
using ParametricBodies
body = ParametricBody(curve, (0,1))
@assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
@assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]])

# Use mem=CUDA
using CUDA; @assert CUDA.functional()
#body = ParametricBody((θ,t) -> [cos(θ),sin(θ)],(0.,2π);step=0.25,T=Float32,mem=CUDA.CuArray) # does't work because HashedLocator requires SA output. 
body = ParametricBody((θ,t) -> SA[cos(θ),sin(θ)],(0.,2π);step=0.25,T=Float32,mem=CUDA.CuArray) # works
# body = ParametricBody(curve, (0,1); T=Float32, mem=CUDA.CuArray) # doesn't work. maybe need to `adapt` curve before using
# @assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
# @assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]])

2+2
# Plot the contours of signed distance function.
# function square_sdf(x, y, L)
#     # Calculate the distances to the edges of the square
#     dx = abs(x) - L / 2
#     dy = abs(y) - L / 2

#     # Calculate the SDF value based on the distances
#     if dx <= 0 && dy <= 0
#         # Inside the square
#         return max(dx, dy)
#     else
#         # Outside the square
#         return sqrt(max(dx, 0)^2 + max(dy, 0)^2)
#     end
# end

# x = -1.0:0.05:1.0
# y = -1.0:0.045:1.0
# z = zeros(length(x), length(y))
# for i in range(1, length(x))
#     for j in range(1, length(y))
#         z[i, j] = square_sdf(x[i], y[j], 1.0)
#     end
# end
# Plots.contourf(y, x, z, levels=vcat(-0.5:0.1:0.5), color=:bwr)
# Plots.plot!(xy[1, :], xy[2, :], label="", color=:black, lw=2,
#     aspect_ratio=:equal, xlabel="x", ylabel="y", dpi=200, size=(1200, 600))
# Plots.plot!(cps[1, :], cps[2, :], marker=:circle, linestyle=:dash, label="", color=:black, alpha=0.5, lw=2)
# Plots.savefig("complex_body_shape_test_distFunction.png")

# The code below can be used to make an actual simulation using the body once it's been fixed.
#=
# Create the simulation and try to make it work.
include("viz.jl");Makie.inline!(false);

function make_sim(;L=32,Re=1e3,St=0.3,αₘ=-π/18,U=1,n=8,m=4,T=Float32,mem=Array)
    # Position of the body.
    position = SA[L, 0.5f0m*L]

    function map(x,t)
        # move to the desired position
        ξ = x-position
        # Return the transformed coordinate.
        return SA[ξ[1], ξ[2]]
    end

    # Define a function that creates the shape of the body
    function some_shape(s, t)
        # Compute the body coordinate for the given s-parameter value.
        return evaluate_spline(cps, s)[:, 1]
    end

    # Wrap the shape function inside the parametric body class.
    body = ParametricBody(some_shape, (0,1); map, T, mem)

    # Return an initialised simulation instance.
    Simulation((n*L, m*L), (U, 0), L; ν=U*L/Re, body, T, mem)
end

sim = make_sim(mem=CuArray);
fig,viz = body_omega_fig(sim);fig
for _ in 1:200
    sim_step!(sim,sim_time(sim)+0.1)
    update!(viz,sim)
end

function make_video(L=128,St=0.3,name="out.mp4")
    cycle = range(0,2/St,240)
    sim = make_sim(;L,St,mem=CuArray)
    fig,viz = body_omega_fig(sim)
    Makie.record(fig,name,2cycle) do t
        sim_step!(sim,t)
        update!(viz,sim)
    end
end
=#
