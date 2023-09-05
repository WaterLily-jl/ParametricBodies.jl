using WaterLily,StaticArrays
using ParametricBodies # New package
using Plots

# Define spline control points. Something similar to a foil.
cps = hcat(
    [1.0, 0.0],
    [0.5, 0.1],
    [0.0, 0.0],
    [0.5, -0.1],
    [1.0, 0.0]
)

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
    return (((u-knots[k+1])/max(1e-12, knots[k+d+1]-knots[k+1]))*coxDeBoor(knots, u, k, (d-1), count)
        + ((knots[k+d+2]-u)/max(1e-12, knots[k+d+2]-knots[k+2]))*coxDeBoor(knots, u, (k+1), (d-1), count))
end

function bspline(cv, s; d=3)
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
    knots = vcat(zeros(d), range(0, count-d) / (count-d), ones(d))
    pt = zeros(size(cv, 1))
    for k in range(0, count-1)
        pt += coxDeBoor(knots, s, k, d, count) * cv[:, k+1]
    end
    return pt
end

function evaluate_spline(cps, s)
    """
        evaluate_spline(cps, s)

    Evaluate a B-spline curve at multiple parameter values.

    The `evaluate_spline` function evaluates a B-spline curve at the specified parameter values `s`.

    Arguments:
    - `cps`: A 2D array representing the control points of the B-spline curve.
    - `s`: An array of parameter values at which the B-spline curve should be evaluated.

    Returns:
    A 2D array where each column corresponds to a point on the B-spline curve at the parameter values in `s`.

    Note:
    - This function assumes a column-major orientation of points as Julia gods intended.
    """
    return hcat([bspline(cps, u, d=2) for u in s]...)
end

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

using CUDA; @assert CUDA.functional()

# Plot the body shape before doing anything else.
s = 0:0.01:1
xy = evaluate_spline(cps, s)
Plots.plot(xy[1, :], xy[2, :], label="", color=:red, lw=2,
    aspect_ratio=:equal, xlabel="x", ylabel="y", dpi=200, size=(1200, 600))
Plots.plot!(cps[1, :], cps[2, :], marker=:circle, linestyle=:dash, label="", color=:black, alpha=0.5, lw=2)
Plots.savefig("complex_body_shape_test.png")

# Create the simulation and try to make it work.
include("viz.jl");Makie.inline!(false);

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
