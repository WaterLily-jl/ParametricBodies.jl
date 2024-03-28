using WaterLily
using ParametricBodies
using StaticArrays
include("viz.jl")

# parameters
L=2^5
Re=250
U =1

# NURBS points, weights and knot vector for a circle
cps = SA[1 1 0 -1 -1 -1  0  1 1
         0 1 1  1  0 -1 -1 -1 0]*L/2 .+ [2L,3L]
weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots =   SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

# make a nurbs curve
circle = NurbsCurve(cps,knots,weights)

# make a body and a simulation
Body = ParametricBody(circle,(0,1))
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)

# set -up simulations time and time-step for ploting
t₀ = round(sim_time(sim))
duration = 10; tstep = 0.1

# run
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# save gif
gif(anim, "/tmp/jl_sfawfg.gif", fps=24)


# measure_sdf!(sim.flow.σ,sim.body,0.0)
