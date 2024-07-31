using WaterLily
using ParametricBodies
using StaticArrays
include("viz.jl")

function circle(;L=2^6,Re=250,U=1,mem=Array,T=Float32)

    # NURBS points, weights and knot vector for a circle
    cps = SA_F32[1 1 0 -1 -1 -1  0  1 1
                0 1 1  1  0 -1 -1 -1 0]*L/2 .+ [2L,3L]
    weights = SA_F32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   SA_F32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

    # make a nurbs curve and a body for the simulation
    Body = ParametricBody(NurbsCurve(cps,knots,weights))
    Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=T,mem=mem)
end

# make a sim
sim = circle()

# set -up simulations time and time-step for ploting
t₀,duration,tstep = round(sim_time(sim)), 1, 0.1

# storage
pforce,vforce,pmom = [],[],[]

# run
@gif for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)
    plot!(sim.body.curve;shift=(1.5,1.5),add_cp=true)

    # compute and store force
    push!(pforce,WaterLily.pressure_force(sim)[1])
    push!(vforce,WaterLily.viscous_force(sim)[1])
    # push!(pmom,pressure_moment(SA_F32[2sim.L,3sim.L],sim)) # should be zero-ish

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
plot(range(t₀,t₀+duration;step=tstep),2pforce/sim.L,label="pressure",xlabel="tU/L",ylabel="force")
plot!(range(t₀,t₀+duration;step=tstep),2vforce/sim.L,label="viscous")