using WaterLily
using ParametricBodies
using StaticArrays
include("examples/TwoD_plots.jl")

# parameters
L=2^4
Re=250
U =1
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# # NURBS points, weights and knot vector for a circle
# cps = SA[1 1 0 -1 -1 -1  0  1 1
#          0 1 1  1  0 -1 -1 -1 0]*L/2 .+ [2L,3L]
# cps_m = MMatrix(cps)
# weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
# knots =   SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

cps = SA[-1 0 1
         0.5 0.25 0]*L .+ [2L,3L]
cps_m = MMatrix(cps)
weights = SA[1.,1.,1.]
knots =   SA[0,0,0,1,1,1.]

# make a nurbs curve
circle = NurbsCurve(copy(cps_m),knots,weights)

# some motion
static(x,t) = x
heave(x,t) = x .- SA[0,L*sin(t*U/L)]

# make a body and a simulation
# Body = ParametricBody(circle,(0,1);map=static)
Body = DynamicBody(circle,(0,1))
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)

# intialize
t₀ = sim_time(sim)
duration = 10
tstep = 0.1

# run
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    # sim_step!(sim,tᵢ,remeasure=false)
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        # random update
        new_pnts = SA[-1 0 1
                      0.5 0.25+0.5*sin(π/4*t/sim.L) 0]*L .+ [2L,3L]
        ParametricBodies.update!(Body,new_pnts,sim.flow.Δt[end])
        measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
    end

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)
    Xs = reduce(hcat,[Body.surf(s,0.0) for s ∈ 0:0.01:1])
    plot!(Xs[1,:].+0.5,Xs[2,:].+0.5,color=:black,lw=thk,legend=false)
    plot!(Body.surf.pnts[1,:].+0.5,Body.surf.pnts[2,:].+0.5,marker=:circle,legend=:none)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# save gif
gif(anim, "circle_flow.gif", fps=24)
