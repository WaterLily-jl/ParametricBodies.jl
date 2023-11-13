using WaterLily
using ParametricBodies
using StaticArrays
using Plots

# parameters
L=2^4
Re=250
U =1
ϵ=0.5
thk=2ϵ+√2

# define a flat plat at and angle of attack
cps = SA[-1   0   1
         0.5 0.25 0]*L .+ [2L,3L]

# needed if control points are moved
cps_m = MMatrix(cps)
# weights = SA[1.,1.,1.]
# knots =   SA[0,0,0,1,1,1.]

# make a nurbs curve
# circle = NurbsCurve(cps_m,knots,weights)
circle = BSplineCurve(cps_m;degree=2)

# make a body and a simulation, overloead distance function
dist(p,n)=√(p'*p)-thk/2
Body = DynamicBody(circle,(0,1),dist=dist)
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)

# intialize
t₀ = sim_time(sim)
duration = 10
tstep = 0.1

# run
anim = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        # random update
        new_pnts = SA[-1     0   1
                      0.5 0.25+0.5*sin(π/4*t/sim.L) 0]*L .+ [2L,3L]
        ParametricBodies.update!(Body,new_pnts,sim.flow.Δt[end])
        measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
    end

    # flood plot
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    contourf(clamp.(sim.flow.σ,-10,10)',dpi=300,
             color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
             aspect_ratio=:equal, legend=false, border=:none)
    plot!(Body.surf; add_cp=true)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
# save gif
gif(anim, "DynamicBody_flow.gif", fps=24)
