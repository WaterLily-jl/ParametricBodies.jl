using WaterLily,StaticArrays
using ParametricBodies # New package
function make_sim(;L=32,Re=1e3,St=0.3,U=1,n=8,m=4,Λ=5,T=Float64,mem=Array)
    # Map from simulation coordinate x to surface coordinate ξ
    b = T(1/Λ); h₀=T(L÷4); ω=T(π*St*U/h₀)
    ellipse(θ,t) = 0.5f0L*SA[1+cos(θ),b*sin(θ)] # define parametric curve
    map(x,t) = x-SA[n*L÷4,m*L÷2-h₀*sin(ω*t)]
    body = HashedBody(ellipse,(0,2π);T,map)  # automatically finds closest point
    # make a sim
    Simulation((n*L,m*L),(U,0),L;ν=U*L/Re,body,T,mem)
end
using CUDA
include("viz.jl");

# set -up simulations time and time-step for ploting
sim = make_sim(;mem=Array,T=Float32)
t₀ = round(sim_time(sim))
duration = 1; tstep = 0.1
p,ν = [],[]

# run
@gif for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=true)

    # flood plot
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)
    plot!(sim.body,WaterLily.time(sim))
    push!(p,pressure_force(sim)[1])
    push!(ν,viscous_force(sim)[1])

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
plot(p,label="Pressure force")
plot!(ν,label="Viscous force")