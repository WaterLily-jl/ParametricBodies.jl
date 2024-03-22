using WaterLily,StaticArrays
using ParametricBodies # New package
function make_sim(;L=32,Re=1e3,St=0.3,U=1,n=8,m=4,Λ=5,T=Float32,mem=Array)
    # Map from simulation coordinate x to surface coordinate ξ
    b = T(1/Λ); h₀=T(L÷4); ω=T(π*St*U/h₀)
    ellipse(θ,t) = 0.5f0L*SA[1+cos(θ),b*sin(θ)] # define parametric curve
    map(x,t) = x-SA[n*L÷4,m*L÷2-h₀*sin(ω*t)]
    body = ParametricBody(ellipse,(0,2π);map,T,mem)  # automatically finds closest point
    # make a sim
    Simulation((n*L,m*L),(U,0),L;ν=U*L/Re,body,T,mem)
end
# using CUDA
include("viz.jl");Makie.inline!(false);

function make_video(L=32,St=0.3,name="parametric_ellipse.mp4")
    cycle = range(0,2/St,240)
    sim = make_sim(;L,St,mem=Array)
    fig,viz = body_omega_fig(sim)
    Makie.record(fig,name,2cycle) do t
        sim_step!(sim,t,verbose=true)
        update!(viz,sim)
    end
end