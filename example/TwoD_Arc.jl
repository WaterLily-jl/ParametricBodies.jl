using WaterLily
using ParametricBodies
using StaticArrays
include("../../WaterLily/examples/TwoD_plots.jl")
function arc_sim(R = 64, α = π/16, U=1, Re=100)
    curve(θ,t) = SA[cos(θ),sin(θ)] # ξ-space: angle=0,center=0,radius=1
    Rotate = SA[cos(α) -sin(α); sin(α) cos(α)]
    center = SA[R,-2R÷5]
    scale = R
    map(x,t) = Rotate*(x-center)/scale # map from x-space to ξ-space
    arc = HashedBody(curve,(π/3,2π/3),thk=√2/2+1,boundary=false,map=map)
    Simulation((3R,R),(U,0),R,body=arc,ν=U*R/Re)
end
sim = arc_sim()
sim_gif!(sim;duration=1.0,step=0.1,clims=(-10,10))