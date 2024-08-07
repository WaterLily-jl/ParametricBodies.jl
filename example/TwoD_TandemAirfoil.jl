using WaterLily,StaticArrays
using ParametricBodies # New package
function make_sim(;L=48,Re=1e3,St=0.3,αₘ=-π/18,U=1,T=Float32,mem=Array)
    # Map from simulation coordinate x to surface coordinate ξ
    nose,pivot = SA[L,2L],SA[0.25f0L,0]
    θ₀ = T(αₘ+atan(π*St)); h₀=T(L); ω=T(π*St*U/h₀)
    function map(x,t)
        back = x[1]>nose[1]+2L       # back body?
        ϕ = back ? 5.5f0 : 0         # phase shift
        S = back ? 3L : 0            # horizontal shift
        θ = θ₀*cos(ω*t+ϕ); R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
        h = SA[S,h₀*sin(ω*t+ϕ)]
        ξ = R*(x-nose-h-pivot)+pivot # move to origin and align with x-axis
        return SA[ξ[1],abs(ξ[2])]    # reflect to positive y
    end

    # Define foil using NACA0012 profile equation: https://tinyurl.com/NACA00xx
    NACA(s) = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    foil(s,t) = L*SA[(1-s)^2,NACA(1-s)]
    body = HashedBody(foil,(0,1);map,T,mem)

    Simulation((8L,4L),(U,0),L;ν=U*L/Re,body,T,mem)
end
# using CUDA
include("viz.jl");
sim = make_sim() #mem=CuArray)
sim_gif!(sim,duration=10,step=0.1,clims=(-16,16),remeasure=true,
         plotbody=true,shift=(-2,-1.5),axis=([], false),cfill=:seismic,
         legend=false,border=:none)