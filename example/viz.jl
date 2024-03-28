using Plots

function get_omega!(sim)
    body(I) = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * body(I) * sim.L / sim.U
end

plot_vorticity(ω; limit=maximum(abs,ω)) = Plots.contourf(clamp.(ω|>Array,-limit,limit)',dpi=300,
               color=palette(:RdBu_11), clims=(-limit, limit), linewidth=0,
               aspect_ratio=:equal, legend=false, border=:none)

function get_body!(bod,sim,t=WaterLily.time(sim))
    @inside sim.flow.σ[I] = WaterLily.sdf(sim.body,SVector(Tuple(I).-0.5f0),t)
    copyto!(bod,sim.flow.σ[inside(sim.flow.σ)])
end
using GLMakie
function body_omega_fig(sim,resolution=(1400,700))
    #Set up figure
    fig = Figure(;resolution)
    ax = Axis(fig[1, 1]; autolimitaspect=1)
    hidedecorations!(ax); hidespines!(ax)

    # Get first vorticity viz
    vort = sim.flow.σ[inside(sim.flow.σ)] |> Array; ovort = Observable(vort)
    get_omega!(vort,sim); notify(ovort)
    heatmap!(ax,ovort,colorrange=(-20.,20.),colormap=:curl,interpolate=true)

    # Set up body viz
    bod = sim.flow.σ[inside(sim.flow.σ)] |> Array; obod = Observable(bod)
    get_body!(bod,sim); notify(obod)
    colormap = to_colormap([:grey30,(:grey,0.5)])
    contourf!(ax,obod,levels=[-100,0,1];colormap)
    fig,(vort,ovort,bod,obod)
end
function update!(viz,sim)
    vort,ovort,bod,obod = viz
    get_omega!(vort,sim); notify(ovort)
    get_body!(bod,sim); notify(obod)
end