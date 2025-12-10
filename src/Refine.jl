"""
    refine(curve,lims,closed)::Function

Returns a `function(u₀,x,t)` which finds `u⁺ = argmin(d²(u)=|curve(u,t)-x|²)` by solving

    d²′ = (curve(u⁺,t)-x)'*tangent(curve,u⁺,t) = 0

starting from an initial guess `u₀`. The function attempts to Newton step to the root, falling back on
gradient descent if `d²′′<0`. The resulting minimizer respects `u⁺ ∈ lims` and `closed` curves.

    Note: A good inital guess `u₀` is critical for robustly finding the _global_ minimizer.

    Optional inputs:
    - `itmx=10`: Maximum number of steps to take
    - `stpmx=(lims[2]-lims[1])/20`: Maximum step size
    - `stpmn=stpmx/100`: Minimum step size before the loop will exit
    - `stpmd=stpmx/10`: Step size below which `d² ≥ fastd²` will exit the loop
"""
function refine(curve,lims,closed)::Function
    align(u,x,t) = (curve(u,t)-x)'*tangent(curve,u,t)
    dalign(u,x,t) = ForwardDiff.derivative(u->align(u,x,t),u)
    return function(u::T,x,t;fastd²=Inf,itmx=10,stpmx=(lims[2]-lims[1])/20,stpmn=stpmx/100,stpmd=stpmx/10) where T
        for _ in 1:itmx
            u₀,a,da = u,align(u,x,t),dalign(u,x,t)
            step = da < eps(T) ? -copysign(stpmx,a) : -clamp(a/da,-stpmx,stpmx)
            u = closed ? mymod(u+step,lims...) : clamp(u+step,lims...)
            Δ = min(abs(step),abs(u-u₀))
            (Δ<stpmn || isfinite(fastd²) && Δ<stpmd && sum(abs2,curve(u,t)-x)≥fastd²) && break
        end; u
    end
end
@inline mymod(x,low,high) = low+mod(x-low,high-low)
