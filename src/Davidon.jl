# Davidon minimizer (should be in Optim, but isn't yet)
using ForwardDiff: Dual,Tag,value,partials
function fdual(f::F,::R) where {F<:Function,R<:Real}
    T = typeof(Tag(f,R))
    function (x)
        fx = f(Dual{T}(x,one(R)))
        (x=x,f=value(fx),∂=partials(T,fx,1))
    end
end
function davidon(f,ab::Tuple{R,R};kwargs...) where R<:Real
    eval = fdual(f,ab[1])
    _davidon(eval,eval.(ab)...;kwargs...).x
end
function _davidon(f,a,b;tol=1e-3,∂tol=0,verbose=false,itmx=1000)
    Δ = b.x-a.x
    verbose && @show a,b
    for _ in 1:itmx
        v = a.∂+b.∂-3*(b.f-a.f)/Δ; w = copysign(√(v^2-a.∂*b.∂),Δ)
        x = b.x-Δ*(b.∂+w-v)/(b.∂-a.∂+2w)
        c = f(clamp(x,min(a.x,b.x)+max(Δ/8,tol),max(a.x,b.x)-max(Δ/8,tol)))
        verbose && @show c
        c.f > max(a.f,b.f) && break
        (c.f < min(a.f,b.f) ?  c.∂*Δ < 0 : a.f > b.f) ? (a=c) : (b=c)
        Δ = b.x-a.x; (abs(Δ) ≤ 2tol || abs(a.f>b.f ? b.∂ : a.∂)<∂tol) && break
    end
    a.f < b.f ? a : b
end