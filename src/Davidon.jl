# Davidon minimizer (should be in Optim, but isn't yet)
function davidon(f,a::R,b::R;kwargs...) where R<:Real
    eval = fdual(f,a)
    davidon(eval,eval(a),eval(b);kwargs...).x
end
function davidon(f,a,b;tol=√eps(a.x),∂tol=tol,verbose=false,itmx=1000)
    a.f>b.f && ((a,b)=(b,a)) # a is current minimizer
    for _ in 1:itmx
        (abs(a.x-b.x) ≤ 2tol || abs(a.∂) < ∂tol) && break
        u,v = inv_cubic(f,a,b;tol)
        verbose && @show u,v
        (u,v)==(a,b) && break
        (a,b)=(u,v)
    end; a
end
using ForwardDiff: Dual,Tag,value,partials
function fdual(f::F,::R) where {F<:Function,R<:Real}
    T = typeof(Tag(f,R))
    function (x)
        fx = f(Dual{T}(x,one(R)))
        (x=x,f=value(fx),∂=partials(T,fx,1))
    end
end
function inv_cubic(f,a,b;tol=√eps(a.x))
    a.f>b.f && ((a,b)=(b,a)) # a is current minimizer
    Δ = b.x-a.x
    v = a.∂+b.∂-3(b.f-a.f)/Δ; w = v^2-a.∂*b.∂
    w < 0 && return a,b      # bust!
    w = copysign(√w,Δ); q = b.∂-a.∂+2w
    q≈0 && return a,b        # bust!
    m,d = a.x+0.5f0Δ,abs(0.5f0Δ)-max(0.1f0abs(Δ),tol)
    c = f(clamp(b.x-Δ*(b.∂+w-v)/q,m-d,m+d))
    c.f > b.f && return a,b  # bust!
    c.f > a.f && return a,c  # save minimizer
    c,(c.∂*Δ<0 ? b : a)      # pick "downhill" bracket
end
function inv_quad(f,a,b,c,tol=√eps(b.x))
    s,r=b.x-a.x,b.x-c.x
    ss,rr = s*(b.f-c.f),r*(b.f-a.f)
    q,p = 2(ss-rr),s*ss-r*rr
    u = f(clamp(b.x-p/q,a.x+tol,c.x-tol))
    x = u.f<b.f ? u : b
    y = a.f<c.f ? a : c
    x.f<y.f ? x : y
end