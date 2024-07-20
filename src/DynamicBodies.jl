
# using LinearAlgebra: dot
# function norm_dir(nurbs::NurbsCurve{3},uv::Number,p::SVector{3},t)
#     s = ForwardDiff.derivative(uv->nurbs(uv,t),uv); s/=√(s'*s)
#     return p-dot(p,s)*s
# end

# """
#     ∮nds(p,body::DynamicBody,t=0)

# Surface normal pressure integral along the parametric curve(s)
# """
# function ∮nds(p::AbstractArray{T},body::DynamicBody,t=0;N=64) where T
#     open = !all(body.surf(body.locate.lims[1],t).≈body.surf(body.locate.lims[2],t))
#     integrate(s->_pforce(body.surf,p,s,t,Val{open}()),body.surf,t,body.locate.lims;N)
# end
# """
#     ∮τnds(u,body::DynamicBody,t=0)

# Surface normal pressure integral along the parametric curve(s)
# """
# function ∮τnds(u::AbstractArray{T},body::DynamicBody,t=0;N=64) where T
#     open = !all(body.surf(body.locate.lims[1],t).≈body.surf(body.locate.lims[2],t))
#     integrate(s->_vforce(body.surf,u,s,t,body.velocity(s,0),Val{open}()),body.surf,t,body.locate.lims;N)
# end