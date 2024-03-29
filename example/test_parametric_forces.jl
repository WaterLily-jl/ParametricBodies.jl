using ParametricBodies
using StaticArrays

L = 32
T = Float64

# NURBS points, weights and knot vector for a circle
cps = SA{T}[1 1 0 -1 -1 -1  0  1 1
            0 1 1  1  0 -1 -1 -1 0]*L/2 .+ [L,L]
weights = SA{T}[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots =   SA{T}[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

# make different types of bodies
close_parametric_body = ParametricBody(NurbsCurve(cps,knots,weights),(0,1);T=T)
close_dynamic_body = DynamicBody(NurbsCurve(cps,knots,weights),(0,1);T=T)
open_parametric_body = ParametricBody(BSplineCurve(copy(cps[:,1:end-3]);degree=2),(0,1);T=T)
open_dynamic_body = DynamicBody(BSplineCurve(copy(cps[:,1:end-3]);degree=2),(0,1);T=T)

# dymmy pressure and velocity
p = ones(2L,2L); u = ones(2L,2L,2);

# check clobal behaviour
@show ParametricBodies.∮nds(p,close_parametric_body,0.0)
@which ParametricBodies.∮nds(p,close_parametric_body,0.0)

@show ParametricBodies.∮nds(p,close_dynamic_body,0.0)
@which ParametricBodies.∮nds(p,close_dynamic_body,0.0)

@show ParametricBodies.∮nds(p,open_parametric_body,0.0)
@which ParametricBodies.∮nds(p,open_parametric_body,0.0)

@show ParametricBodies.∮nds(p,open_dynamic_body,0.0)
@which ParametricBodies.∮nds(p,open_dynamic_body,0.0)

# check nested behaviour
@show ParametricBodies._pforce(close_parametric_body.surf,p,0.0,0.0,Val{false}())
@which ParametricBodies._pforce(close_parametric_body.surf,p,0.0,0.0,Val{false}())

@show ParametricBodies._pforce(close_dynamic_body.surf,p,0.0,0.0,Val{false}())
@which ParametricBodies._pforce(close_dynamic_body.surf,p,0.0,0.0,Val{false}())

@show ParametricBodies._pforce(open_parametric_body.surf,p,0.0,0.0,Val{true}())
@which ParametricBodies._pforce(open_parametric_body.surf,p,0.0,0.0,Val{true}())

@show ParametricBodies._pforce(open_dynamic_body.surf,p,0.0,0.0,Val{true}())
@which ParametricBodies._pforce(open_dynamic_body.surf,p,0.0,0.0,Val{true}())