# using WaterLily
using ParametricBodies
using StaticArrays
using Plots

# parameters
L=2^4
ϵ=0.5
thk=2ϵ+√2

# overload the distance function
ParametricBodies.dis(p,n) = √(p'*p) - thk/2

# # NURBS points, weights and knot vector for a nurbs
# cps = SA[1 1 0 -1 -1 -1  0  1 1
#          0 1 1  1  0 -1 -1 -1 0]*L/2 .+ [2L,3L]
# cps_m = MMatrix(cps)
# weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
# knots =   SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

cps = SA[-1 0 1
         0.5 0.25 0]*L
cps_m = MMatrix(cps)
weights = SA[1.,1.,1.]
knots =   SA[0,0,0,1,1,1.]

# make a nurbs curve
nurbs = NurbsCurve(copy(cps_m),knots,weights)

# make a body and a simulation
Body = DynamicBody(nurbs,(0,1))
# sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)

x = -2L:1:2L
dist = zeros(length(x),length(x))
for i in eachindex(x), j in eachindex(x)
    dist[i,j] = sdf(Body,[x[i],x[j]],0.0)
end
Plots.contourf(x,x,dist',cmpa=:RdBu,levels=21)


# update
ParametricBodies.update!(Body,0.0*cps.-[0,L],1.0)

dist = zeros(length(x),length(x))
for i in eachindex(x), j in eachindex(x)
    dist[i,j] = sdf(Body,[x[i],x[j]],0.0)
end
Plots.contourf(x,x,dist',cmpa=:RdBu,levels=21)
for i in eachindex(x), j in eachindex(x)
    dist[i,j] = measure(Body,[x[i],x[j]],0.0)[3][2]
end
Plots.contourf(x,x,dist',cmpa=:RdBu,levels=21)
