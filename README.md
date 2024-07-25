# ParametricBodies

Tutorial video [![tutorial video link](https://img.youtube.com/vi/6PmJJKVOfvc/hqdefault.jpg)](https://www.youtube.com/watch?v=6PmJJKVOfvc)


This pacakge to enable working with parametrically defined shapes in [WaterLily](https://github.com/weymouth/WaterLily.jl). You can add this package via the Julia package manager
```
] add ParametricBodies
using ParametricBodies
```

A `ParametricBody` is a subtype of  `WaterLily.AbstractBody`, defining the `measure(body,x,t)` function needed to run WaterLily simulations.

### Parametric Bodies

A `ParametricBody` is defined using a parametric function repsentation of that shape. For example, a circle can be defined as
```julia
using StaticArrays
curve(θ,t) = SA[cos(θ),sin(θ)]
locate(x::SVector{2},t) = atan(x[2],x[1])
body = ParametricBody(curve,locate)
```
The parametric `curve` function defines the boundary of the circle, and the `locate` function is a general inverse of `curve`: locating the value of the parameter `θ` which is closest to point `x` at time `t`. The `body` can now be used inside a 2D WaterLily simulation. 

A few important features to note: **First**, the parametric curve should _always_ return an `SVector`! Returning an array will allocate and won't run on GPUs. 

**Second**, a 2D curve can either define the boundary of a closed body, or it can define a space-curve body with some finite thickness. The default in 2D is to assume the curve is a boundary, but you can change this by supplying additional arguments. For example, we could model a thin wing section defined by a circular-arc as
```julia
locate(x::SVector{2},t) = clamp(atan(x[2],x[1]),π/3,2π/3)
arc = ParametricBody(curve,locate,thk=0.1,boundary=false)
```
All ParametricBodies defined by 3D curves are space-curves since the normal isn't uniquely defined, but the `PlanarBody` type discussed below lets us define closed membranes in 3D.

**Third**, the functions `curve` and especially `locate` can be tricky to define for general curves. The next few sections discuss methods to simplify and automate the construction of these functions. 

### Hashed Locator & Body

A `HashedLocator` struct has been defined to automate locating the closest point on a supplied 2D curve. 
```julia
lims = (0.,2π) # limits of the parametric function
locator = HashedLocator(curve,lims,step=0.25)
body = ParametricBody(curve,locate)
```
or the convience constructor
```julia
body = HashedBody(curve,(0.,2π),step=0.25)
```
This locator function samples the curve over the supplied parametric limits and uses a Newton root finding method to locate the parameter value. A 2D array of parameter data (a hash table) is used to supply a good initial guess to the Newton solver. This hash must be updated if the curve is time-varying, and must be stored in a GPU array when computing on the GPU. See the example folder. The `HashedLocator` is currently only available for 2D curves, although it can be used with mapping and `PlanarBodies`, to define bodies for 3D simulations, see below.

### NurbsCurve & Locator

A NURBS (Non-Uniform Rational B-Spline) based `NurbsCurve` struct make both the definition and location process more simple, and extends to 3D space-curves. For example, given a matrix of 2D points `pnts` along a desired curve, we can define a body easily
```julia
pnts = SA[0. 3. -1. -4  -4.
          0. 4.  4.  0. -3.]
nurbs = interpNurbs(pnts;p=3) # fit a cubic NurbsCurve through the points
body = ParametricBody(nurbs)
```
An efficient (and hash-free) `NurbsLocator` is created automatically for `NurbsCurve`s. 

You can also create a NURBS by supplying the control points, knots and weights directly. For example, the code below defines a torus in 3D using a NURBS to describe the major circle of radius 7, and thickening the space-curve with a minor radius of 1.
```julia
cps = SA_32[7 7 0 -7 -7 -7  0  7 7
            0 7 7  7  0 -7 -7 -7 0
            0 0 0  0  0  0  0  0 0] # a (planar) 3D circle
weights = SA_32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.] # A perfect circle requires...
knots = SA_32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]  # non-uniform knot and weights
circle = NurbsCurve(cps,knots,weights)
torus = ParametricBody(circle,thk=1,boundary=false)
```

### Mappings & 3D surfaces

Like `WaterLily.AutoBodies`, the utility of `ParametricBodies` is greatly increased through an optional `map` function. 

Primarily, mapping can be used to move, rotate and scale the body in the simulation. For example, let's place the arc-wing above in a simulation:
```julia
using WaterLily
function arc_sim(R = 32, α = π/10, U=1, Re=100)
    curve(θ,t) = SA[cos(θ),sin(θ)] # ξ-space: angle=0,center=0,radius=1
    Rotate = SA[cos(α) sin(α); -sin(α) cos(α)]
    center = SA[R,R÷2]
    scale = R
    map(x,t) = Rotate*(x-center)/scale # map from x-space to ξ-space
    arc = HashedBody(curve,(π/3,2π/3),thk=√(2/2)+1,boundary=false,map=map)
    Simulation((6R,2R),(U,0),R,body=arc,ν=U*R/Re)
end
sim = arc_sim()
```
See the WaterLily repo, the video above, and the examples in the next section for more discussion of this primary use of the map function. 

A secondary application of the mapping function for `ParametricBodies` is to map from a 3D x-space to a 2D ξ-space, effectively extruding a 2D parametric curve into a 3D surface. For example, we can make a cylinder or sphere starting from a 2D NURBS circle using
```julia
# Make a cylinder
map(x::SVector{3},t) = SA[x[2],x[3]] # extrude along x[1]-axis
cylinder = ParametricBody(circle;map,scale=1f0)

# Make a sphere
map(x::SVector{3},t) = SA[x[1],√(x[2]^2+x[3]^2)] # revolve around x[1]-axis
sphere = ParametricBody(circle;map,scale=1f0)
```
and if we started from, say, a NACA profile, the same technique could make a uniform 3D wing or a 3D air-ship hull. Note that the scale argument must be passed in this case since the determinant of this mixed dimensional `map` isn't defined. 

A mapping is not sufficient to make 3D planar geometries, so a simple wrapper struct `PlanarBody` is defined for this purpose. For example, a circular disk can be created using
```julia
disk = PlanarBody(circle;map=(x,t)->SA[2x[2],2x[3],2x[1]])
```
where the mapping has been used to scale and rotate the disk as well.

### Dynamic Bodies

`ParametricBodies` have two ways to be dynamic: 
1. A time-varying parametric curve
2. A time-varying map

and these methods can be used together. There are many examples of time varying mappings in the WaterLily repo and the video above, so we'll focus on option 1. 

If `curve` depends explicitly on `t`, this will automatically be reflected in the position and velocity of the body in a simulation. For example, a spinning circle is easily acheived using
```julia
curve(θ,t) = SA[cos(θ+t),sin(θ+t)]
locate(x::SVector{2},t) = atan(x[2],x[1])-t
spinning_body = ParametricBody(curve,locate)
```
If you are using a `HashedLocator` for a dynamic curve you will need to call `update!(body,t)` frequently (probably every time step of the simulation) so that the initial guess of the locator reflects the correct position of the body.

We supply a special function `DynamicNurbsBody` for dynamics NURBS which defines a second spline for the velocity. This function also requires calling `update!(body,...)`, but in this case, the control points for both the position and velocity are updated. Here's an example
```
body = DynamicNurbsBody(circle)
dt,dx = 0.1,0.1                  # time step and uniform displacement
new_pnts = circle.pnts .+ dx     # define updated control points
body = update!(body,new_pnts,Δt) # new body will have unit velocity (dx/dt=1)
```
