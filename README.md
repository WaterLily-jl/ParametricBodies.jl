# ParametricBodies

Tutorial video [![tutorial video link](https://img.youtube.com/vi/6PmJJKVOfvc/hqdefault.jpg)](https://www.youtube.com/watch?v=6PmJJKVOfvc)


This is a preliminary (unregistered) pacakge to enable working with parametrically defined shapes in [WaterLily](https://github.com/weymouth/WaterLily.jl). It defines two types, a [ParametricBody](https://github.com/weymouth/ParametricBodies.jl/blob/ec16d7efb5964c2200da65c71e643d7fbaf064c2/src/ParametricBodies.jl#L35) to hold the shape definition and shape interrogation methods, and a [HashedLocator](https://github.com/weymouth/ParametricBodies.jl/blob/ec16d7efb5964c2200da65c71e643d7fbaf064c2/src/HashedLocators.jl#L33) to robustly locate the closest point on the shape. Many of the methods are currently specific to curves (defined by only one parameter) but could be extended to surfaces fairly directly.

Until this package matures and is registered, you need to either [add it via github](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages) 
```
] add https://github.com/weymouth/ParametricBodies.jl
```
or download the github repo and then [activate the environment](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project)
```
shell> git clone https://github.com/weymouth/ParametricBodies.jl
Cloning into 'ParametricBodies.jl'...
...
] activate ParametricBodies
] instantiate
```

### Usage

`ParametricBodies.jl` allows to define two types of bodies: `ParametricBody` and `DynamicBody`.

##### Parametric Bodies

A `ParametricBody` can be define from a parametric function repsentation of that shape. For example, a circle can be defined as
```julia
circle(s,t) = SA[sin(2π*s),cos(2π*s)] # a circle
Body = ParametricBody(circle, (0,1))
```
The interval `(0,1)` gives the interval in which the parametric curve is defined

`ParametricBody` can be made dynamic by specifying a mapping to the `ParametricBody` type
```julia
heave(x,t) = SA[0.,sin(2π*t)]
Body = ParametricBody(circle, (0,1); map=heave)
```


##### Dynamic Bodies

`DynamicBodies` can be constructed from `NurbsCurves` and `BSplineCurves` and allow the user to adjust (move) the curve's control points during the simulations. Internally, `BSplineCurves` are `NurbsCurves` and the `DynamicBody` is constructed from the `NurbsCurve` and a `NurbsLocator` is created to allow for a fast and robust evaluation of the location of the closest point on the curve.

Dynamic adjustement of control points requires creating the curves' control point with a `MMatrix` from the `StaticArrays` package as such

```julia
cps = MMatrix(SA[1 0 -1; 0 0 0])
spline = BSplineCurve(cps; degree=2)
Body = DynamicBody(spline,(0,1))
```
Updating the control point is can be simply done by passing the new control point position and a time step (used to compute the NURBS' velocity)
```julia
update!(DynamicBody, new_cps, Δt)
```
