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
