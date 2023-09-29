using WaterLily,ParametricBodies
using Test

using StaticArrays
@testset "ParametricBodies.jl" begin
    surf(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locate(x::SVector{2},t) = atan(x[2],x[1])-t
    body = ParametricBody(surf,locate)

    @test body.surf(body.locate(SA[4.,3.],1),1) == SA[4/5,3/5]

    n = ParametricBodies.norm_dir(body.surf,π/2,0)
    @test n/√(n'*n) ≈ SA[0,1]

    @test WaterLily.sdf(body,SA[-.3,-.4],2.) ≈ -0.5

    d,n,V = measure(body,SA[-.75,1],4.)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5]
    @test V ≈ SA[-4/5,-3/5]

    # use mapping to double and move circle
    U=0.1; map(x,t)=(x-SA[U*t,0])/2
    body = ParametricBody(surf,locate;map)
    d,n,V = measure(body,SA[4U,-2.1],4.)
    @test d ≈ 0.1
    @test n ≈ SA[0,-1]
    @test V ≈ SA[2+U,0]
end

@testset "HashedLocators.jl" begin
    surf(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locator = HashedLocator(surf,(0.,2π),T=Float32)
    @test typeof(locator(SA_F32[0,1],0.f0))==Float32

    t = 0.
    locator = HashedLocator(surf,(0.,2π),t⁰=t,step=0.25,buffer=1)
    @test isapprox(locator.lower,SA[-1.25,-1.25],rtol=0.01)
    @test isapprox(locator(SA[.3,.4],t),atan(4,3),rtol=1e-4)
    @test isapprox(surf(locator(SA[-1.2,.9],t),t),SA[-4/5,3/5],rtol=1e-4)

    body = ParametricBody(surf,locator)
    @test isapprox(WaterLily.sdf(body,SA[-3.,-4.],t),4.,rtol=1.5e-2) # outside hash

    t = 0.5; ParametricBodies.update!(body,t)
    d,n,V = measure(body,SA[-.75,1],t)
    @test d ≈ 0.25
    @test isapprox(n,SA[-3/5, 4/5],rtol=1e-4)
    @test isapprox(V,SA[-4/5,-3/5],rtol=1e-4)

    # use mapping to double, move and rotate circle
    U=0.1; map(x,t)=SA[cos(t) sin(t); -sin(t) cos(t)]*(x-SA[U*t,0])/2
    body = ParametricBody((θ,t) -> SA[cos(θ),sin(θ)],(0.,2π);step=0.25,map)
    d,n,V = measure(body,SA[4U,-2.1],4.)
    @test d ≈ 0.1
    @test n ≈ SA[0,-1]
    @test V ≈ SA[2.1+U,0] # rotation from map ∝ r, not R

    # use mapping to limit the hash to positive quadrant
    body = ParametricBody(surf,(0.,π/2);step=0.25,map=(x,t)->abs.(x))
    d,n,V = measure(body,SA[-0.3,-0.4],0.)
    @test d ≈ -0.5
    @test isapprox(n,SA[-3/5,-4/5],rtol=1e-4)
    @test isapprox(V,SA[4/5,-3/5],rtol=1e-4)
end

using LinearAlgebra,ForwardDiff
@testset "NurbsCurves.jl" begin
    # make a square
    square = BSplineCurve(SA[5 5 0 -5 -5 -5  0  5 5
                             0 5 5  5  0 -5 -5 -5 0],degree=1)
    @test square(1f0,0) ≈ square(0f0,0) ≈ [5,0]
    @test square(.5f0,0) ≈ [-5,0]

    # ceck bspline is type stable. Note that using Float64 for cps will break this!
    @test isa(square(0.,0),SVector)
    @test all([eltype(square(zero(T),0))==T for T in (Float32,Float64)])

    # check winding direction
    cross(a,b) = det([a;;b])
    @test all([cross(square(s,0.),square(s+0.1,0.))>0 for s in range(0,.9,10)])

    # check derivatives
    dcurve(u) = ForwardDiff.derivative(u->square(u,0),u)
    @test dcurve(0f0) ≈ [0,40]
    @test dcurve(0.5f0) ≈ [0,-40]

    # Wrap the shape function inside the parametric body class and check measurements
    body = ParametricBody(square, (0,1));
    @test all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
    @test all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]])

    # Check that the locator works for closed splines
    @test [body.locate(SA_F32[5,s],0) for s ∈ (-2,-1,-0.1)]≈[0.95,0.975,0.9975]

    # NURBS test
    cps = SA[5 5 0 -5 -5 -5  0  5 5
             0 5 5  5  0 -5 -5 -5 0]
    weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots = SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1] # requires non-uniform knot and weights
    circle = NurbsCurve(cps,knots,weights)

    # Check bspline is type stable. Note that using Float64 for cps will break this!
    @test isa(circle(0.,0),SVector)
    @test all([eltype(circle(zero(T),0))==T for T in (Float32,Float64)])

    # Wrap the shape function inside the parametric body class and check measurements
    body = ParametricBody(circle, (0,1));
    @test all(measure(body,[-6,0],0) .≈ [1,[-1,0],[0,0]])
    @test all(measure(body,[ 5,5],0) .≈ [√(5^2+5^2)-5,[√2/2,√2/2],[0,0]])
    @test all(measure(body,[-5,5],0) .≈ [√(5^2+5^2)-5,[-√2/2,√2/2],[0,0]])
end