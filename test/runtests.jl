using ParametricBodies
using StaticArrays,Test

@testset "ParametricBodies.jl" begin
    surf(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locate(x::SVector{2},t) = atan(x[2],x[1])-t
    body = ParametricBody(surf,locate)

    @test body.surf(body.locate(SA[4.,3.],1),1) == SA[4/5,3/5]

    s = ParametricBodies.tangent_dir(body.surf,π/2,0)
    @test s/√(s'*s) ≈ SA[-1,0]
    n = ParametricBodies.norm_dir(s)
    @test n/√(n'*n) ≈ SA[0,1]
    t = ParametricBodies.aligned_dir(SA[0.1,0.5],s) # points in direction n
    @test t/√(t'*t) ≈ SA[0,1]
    t = ParametricBodies.aligned_dir(SA[0.1,-0.5],s) # aligns with p to point down
    @test t/√(t'*t) ≈ SA[0,-1]

    @test sdf(body,SA[-.3,-.4],2.) ≈ -0.5
    d,n,V = measure(body,SA[-.75,1],4.)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5]
    @test V ≈ SA[-4/5,-3/5]

    # use mapping to double and move circle
    U=0.1; map(x,t)=(x-SA[U*t,0])/2
    body = ParametricBody(surf,locate;map,x₀=SA_F64[0,0])
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
    @test locator.lower ≈ SA[-1.25,-1.25] rtol=0.01
    @test locator(SA[.3,.4],t) ≈ atan(4,3) rtol=1e-4
    @test surf(locator(SA[-1.2,.9],t),t) ≈ SA[-4/5,3/5] rtol=1e-4

    body = ParametricBody(surf,locator)
    @test sdf(body,SA[-3.,-4.],t) ≈ 4. rtol=1.5e-2 # outside hash

    t = 0.5; ParametricBodies.update!(body,t)
    d,n,V = measure(body,SA[-.75,1],t)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5] rtol=1e-4
    @test V ≈ SA[-4/5,-3/5] rtol=1e-4

    # use mapping to double, move and rotate circle
    U=0.1; map(x,t)=SA[cos(t) sin(t); -sin(t) cos(t)]*(x-SA[U*t,0])/2
    body = HashedBody((θ,t) -> SA[cos(θ),sin(θ)],(0.,2π);step=0.25,map,T=Float64)
    d,n,V = measure(body,SA[4U,-2.1],4.)
    @test d ≈ 0.1
    @test n ≈ SA[0,-1]
    @test V ≈ SA[2.1+U,0] # rotation from map ∝ r, not R

    # use map to model full circle with only the positive quadrant arc
    step,buffer = 0.2,1
    body = HashedBody(surf,(0.,π/2);step,buffer,map=(x,t)->abs.(x),T=Float64)
    @test body.locate.lower ≈ step*buffer*[-1,-1]
    @test size(body.locate.hash) == (1÷step+1+2buffer,1÷step+1+2buffer) # radius/step+5
    @test [measure(body,SA[-0.3,-0.4],0.)...] ≈ [-0.5,[-3/5,-4/5],[4/5,-3/5]] rtol=1e-4

    # model arc as space-curve with finite thickness
    thk = 0.1
    body = HashedBody(surf,(0.,π/2);step,buffer,T=Float64,thk,signed=false)
    # near the end works
    x = SA[0.9,-0.1]; p = x-SA[1,0]; m = √(p'*p)
    @test [measure(body,x,0.)...] ≈ [m-thk,p ./ m,[0,1]] rtol=1e-4
    # but points inside the arc return "outside" normal!
    @test measure(body,SA[0.4,0.3],0.)[2] ≈ [-4/5,-3/5] rtol=1e-4
end

using LinearAlgebra,ForwardDiff
@testset "NurbsCurves.jl" begin
    T = Float32
    # make a square
    square = BSplineCurve(SA{T}[5 5 0 -5 -5 -5  0  5 5
                                0 5 5  5  0 -5 -5 -5 0],degree=1)
    @test square(1f0,0) ≈ square(0f0,0) ≈ [5,0]
    @test square(.5f0,0) ≈ [-5,0]

    # check bspline is type stable. Note that using Float64 for cps will break this!
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
    body = HashedBody(square, (0,1));
    @test [measure(body,SA[1,2],0)...] ≈ [-3,[0,1],[0,0]]
    @test [measure(body,SA[8,2],0)...] ≈ [3,[1,0],[0,0]]

    # Check that HashedLocator works for closed splines
    @test [body.locate(SA_F32[5,s],0) for s ∈ (-2,-1,-0.1)]≈[0.95,0.975,0.9975]

    # NURBS test
    cps = SA{T}[5 5 0 -5 -5 -5  0  5 5
                0 5 5  5  0 -5 -5 -5 0]
    weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots = SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1] # requires non-uniform knot and weights
    circle = NurbsCurve(cps,knots,weights)

    # Check bspline is type stable. Note that using Float64 for cps will break this!
    @test isa(circle(0.,0),SVector)
    @test all([eltype(circle(zero(T),0))==T for T in (Float32,Float64)])

    # Wrap the shape function inside the parametric body class and check measurements
    body = HashedBody(circle, (0,1));
    @test [measure(body,SA[-6,0],0)...] ≈ [1,[-1,0],[0,0]]
    @test [measure(body,SA[ 5,5],0)...] ≈ [5√2-5,[ √2/2,√2/2],[0,0]] rtol=1e-6
    @test [measure(body,SA[-5,5],0)...] ≈ [5√2-5,[-√2/2,√2/2],[0,0]] rtol=1e-6

    # test interpolation base on the NURBS-book p.367 Ex9.1
    pnts = SA[0. 3. -1. -4  -4.
              0. 4.  4.  0. -3.]
    nurbs,s = interpNurbs(pnts;p=3),ParametricBodies._u(pnts)
    @test s≈[0.,5/17,9/17,14/17,1.]
    @test nurbs.knots≈[0.,0.,0.,0.,28/51,1.,1.,1.,1.]
    @test all(reduce(hcat,nurbs.(s,0.0)).-pnts.<10eps(eltype(pnts)))
end
@testset "NurbsLocator.jl" begin
    # define a circle
    T = Float32
    cps = SA{T}[5 5 0 -5 -5 -5  0  5 5
                0 5 5  5  0 -5 -5 -5 0]
    weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots = SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1] # requires non-uniform knot and weights
    circle = NurbsCurve(cps,knots,weights)

    # Check NurbsLocator
    locate = NurbsLocator(circle)
    @test locate.C¹end # closed & smooth
    @test locate([5,5],0) ≈ 1/8
    body = ParametricBody(circle,locate)
    @test [measure(body,SA[5,5],0)...]≈[5√2-5,[√2/2,√2/2],[0,0]] rtol=1e-6

    # Check DynamicNurbsBody
    body = DynamicNurbsBody(circle)    
    @test [measure(body,SA[5,5],0)...]≈[5√2-5,[√2/2,√2/2],[0,0]] rtol=1e-6
    body = update(body,cps.+T(0.1),T(0.1))
    @test [measure(body,SA[5,5],0)...]≈[4.9√2-5,[√2/2,√2/2],[1,1]] rtol=1e-6
    @test [measure(body,SA[0,0],0)...]≈[0.1√2-5,[-√2/2,-√2/2],[1,1]] rtol=1e-6
end
@testset "Extruded Bodies" begin
    T = Float32
    cps = SA{T}[7 7 0 -7 -7 -7  0  7 7
                0 7 7  7  0 -7 -7 -7 0]
    weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots = SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1] # requires non-uniform knot and weights
    circle = NurbsCurve(cps,knots,weights)
    
    # Make a cylinder
    map(x::SVector{3},t) = SA[x[2],x[3]]
    cylinder = ParametricBody(circle;map,scale=1f0)
    @test [measure(cylinder,SA[2,3,6],0)...] ≈ [√45-7,[0,3,6]./√45,[0,0,0]] atol=1e-4

    # Make a sphere
    map(x::SVector{3},t) = (r = √(x[2]^2+x[3]^2); SA[x[1],r])
    sphere = ParametricBody(circle;map,scale=1f0)
    @test [measure(sphere,SA[2,3,6],0)...] ≈ [0,[2,3,6]./7,[0,0,0]] atol=1e-4
end
@testset "PlanarBodies.jl" begin
    T = Float32
    # make a square plate
    square = BSplineCurve(SA{T}[5 5 0 -5 -5 -5  0  5 5
                                0 5 5  5  0 -5 -5 -5 0],degree=1)
    body = PlanarBody(square,(0,1)) # test with HashedLocator
    @test [measure(body,SA[8,2,0],0)...]≈[2-√3/2,[1,0,0],[0,0,0]] rtol=1e-6
    @test [measure(body,SA[2,2,3],0)...]≈[2-√3/2,[0,0,1],[0,0,0]] rtol=1e-6

    body = PlanarBody(square;map=(x,t)->SA[2x[2],2x[3],2x[1]]) # test with NurbsLocator & map
    @test [measure(body,SA[0,8,2],0)...]≈[9/2-√3/2,[0,1,0],[0,0,0]] rtol=1e-6
    @test [measure(body,SA[3,2,2],0)...]≈[2-√3/2,[1,0,0],[0,0,0]] rtol=1e-6
end
using CUDA,WaterLily
@testset "WaterLily" begin
    function circle_sim(nurbslocate=true,mem=Array,T=Float32)
        cps = SA{T}[5 5 0 -5 -5 -5  0  5 5
                    0 5 5  5  0 -5 -5 -5 0]
        weights = SA[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
        knots = SA[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]
        circle = NurbsCurve(cps,knots,weights)
        body = nurbslocate ? DynamicNurbsBody(circle) : HashedBody(circle,(0,1);T,mem)
        return Simulation((8,8),(1,0),5;body,mem,T)
    end
    for mem in (CUDA.functional() ? [CuArray,Array] : [Array])
        for nurbs in [true,false]
            sim = circle_sim(nurbs,mem); d = sim.flow.σ |> Array
            for I in CartesianIndices((4:6,3:7))
                @test d[I]≈√sum(abs2,WaterLily.loc(0,I))-5 atol=1e-6
            end
            if nurbs # `sim.body=...` requires WaterLily 1.2.0+
                dc = 1f0
                sim.body = update!(sim.body, sim.body.surf.pnts .+ dc, sim.flow.Δt[end])
                measure_sdf!(sim.flow.σ,sim.body); d = sim.flow.σ |> Array
                I = CartesianIndex(5,5)
                @test d[I]≈√sum(abs2,WaterLily.loc(0,I) .- dc)-5 atol=1e-6
            end
        end
    end
end