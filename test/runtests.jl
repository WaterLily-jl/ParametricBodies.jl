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

    @test WaterLily.sdf(body,SA[-.3,-.4],2.) == -0.5

    d,n,V = measure(body,SA[-.75,1],4.)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5]
    @test V ≈ SA[-4/5,-3/5]

    # use mapping to double and move circle
    U=0.1; map(x,t)=(x-SA[U*t,0])/2
    body = ParametricBody(surf,locate,map)
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
    locator = HashedLocator(surf,(0.,2π),t⁰=t,step=0.25)
    @test isapprox(locator.lower,SA[-1.25,-1.25],rtol=0.01)
    @test isapprox(locator(SA[.3,.4],t),atan(4,3),rtol=1e-4)
    @test isapprox(surf(locator(SA[-1.2,.9],t),t),SA[-4/5,3/5],rtol=1e-4)

    body = ParametricBody(surf,locator)
    @test isapprox(WaterLily.sdf(body,SA[-3.,-4.],t),4.,rtol=1e-2) # outside hash

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
    @test n ≈ SA[-3/5,-4/5]
    @test isapprox(V,SA[4/5,-3/5],rtol=1e-4)
end