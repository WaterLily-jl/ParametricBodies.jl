using ParametricBodies
using Test

using StaticArrays
@testset "ParametricBodies.jl" begin
    surf(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locate(x::SVector{2},t) = atan(x[2],x[1])-t
    body = ParametricBody(surf,locate)

    @test body.surf(body.locate(SA[4.,3.],1),1) == SA[4/5,3/5]

    n = ParametricBodies.norm_dir(body.surf,π/2,0)
    @test n/√(n'*n) ≈ SA[0,1]

    @test sdf(body,SA[-.3,-.4],2.) == -0.5

    d,n,V = measure(body,SA[-.75,1],4.)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5]
    @test V ≈ SA[-4/5,-3/5]
end

@testset "HashedLocators.jl" begin
    surf(θ,t) = SA[cos(θ+t),sin(θ+t)]
    t = 0.
    locator = HashedLocator(surf,(0.,2π),SA[-2,-2],SA[2,2],t⁰=t,step=0.25)

    @test isapprox(locator(SA[.3,.4],t),atan(4,3),rtol=1e-4)
    @test isapprox(surf(locator(SA[-1.2,.9],t),t),SA[-4/5,3/5],rtol=1e-4)

    body = ParametricBody(surf,locator)
    @test isapprox(sdf(body,SA[-3.,-4.],t),4.,rtol=1e-2) # outside hash

    t = 0.5; update!(body,t)
    d,n,V = measure(body,SA[-.75,1],t)
    @test d ≈ 0.25
    @test isapprox(n,SA[-3/5, 4/5],rtol=1e-4)
    @test isapprox(V,SA[-4/5,-3/5],rtol=1e-4)
end