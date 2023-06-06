using ParametricBodies
using Test

using StaticArrays
@testset "Rotating circle test" begin
    surf(θ,t) = SA[cos(θ+t),sin(θ+t)]
    locate(x::SVector{2},t) = atan(x[2],x[1])-t
    body = ParametricBodies.ParametricBody(surf,locate)

    @test body.surf(body.locate(SA[4.,3.],1),1) == SA[4/5,3/5]
    @test body.fast_d²(SA[3.,-4.],1) == 16.

    n = ParametricBodies.norm_dir(body.surf,π/2,0)
    @test n/√(n'*n) ≈ SA[0,1]

    @test ParametricBodies.sdf(body,SA[-.3,-.4],2.) == -0.5

    d,n,V = ParametricBodies.measure(body,SA[-.75,1],4.)
    @test d ≈ 0.25
    @test n ≈ SA[-3/5, 4/5]
    @test V ≈ SA[-4/5,-3/5]
end