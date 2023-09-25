struct LinearInterp{A<:AbstractArray} <: Function
    pnts::A
end
function (l::LinearInterp)(s,t)
    n = size(l.pnts,2)
    i = min(1+floor(Int,s*(n-1)),n-1)
    r = 1+s*(n-1)-i
    l.pnts[:,i]*(1-r)+l.pnts[:,i+1]*r # This is will allocate unless pnts::StaticArray
end

using StaticArrays
square = LinearInterp(SA[5 5 0 -5 -5 -5  0  5 5
                         0 5 5  5  0 -5 -5 -5 0])
@assert square(0.5,0)==[-5,0]

# Does it work with KernelAbstractions?
using CUDA; @assert CUDA.functional()
using KernelAbstractions
@kernel function _test!(a::AbstractArray,l::LinearInterp)
    # Map index to physical space
    I = @index(Global)
    s = (I-1)/(length(a)-1)
    q = l(s,0)
    a[I] = q'*q
end
test!(a,l)=_test!(get_backend(a),64)(a,l,ndrange=length(a))

a = CUDA.zeros(64)
test!(a,square)
a|>Array # yes!

# Does it work with ParametricBodies?
using ParametricBodies
body = ParametricBody(square, (0,1); T=Float32, mem=CUDA.CuArray);
CUDA.@allowscalar @assert all(measure(body,[1,2],0) .≈ [-3,[0,1],[0,0]])
CUDA.@allowscalar @assert all(measure(body,[8,2],0) .≈ [3,[1,0],[0,0]]) # yes!
