# Non-allocating method for 2x3 SMatrix solve
using LinearAlgebra
import Base: \
@inline function (\)(a::SMatrix{2,3}, b::SVector{2})
    # columns of a'
    a1,a2 = a[1,:],a[2,:]

    # Q,R decomposition
    r11 = norm(a1)
    q1 = a1/r11
    r12 = q1'*a2
    p = a2-r12*q1
    r22 = norm(p) < eps(r11) ? one(r11) : norm(p)
    q2 = p/r22

    # forward substitution to solve v = R'\b
    v1 = b[1]/r11
    v2 = (b[2]-r12*v1)/r22

    # return solution x = Qv
    return q1*v1+q2*v2
end