"""
    f(C::NurbsCurve, N::Integer=100)

Plot `recipes` for `NurbsCurve`, plot the `NurbsCurve` and the control points.
"""
@recipe function f(C::NurbsCurve, N::Integer=100; add_cp=true, shift=[0.,0.])
    seriestype := :path
    primary := false
    @series begin
        linecolor := :black
        linewidth := 2
        markershape := :none
        c = [C(s,0.0) for s ∈ 0:1/N:1]
        getindex.(c,1).+shift[1],getindex.(c,2).+shift[2]
    end
    @series begin
        linewidth  --> (add_cp ? 1 : 0)
        markershape --> (add_cp ? :circle : :none)
        markersize --> (add_cp ? 4 : 0)
        delete!(plotattributes, :add_cp)
        C.pnts[1,:].+shift[1],C.pnts[2,:].+shift[2]
    end
end

"""
    f(C::ParametricBodies, N::Integer=100)

Plot `recipes` for `ParametricBody`.
"""
@recipe function f(b::ParametricBody, time=0, N::Integer=100; shift=[0.,0.])
    seriestype := :path
    primary := false
    @series begin
        linecolor := :black
        linewidth := 2
        markershape := :none
        c = [-b.map(-b.surf(s,time),time) for s ∈ range(b.locate.lims...;length=N)]
        getindex.(c,1).+shift[1],getindex.(c,2).+shift[2]
    end
end