include("SimplicialFreezeOut.jl")
using .SimplicialFreezeOut

# Example A: 2+1-dimensional spacetime grid (t,x,y).
t = collect(range(0.0, 1.0; length=9))
x = collect(range(-1.0, 1.0; length=17))
y = collect(range(-1.0, 1.0; length=17))
T = [1.3 - ti - 0.15*(xi^2 + yi^2) for ti in t, xi in x, yi in y]
ux = [0.1*xi for ti in t, xi in x, yi in y]
uy = [0.1*yi for ti in t, xi in x, yi in y]

surface3 = isosurface(T, (t, x, y), 0.8; fields=(ux=ux, uy=uy))
println("2+1D: ", length(surface3), " oriented triangles")
if !isempty(surface3)
    e = surface3[1]
    println("  first triangle: ",e.vertices)
    println("  first oriented area covector: ",e.dsigma)
    println("  interpolated fields: ",e.fields)
end

# Example B: 3+1-dimensional spacetime grid (t,x,y,z).
# The exact isotherm t = 0.37 should have integrated orientation
# dSigma = (-1, 0, 0, 0) over the unit spatial cube.
axes4 = ntuple(_ -> [0.0, 0.5, 1.0], 4)
T4 = [ti for ti in axes4[1], xi in axes4[2],
               yi in axes4[3], zi in axes4[4]]
surface4 = isosurface(T4, axes4, 0.37)
flux4 = ntuple(mu -> sum(e.dsigma[mu] for e in surface4), 4)
println("3+1D: ", length(surface4), " oriented tetrahedra")
println("  exact constant-normal check: ",flux4, " (expected (-1,0,0,0))")
J = (1.0,0.0,0.0,0.0)
println("  integrated constant-current flux: ",
    integrate_surface(surface4, e -> sum(J[mu]*e.dsigma[mu] for mu in 1:4)))
