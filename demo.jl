# Usage examples.  Run from this folder:  julia --project=. demo.jl
include("SimplicialFreezeOut.jl")
using .SimplicialFreezeOut, StaticArrays, LinearAlgebra

# --- A. Full 2+1D spacetime grid (t, x, y) with fluid fields ------------------
t = range(0.0, 1.0; length=9); x = range(-1.0, 1.0; length=17); y = x
T  = [1.3 - ti - 0.15 * (xi^2 + yi^2) for ti in t, xi in x, yi in y]
ux = [0.1xi for ti in t, xi in x, yi in y]
uy = [0.1yi for ti in t, xi in x, yi in y]
surface = isosurface(T, (t, x, y), 0.8; fields=(ux=ux, uy=uy))
println("A. 2+1D grid: ", length(surface), " oriented triangles")
e = first(surface)
println("   first triangle: vertices ", e.vertices)
println("                   dΣ ", e.dsigma, "  fields ", e.fields)
# Centroid quadrature of a surface integral, e.g. the flux of the velocity field.
flux = integrate_surface(surface, e -> e.dsigma[1] + e.fields.ux * e.dsigma[2] + e.fields.uy * e.dsigma[3])
println("   ∮ u·dΣ (u⁰ ≈ 1) = ", flux)

# --- B. One 3+1D cell: the exact isotherm t = 0.37 inside a unit hypercube ---
ref4 = ReferenceGrid(Val(4))                      # build once, reuse for every cell
cornerT = map(c -> c.bits[1] * 1.0, SimplicialFreezeOut.corners(Val(4)))   # T = t at the corners
cell = Cell((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
tets = cell_isosurface(cornerT, cell, 0.37; reference=ref4)
println("B. one 4D cell: ", length(tets), " tetrahedra, total dΣ = ", sum(e.dsigma for e in tets), " (expected (-1,0,0,0))")

# --- C. Streaming from a hydro code: two time slices at a time ---------------
# A fake hydro state: an array of state vectors (energy density, ux, uy) per slice.
state(ti) = [SVector(1.3 - ti - 0.15 * (xi^2 + yi^2), 0.1xi, 0.1yi) for xi in x, yi in y]
temperature(u) = u[1]                              # the "equation of state"
ref3 = ReferenceGrid(Val(3))
streamed = element_vector(Val(3), (ux=nothing,))   # one vector collects all steps
u_prev = state(t[1])
for k in 2:length(t)
    u = state(t[k])
    for I in CartesianIndices(size(u) .- 1)
        cell_isosurface!(streamed, slab_corners(temperature, u_prev, u, I), Cell((t[k-1], t[k]), (x, y), I), 0.8;
                         fields=(ux=slab_corners(v -> v[2], u_prev, u, I),), reference=ref3)
    end
    global u_prev = u
end
println("C. streamed over time slices: ", length(streamed), " triangles (same surface as A: ",
        length(streamed) == length(surface), ")")

# --- D. An integral accumulated on the fly, never storing the surface --------
# Flux of the current J = (1, 0, 0) through the whole surface, cell by cell, with the fold.
J = SVector(1.0, 0.0, 0.0)
total_flux = 0.0
u_prev = state(t[1])
for k in 2:length(t)
    u = state(t[k])
    for I in CartesianIndices(size(u) .- 1)
        global total_flux = cell_isosurface(slab_corners(temperature, u_prev, u, I), Cell((t[k-1], t[k]), (x, y), I), 0.8;
                                            reference=ref3) do acc, e
            acc + dot(J, e.dsigma)
        end + total_flux
    end
    global u_prev = u
end
println("D. flux of J = (1,0,0) by fold: ", total_flux, "  (vs. summed from A: ", sum(e.dsigma[1] for e in surface), ")")
