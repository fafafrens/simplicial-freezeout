# Usage examples for SimplicialFreezeOut.
# Run from this folder:  julia --project=. example.jl
include("SimplicialFreezeOut.jl")
using .SimplicialFreezeOut, StaticArrays, LinearAlgebra

# A cooling 2+1D fireball: energy-density-like temperature and a radial flow.
temperature_field(t, x, y) = 1.3 - t - 0.15 * (x^2 + y^2)
const Tfo = 0.8
const t = range(0.0, 1.0; length=9)
const x = range(-1.0, 1.0; length=17)
const y = x

# --- 1. A stored spacetime grid ----------------------------------------------
function stored_grid()
    T  = [temperature_field(ti, xi, yi) for ti in t, xi in x, yi in y]
    ux = [0.1xi for ti in t, xi in x, yi in y]
    uy = [0.1yi for ti in t, xi in x, yi in y]
    surface = isosurface(T, (t, x, y), Tfo; fields=(ux=ux, uy=uy))

    e = first(surface)
    println("1. stored grid: ", length(surface), " oriented triangles")
    println("   first element: vertices ", e.vertices)
    println("                  dΣ ", e.dsigma, ", ux ", e.fields.ux, ", uy ", e.fields.uy)

    # Centroid quadrature of a surface integral: flux of the current J = (1, 0, 0).
    flux = integrate_surface(surface, e -> e.dsigma[1])
    println("   ∫ dΣ₀ over the surface = ", flux)
    surface
end

# --- 2. One cell, by hand -----------------------------------------------------
function one_cell()
    # The unit 4D hypercube with T = t: the isotherm T = 0.37 is the plane t = 0.37.
    cornerT = Tuple(Float64(ti) for ti in 0:1, xi in 0:1, yi in 0:1, zi in 0:1)   # 2×2×2×2, time first
    cell = Cell((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    tets = cell_isosurface(cornerT, cell, 0.37)
    println("2. one 4D cell: ", length(tets), " tetrahedra, total dΣ = ",
            sum(e.dsigma for e in tets), "  (expected (-1, 0, 0, 0): hot is t ≥ 0.37)")
end

# --- 3. Streaming from a hydro code, two time slices at a time ---------------
# A fake hydro state: per time slice, an array of named state tuples.
state(ti) = [(T=temperature_field(ti, xi, yi), ux=0.1xi, uy=0.1yi) for xi in x, yi in y]

function streamed()
    ref = ReferenceGrid(Val(3))                          # build once, reuse
    out = element_vector(Val(3), (ux=nothing, uy=nothing))
    u_prev = state(t[1])
    for k in 2:length(t)
        u = state(t[k])
        for I in CartesianIndices(size(u) .- 1)
            cell_isosurface!(out, slab_corners(s -> s.T, u_prev, u, I),
                             Cell((t[k-1], t[k]), (x, y), I), Tfo;
                             fields=(ux=slab_corners(s -> s.ux, u_prev, u, I),
                                     uy=slab_corners(s -> s.uy, u_prev, u, I)),
                             reference=ref)
        end
        u_prev = u
    end
    println("3. streamed over time slices: ", length(out), " triangles")
    out
end

# --- 4. An integral accumulated on the fly, never storing the surface --------
# The fold form: cell_isosurface(f, init, cornerT, cell, Tfo) returns
# f(…f(f(init, e₁), e₂)…, eₙ) over the pieces of the cell.
function on_the_fly()
    ref = ReferenceGrid(Val(3))
    momenta = range(0.5, 2.0; length=4)
    spectrum = zeros(length(momenta))        # a toy "spectrum": Σ p·dΣ₀ exp(-p u⁰ / Tfo)
    flux = 0.0
    u_prev = state(t[1])
    for k in 2:length(t)
        u = state(t[k])
        for I in CartesianIndices(size(u) .- 1)
            cornerT = slab_corners(s -> s.T, u_prev, u, I)
            cell = Cell((t[k-1], t[k]), (x, y), I)
            fields = (ux=slab_corners(s -> s.ux, u_prev, u, I), uy=slab_corners(s -> s.uy, u_prev, u, I))

            # scalar accumulator
            flux = cell_isosurface((acc, e) -> acc + e.dsigma[1], flux, cornerT, cell, Tfo; reference=ref)

            # array accumulator, with a do block
            cell_isosurface(spectrum, cornerT, cell, Tfo; fields, reference=ref) do S, e
                u0 = sqrt(1 + e.fields.ux^2 + e.fields.uy^2)
                for (j, p) in enumerate(momenta)
                    S[j] += p * e.dsigma[1] * exp(-p * u0 / Tfo)
                end
                S
            end
        end
        u_prev = u
    end
    println("4. on the fly: ∫ dΣ₀ = ", flux)
    println("   toy spectrum at p = ", collect(momenta), ": ", spectrum)
    flux
end

surface = stored_grid()
one_cell()
out = streamed()
flux = on_the_fly()

# The three ways of extracting the 2+1D surface agree.
@assert length(out) == length(surface)
@assert flux ≈ sum(e.dsigma[1] for e in surface)
println("checks passed: streamed and on-the-fly results match the stored-grid surface")
