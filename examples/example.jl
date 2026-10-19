# Usage examples for SimplicialFreezeOut, ending with a figure of the surface.
# Run from the repository root:
#   julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
#   julia --project=examples examples/example.jl
include(joinpath(@__DIR__, "..", "SimplicialFreezeOut.jl"))
using .SimplicialFreezeOut, StaticArrays, LinearAlgebra
using CairoMakie

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
    elements = SimplicialFreezeOut.isosurface(T, (t, x, y), Tfo; fields=(ux=ux, uy=uy))

    e = first(elements)
    println("1. stored grid: ", length(elements), " oriented triangles")
    println("   first element: vertices ", e.vertices)
    println("                  dΣ ", e.dsigma, ", ux ", e.fields.ux, ", uy ", e.fields.uy)

    # Centroid quadrature of a surface integral: flux of the current J = (1, 0, 0).
    flux = integrate_surface(elements, e -> e.dsigma[1])
    println("   ∫ dΣ₀ over the surface = ", flux)
    elements
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

grid_elements = stored_grid()
one_cell()
out = streamed()
flux = on_the_fly()

# The three ways of extracting the 2+1D surface agree.
@assert length(out) == length(grid_elements)
@assert flux ≈ sum(e.dsigma[1] for e in grid_elements)
println("checks passed: streamed and on-the-fly results match the stored-grid surface")

# --- 5. A figure of the freeze-out surface ------------------------------------
# Left: the surface in (x, y, t) as a triangle mesh, coloured by the flow speed
# interpolated at each triangle. Right: freeze-out time against radius for every
# element, on top of the exact isotherm t(r) = 1.3 - Tfo - 0.15 r².
function plot_surface(path)
    tf = range(0.0, 0.6; length=31); xf = range(-1.9, 1.9; length=61); yf = xf
    T  = [temperature_field(ti, xi, yi) for ti in tf, xi in xf, yi in yf]
    ux = [0.1xi for ti in tf, xi in xf, yi in yf]
    uy = [0.1yi for ti in tf, xi in xf, yi in yf]
    elements = isosurface(T, (tf, xf, yf), Tfo; fields=(ux=ux, uy=uy))

    # One mesh vertex per triangle corner, so each triangle can carry its own colour.
    points = [Point3f(v[2], v[3], v[1]) for e in elements for v in e.vertices]   # (x, y, t)
    faces  = [CairoMakie.GeometryBasics.TriangleFace(3k - 2, 3k - 1, 3k) for k in eachindex(elements)]
    speed  = [hypot(e.fields.ux, e.fields.uy) for e in elements for _ in 1:3]

    # Sequential single-hue ramp, light (slow) to dark (fast).
    blues = cgrad(["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
    ink, muted = "#2b2b29", "#6b6b66"

    fig = Figure(size=(1200, 540), backgroundcolor="#fcfcfb", fontsize=15, figure_padding=(36, 16, 12, 12))
    ax3 = Axis3(fig[1, 1]; xlabel="x", ylabel="y", zlabel="t", azimuth=0.35π, elevation=0.17π,
                aspect=(1, 1, 0.55), limits=(nothing, nothing, (0, 0.52)),
                title="Freeze-out surface T = $Tfo", titlecolor=ink, titlealign=:left,
                xgridcolor=(:black, 0.06), ygridcolor=(:black, 0.06), zgridcolor=(:black, 0.06),
                xlabelcolor=muted, ylabelcolor=muted, zlabelcolor=muted)
    m = mesh!(ax3, points, faces; color=speed, colormap=blues, shading=true)
    Colorbar(fig[1, 2], m; label="flow speed |u|", labelcolor=muted, height=Relative(0.6))

    r  = [hypot(e.centroid[2], e.centroid[3]) for e in elements]
    tc = [e.centroid[1] for e in elements]
    rr = range(0, maximum(r); length=200)
    ax2 = Axis(fig[1, 3]; xlabel="r", ylabel="t", title="Freeze-out time against radius",
               titlecolor=ink, titlealign=:left, xlabelcolor=muted, ylabelcolor=muted,
               xgridcolor=(:black, 0.06), ygridcolor=(:black, 0.06), topspinevisible=false, rightspinevisible=false)
    lines!(ax2, rr, 1.3 .- Tfo .- 0.15 .* rr .^ 2; color=(ink, 0.5), linewidth=8, label="exact isotherm")
    scatter!(ax2, r, tc; color=blues[0.75], markersize=3, label="element centroids")
    axislegend(ax2; position=:lb, framevisible=false, labelcolor=muted)

    colsize!(fig.layout, 1, Relative(0.55))
    save(path, fig; px_per_unit=2)
    println("5. figure: ", length(elements), " triangles drawn, saved to ", path)
end

plot_surface(joinpath(@__DIR__, "freezeout_surface.png"))
