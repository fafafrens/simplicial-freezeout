# Compare SimplicialFreezeOut with Cornelius (P. Huovinen, H. Petersen,
# arXiv:1206.3371) on the same 4D field: agreement of the integrated dΣ_μ and
# of the enclosed volume, number of elements, and extraction time.
# Run from the repository root:  julia --project=. benchmark/compare_cornelius.jl
# Requires cornelius.f90 (not included, see below) and gfortran; the library is
# compiled on first use.
using SimplicialFreezeOut, StaticArrays, LinearAlgebra, Printf, Libdl

const CORNELIUS_SRC = joinpath(@__DIR__, "cornelius.f90")
isfile(CORNELIUS_SRC) || error("""
    cornelius.f90 not found. Cornelius is not distributed with this repository:
    get it from P. Huovinen and H. Petersen, arXiv:1206.3371 (cite that paper
    when using it) and place cornelius.f90 next to this script.""")
const LIBCORNELIUS = joinpath(@__DIR__, "libcornelius.dylib")
isfile(LIBCORNELIUS) || run(`gfortran -O2 -shared -fPIC -o $LIBCORNELIUS $CORNELIUS_SRC`)
const CORNELIUS = dlsym(dlopen(LIBCORNELIUS), :cornelius_)

function cornelius!(E0, cube, dt, dx, dy, dz, dSigma, Vmid, nsurf, nambi, ndisc)
    ccall(CORNELIUS, Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Float64},
           Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
          E0, cube, dSigma, nsurf, Vmid, dt, dx, dy, dz, nambi, ndisc)
    nsurf[]
end

# --- 1. one cube, T = 1.5 - t, Tfo = 1: expect dSigma = (1,0,0,0), centroid t = 0.5
cube = [1.5 - t for t in 0:1, x in 0:1, y in 0:1, z in 0:1]
dS = zeros(4, 8); Vm = zeros(4, 8); ns = Ref{Int32}(0); na = Ref{Int32}(0); nd = Ref{Int32}(0)
n = cornelius!(1.0, cube, 1.0, 1.0, 1.0, 1.0, dS, Vm, ns, na, nd)
println("Cornelius: ", n, " element(s), dSigma=", dS[:, 1], " centroid=", Vm[:, 1])
simplices = surface_simplices(cube, (0.0,0.0,0.0,0.0), (1.0,1.0,1.0,1.0), 1.0)
println("Simplicial: ", length(simplices), " simplices, Σ dSigma=", sum(e.dsigma for e in simplices),
        " area-weighted centroid=", sum(norm(e.dsigma) * e.centroid for e in simplices) / sum(norm(e.dsigma) for e in simplices))

# --- 2. same 4D sphere field for both
n = 13; x = collect(range(-1.0, 1.0; length=n)); h = step(range(-1.0, 1.0; length=n))
T = [1.0 - (t^2 + a^2 + b^2 + c^2) for t in x, a in x, b in x, c in x]
Tfo = 0.5137; R = sqrt(1 - Tfo); exact = pi^2/2 * R^4

function run_cornelius(T, x, h, Tfo)
    cube = zeros(2, 2, 2, 2); dS = zeros(4, 8); Vm = zeros(4, 8)
    ns = Ref{Int32}(0); na = Ref{Int32}(0); nd = Ref{Int32}(0)
    count = 0; cells = 0; total = zeros(4); vol = 0.0
    for I in CartesianIndices(size(T) .- 1)
        for J in CartesianIndices((2, 2, 2, 2)); cube[J] = T[I + J - oneunit(J)]; end
        (minimum(cube) <= Tfo <= maximum(cube)) || continue
        k = cornelius!(Tfo, cube, h, h, h, h, dS, Vm, ns, na, nd)
        k == 0 && continue
        cells += 1
        origin = SVector(x[I[1]], x[I[2]], x[I[3]], x[I[4]])
        for s in 1:k
            count += 1
            d = SVector{4}(dS[:, s]); c = origin + SVector{4}(Vm[:, s])
            total .+= d; vol += dot(c, d)
        end
    end
    (; count, cells, total, vol = vol / 4)
end
function run_simplicial(T, x, Tfo)
    simplices = surface_simplices(T, (x, x, x, x), Tfo)
    (; count = length(simplices), total = sum(e.dsigma for e in simplices), vol = sum(dot(e.centroid, e.dsigma) for e in simplices) / 4)
end

rc = run_cornelius(T, x, h, Tfo); rs = run_simplicial(T, x, Tfo)       # warm up
tc = minimum(@elapsed(run_cornelius(T, x, h, Tfo)) for _ in 1:5)
ts = minimum(@elapsed(run_simplicial(T, x, Tfo)) for _ in 1:5)
println()
@printf("%-12s %9s %10s %12s %14s %10s\n", "", "elements", "time", "|Σ dSigma|", "volume", "exact")
@printf("%-12s %9d %8.3fs %12.1e %14.5f %10.5f\n", "Cornelius", rc.count, tc, norm(rc.total), rc.vol, exact)
@printf("%-12s %9d %8.3fs %12.1e %14.5f %10.5f\n", "Simplicial", rs.count, ts, norm(rs.total), rs.vol, exact)
println("crossing cells: ", rc.cells, "   Cornelius ambiguous faces: ", na[], "  disconnected: ", nd[])

# --- 3. a field that is not affine inside cells --------------------------------
# T = 1 - Σx² above is a sum of one-variable functions, so its multilinear
# interpolant is affine in every cell: both methods cut each cell with the same
# plane and agree to machine precision. A Gaussian with a cross term is not
# affine inside cells; the two methods then differ per cell at O(h²) and
# converge to the same surface integrals.
gaussian(x) = [exp(-(t^2 + a^2 + b^2 + c^2) + 0.6t * a) for t in x, a in x, b in x, c in x]
const Tfo_g = exp(-0.4863)
weight(p) = exp(0.8p[1] + 0.5p[2] - 0.3p[3])           # a smooth test integrand, ∫ weight dΣ₀

# Per crossing cell: (dΣ, centroid) from Cornelius, from merged simplices, and the
# integral of the unmerged simplices.
function per_cell(T, x, h, Tfo)
    cube = zeros(2, 2, 2, 2); dS = zeros(4, 8); Vm = zeros(4, 8)
    ns = Ref{Int32}(0); na = Ref{Int32}(0); nd = Ref{Int32}(0)
    ref = ReferenceGrid(Val(4))
    corn = Tuple{SVector{4,Float64},SVector{4,Float64}}[]; merged = similar(corn); unmerged = 0.0
    for I in CartesianIndices(size(T) .- 1)
        for J in CartesianIndices((2, 2, 2, 2)); cube[J] = T[I + J - oneunit(J)]; end
        (minimum(cube) <= Tfo <= maximum(cube)) || continue
        k = cornelius!(Tfo, cube, h, h, h, h, dS, Vm, ns, na, nd)
        k == 1 || error("several Cornelius elements in one cell; not handled here")
        origin = SVector(x[I[1]], x[I[2]], x[I[3]], x[I[4]])
        push!(corn, (SVector{4}(dS[:, 1]), origin + SVector{4}(Vm[:, 1])))
        cT = cell_corners(T, I); cell = Cell((x, x, x, x), I)
        e = surface_element(cT, cell, Tfo; reference=ref)
        push!(merged, (e.dsigma, e.centroid))
        unmerged = surface_simplices((a, s) -> a + weight(s.centroid) * s.dsigma[1], unmerged, cT, cell, Tfo; reference=ref)
    end
    integral(v) = sum(weight(c) * d[1] for (d, c) in v)
    (; corn, merged, Ic = integral(corn), Im = integral(merged), Iu = unmerged)
end
function count_cornelius(T, h, Tfo)                     # extraction only, for timing
    cube = zeros(2, 2, 2, 2); dS = zeros(4, 8); Vm = zeros(4, 8)
    ns = Ref{Int32}(0); na = Ref{Int32}(0); nd = Ref{Int32}(0); n = 0
    for I in CartesianIndices(size(T) .- 1)
        for J in CartesianIndices((2, 2, 2, 2)); cube[J] = T[I + J - oneunit(J)]; end
        (minimum(cube) <= Tfo <= maximum(cube)) || continue
        n += cornelius!(Tfo, cube, h, h, h, h, dS, Vm, ns, na, nd)
    end
    n
end

println("\nGaussian field, not affine inside cells")
x41 = collect(range(-1.0, 1.0; length=41)); h41 = x41[2] - x41[1]; T41 = gaussian(x41)
count_cornelius(T41, h41, Tfo_g); surface_elements(T41, (x41, x41, x41, x41), Tfo_g); surface_simplices(T41, (x41, x41, x41, x41), Tfo_g)
tc = minimum(@elapsed(count_cornelius(T41, h41, Tfo_g)) for _ in 1:3)
tm = minimum(@elapsed(surface_elements(T41, (x41, x41, x41, x41), Tfo_g)) for _ in 1:3)
tu = minimum(@elapsed(surface_simplices(T41, (x41, x41, x41, x41), Tfo_g)) for _ in 1:3)
@printf("41^4 grid   %-28s %9s %9s\n", "", "elements", "time")
@printf("            %-28s %9d %8.3fs\n", "Cornelius", count_cornelius(T41, h41, Tfo_g), tc)
@printf("            %-28s %9d %8.3fs\n", "surface_elements (merged)", length(surface_elements(T41, (x41, x41, x41, x41), Tfo_g)), tm)
@printf("            %-28s %9d %8.3fs\n", "surface_simplices", length(surface_simplices(T41, (x41, x41, x41, x41), Tfo_g)), tu)

results = Dict{Int,Any}()
for n in (41, 81)
    xn = collect(range(-1.0, 1.0; length=n)); hn = xn[2] - xn[1]
    results[n] = (h = hn, r = per_cell(gaussian(xn), xn, hn, Tfo_g))
end
r41 = results[41].r
Δ = [norm(a[1] - b[1]) for (a, b) in zip(r41.corn, r41.merged)]
@printf("per cell on 41^4, Cornelius vs merged: mean |ΔdΣ|/|dΣ| = %.1e (area-weighted)\n",
        sum(Δ) / sum(norm(b[1]) for b in r41.merged))
q = (results[41].h / results[81].h)^2
exact = results[81].r.Iu + (results[81].r.Iu - r41.Iu) / (q - 1)   # Richardson, error ∝ h²
@printf("∫ weight dΣ₀ extrapolated to h → 0: %.7f; relative errors:\n", exact)
@printf("  %-7s %11s %11s %11s\n", "h", "Cornelius", "merged", "simplices")
for n in (41, 81)
    r = results[n].r
    @printf("  %-7.4f %11.1e %11.1e %11.1e\n", results[n].h, abs(r.Ic / exact - 1), abs(r.Im / exact - 1), abs(r.Iu / exact - 1))
end
