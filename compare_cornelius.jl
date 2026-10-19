# Compare SimplicialFreezeOut with Cornelius (P. Huovinen, H. Petersen,
# arXiv:1206.3371) on the same 4D field: agreement of the integrated dΣ_μ and
# of the enclosed volume, number of elements, and extraction time.
# Run from this folder:  julia --project=. compare_cornelius.jl
# Requires cornelius.f90 (not included, see below) and gfortran; the library is
# compiled on first use.
isdefined(Main, :SimplicialFreezeOut) || include(joinpath(@__DIR__, "SimplicialFreezeOut.jl"))   # once per session
using .SimplicialFreezeOut, StaticArrays, LinearAlgebra, Printf, Libdl

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
els = cell_isosurface(cube, (0.0,0.0,0.0,0.0), (1.0,1.0,1.0,1.0), 1.0)
println("Simplicial: ", length(els), " elements, Σ dSigma=", sum(e.dsigma for e in els),
        " area-weighted centroid=", sum(norm(e.dsigma) * e.centroid for e in els) / sum(norm(e.dsigma) for e in els))

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
    els = isosurface(T, (x, x, x, x), Tfo)
    (; count = length(els), total = sum(e.dsigma for e in els), vol = sum(dot(e.centroid, e.dsigma) for e in els) / 4)
end

rc = run_cornelius(T, x, h, Tfo); rs = run_simplicial(T, x, Tfo)       # warm up
tc = minimum(@elapsed(run_cornelius(T, x, h, Tfo)) for _ in 1:5)
ts = minimum(@elapsed(run_simplicial(T, x, Tfo)) for _ in 1:5)
println()
@printf("%-12s %9s %10s %12s %14s %10s\n", "", "elements", "time", "|Σ dSigma|", "volume", "exact")
@printf("%-12s %9d %8.3fs %12.1e %14.5f %10.5f\n", "Cornelius", rc.count, tc, norm(rc.total), rc.vol, exact)
@printf("%-12s %9d %8.3fs %12.1e %14.5f %10.5f\n", "Simplicial", rs.count, ts, norm(rs.total), rs.vol, exact)
println("crossing cells: ", rc.cells, "   Cornelius ambiguous faces: ", na[], "  disconnected: ", nd[])
