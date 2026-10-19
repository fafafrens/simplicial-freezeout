# EDUCATIONAL WALKTHROUGH: one spacetime cell, step by step, for D = 2, 3, 4.
# Run from this folder:  julia --project=. walkthrough.jl
include("SimplicialFreezeOut.jl")
using .SimplicialFreezeOut
using .SimplicialFreezeOut: corners, linear_index, case_index, vertex_values

function walk(D)
    println("\n", "="^70)
    println("D = $D spacetime coordinates = $(D-1) spatial dimensions")

    # 1. The reference geometry: D! simplices and 2^(D+1) cut patterns, data independent.
    ref = ReferenceGrid(Val(D))
    println("reference: $(length(ref.simplices)) simplices, $(length(ref.cases)) hot/cold patterns")
    println("corners of the unit cell, in linear-index order:")
    println("   ", join((string(c.bits) for c in corners(Val(D))), " "))

    # 2. One unit cell with a cooling field T = 1.5 - t. The isotherm Tfo = 1 is the plane t = 0.5.
    Tfo = 1.0
    cornerT = map(c -> 1.5 - c.bits[1], corners(Val(D)))        # value at each corner
    cell = Cell(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))
    q = map(c -> 3.0 + sum(k * c.bits[k] for k in 1:D), corners(Val(D)))   # an affine field
    println("corner temperatures: ", cornerT)

    # 3. For each simplex: which vertices are hot, which pieces does that pattern give.
    for (k, simplex) in enumerate(ref.simplices)
        T = vertex_values(cornerT, simplex)
        is_hot = T .>= Tfo
        pieces = ref.cases[case_index(is_hot)]
        println(" simplex $k: vertices ", Tuple(linear_index(c) for c in simplex),
                "  hot ", Int.(is_hot), "  → $(length(pieces)) piece(s)",
                k <= 2 ? "  crossed edges $(pieces)" : "")
    end

    # 4. The pieces themselves, with oriented area covectors and the interpolated field.
    elements = cell_isosurface(cornerT, cell, Tfo; fields=(q=q,), reference=ref)
    total = sum(e.dsigma for e in elements)
    println("elements: $(length(elements)); total dΣ = $total  (expected (1, 0, …): hot at t < 0.5)")
    for (k, e) in enumerate(Iterators.take(elements, 2))
        println(" element $k: vertices ", e.vertices)
        println("            centroid ", e.centroid, "  dΣ ", e.dsigma, "  q ", e.fields.q)
    end
    @assert total[1] ≈ 1 && all(abs(total[k]) < 1e-12 for k in 2:D)
    @assert all(e.centroid[1] ≈ 0.5 for e in elements)
    @assert all(e.fields.q ≈ 3.0 + sum(k * e.centroid[k] for k in 1:D) for e in elements)

    # 5. The same cell as a fold: integrate a constant current J = (1, 0, …) without storing pieces.
    flux = cell_isosurface((acc, e) -> acc + e.dsigma[1], 0.0, cornerT, cell, Tfo; reference=ref)
    println("flux of J = (1,0,…) through the cell, by fold: ", flux)
end

foreach(walk, 2:4)
