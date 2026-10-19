# EDUCATIONAL WALKTHROUGH: one Cartesian spacetime cell, step by step.
# Run: julia walkthrough.jl
# No external Julia packages are needed.
include("SimplicialFreezeOut.jl")
using .SimplicialFreezeOut

# Corner labels are 1-based bit codes: bit 0 = time, bit 1 = first spatial
# coordinate, ... This equals the linear index of a 2×…×2 array.
function walk(D)
    @assert D in (2,3,4)
    println("\n", repeat("=", 65))
    println("D=$D spacetime coordinates = $(D-1) spatial dimensions")
    ref = ReferenceGrid(Val(D))

    # ONE unit spacetime cell. A strictly cooling field T=1.5-t.
    # Isotherm Tfo=1 is the hyperplane t=0.5.
    axes = ntuple(_ -> [0.0,1.0], D)
    Tfo = 1.0
    temp = [1.5 - axes[1][I[1]]
            for I in CartesianIndices(ntuple(_ -> 2, D))]
    q = [3.0 + sum(k*axes[k][I[k]] for k in 1:D)
         for I in CartesianIndices(ntuple(_ -> 2, D))]

    println("One cube/hypercube contains $(2^D) grid vertices.")
    println("Reference subdivision contains $(length(ref.simplices)) simplices.")
    for (k,s) in enumerate(ref.simplices)
        corners = [SimplicialFreezeOut.linear_index(v) for v in s]
        is_hot = Tuple(temp[c] >= Tfo for c in corners)
        pieces = ref.cases[SimplicialFreezeOut.case_index(is_hot)]
        println(" simplex $k corners=", Tuple(corners),
                " hot=", Int.(is_hot),
                " output elements=", length(pieces))
        if k <= 3
            println("   crossing-edge patterns: ", pieces)
        end
    end

    elements = isosurface(temp, axes, Tfo; fields=(q=q,))
    total = ntuple(mu -> sum(e.dsigma[mu] for e in elements), D)
    println("Freeze-out elements constructed: ", length(elements))
    println("Total integrated coordinate area covector: ", total)
    println("Expected: (1,0,...,0), since hot t<0.5 and cold t>0.5")
    @assert abs(total[1] - 1) < 1e-10
    @assert all(abs(total[k]) < 1e-10 for k in 2:D)
    @assert all(abs(e.centroid[1] - 0.5) < 1e-10 for e in elements)
    for (k,e) in enumerate(Iterators.take(elements, 2))
        println(" example element $k vertices=", e.vertices)
        println("   centroid=", e.centroid, " dsigma=", e.dsigma)
        println("   affine scalar field q(centroid)=", e.fields.q)
    end
end

for D in 2:4
    walk(D)
end
