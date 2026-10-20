using SimplicialFreezeOut
using Test, StaticArrays, LinearAlgebra

@testset "Simplicial freeze-out in 1, 2 and 3 spatial dimensions" begin
    for D in 2:4
        reference = ReferenceGrid(Val(D))
        @test length(reference.simplices) == factorial(D)
        @test length(reference.cases) == 2^(D+1)
        @test all(length(s)==D+1 for s in reference.simplices)

        # The exact isotherm is t=0.37. Hot region is at t>0.37:
        # its outward covector must point toward negative time.
        axes = ntuple(k -> k==1 ? [0.0,0.5,1.0] : [0.0,0.4,1.0],D)
        sizes = map(length, axes)
        temperature = [axes[1][I[1]] for I in CartesianIndices(sizes)]
        affine_field = [2.0 + sum(k*axes[k][I[k]] for k in 1:D)
                        for I in CartesianIndices(sizes)]
        simplices = surface_simplices(temperature, axes, 0.37; fields=(q=affine_field,))
        @test !isempty(simplices)
        @test all(abs(e.centroid[1] - 0.37) < 1e-12 for e in simplices)
        for e in simplices
            @test isapprox(e.fields.q,
                2.0+sum(k*e.centroid[k] for k in 1:D); atol=1e-11)
        end
        area = ntuple(mu -> sum(e.dsigma[mu] for e in simplices),D)
        @test isapprox(area[1],-1.0;atol=1e-10)
        @test all(abs(area[mu]) < 1e-10 for mu in 2:D)

        # Oblique affine temperature: every produced point is on the
        # corresponding exact plane and is oriented hot -> cold.
        g = ntuple(k -> (-1)^k * (0.2+k*0.11), D)
        temp2 = [sum(g[k]*axes[k][I[k]] for k in 1:D)
                 for I in CartesianIndices(sizes)]
        level = (minimum(temp2)+maximum(temp2))/2
        simplices2 = surface_simplices(temp2, axes, level)
        @test !isempty(simplices2)
        for e in simplices2
            @test all(abs(sum(g[mu]*v[mu] for mu in 1:D)-level) < 1e-10
                      for v in e.vertices)
            @test sum(e.dsigma[mu]*g[mu] for mu in 1:D) <= 1e-12
        end
    end
end

@testset "Grid node exactly on the isotherm" begin
    # T = 1 - r^2 on a grid that contains nodes with r^2 = 1/2 exactly.
    x = collect(range(-1.0, 1.0; length=25))
    T = [1.0 - (xi^2 + yi^2 + zi^2) for xi in x, yi in x, zi in x]
    simplices = surface_simplices(T, (x, x, x), 0.5)
    @test all(allunique(e.vertices) for e in simplices)
    @test all(sum(e.centroid .* e.dsigma) > 0 for e in simplices)   # outward
    # Closed surface: every edge is shared by exactly two triangles.
    edges = Dict{Any,Int}()
    for e in simplices, drop in 1:3
        key = Tuple(sort([round.(e.vertices[i]; digits=8) for i in 1:3 if i != drop]))
        edges[key] = get(edges, key, 0) + 1
    end
    @test all(==(2), values(edges))
    # Divergence theorem: (1/3) ∮ x·dΣ ≈ volume of the ball of radius √0.5.
    volume = sum(sum(e.centroid .* e.dsigma) for e in simplices) / 3
    @test isapprox(volume, 4π/3 * 0.5^1.5; rtol=0.02)
end

@testset "Single hypercube" begin
    # One unit cube, T = 1.5 - t, isotherm t = 0.5 (hot at t < 0.5).
    T = [1.5 - t for t in 0:1, x in 0:1, y in 0:1]          # 2×2×2 array
    q = [3.0 + t + 2x + 3y for t in 0:1, x in 0:1, y in 0:1]
    simplices = surface_simplices(T, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0; fields=(q=q,))
    @test length(simplices) == 8
    @test all(abs(e.centroid[1] - 0.5) < 1e-12 for e in simplices)
    @test sum(e.dsigma[1] for e in simplices) ≈ 1.0
    @test all(e.fields.q ≈ 3.0 + sum((1,2,3) .* e.centroid) for e in simplices)
    # Corner tuples give the same result as the 2×2×2 array.
    @test surface_simplices(Tuple(T), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0) ==
          surface_simplices(T, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0)
    # The grid driver on the same cube agrees.
    @test surface_simplices(T, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]), 1.0; fields=(q=q,)) == simplices
    @test isempty(surface_simplices(T, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2.0))
end

@testset "Streaming from two time slices" begin
    t = collect(range(0.0, 1.0; length=6)); x = collect(range(-1.0, 1.0; length=9)); y = x
    # Hydro-like state: an array of state vectors (energy density, ux, uy) per time slice.
    state(ti) = [SVector(1.3 - ti - 0.15 * (xi^2 + yi^2), 0.1xi, 0.1yi) for xi in x, yi in y]
    temperature(u) = u[1]^(1/4) * 1.1                     # a fake equation of state
    T = [temperature(state(ti)[i, j]) for ti in t, i in eachindex(x), j in eachindex(y)]
    ux = [0.1xi for ti in t, xi in x, yi in y]
    reference_result = surface_simplices(T, (t, x, y), 0.95; fields=(ux=ux,))

    @test slab_corners((10, 20, 30, 40), (11, 21, 31, 41)) == (10, 11, 20, 21, 30, 31, 40, 41)
    ref = ReferenceGrid(Val(3))
    out = simplex_type(3, (:ux,))[]
    u_prev = state(t[1])
    for k in 2:length(t)
        u = state(t[k])
        for I in CartesianIndices(size(u) .- 1)
            surface_simplices!(out, slab_corners(temperature, u_prev, u, I), Cell((t[k-1], t[k]), (x, y), I), 0.95;
                             fields=(ux=slab_corners(v -> v[2], u_prev, u, I),), reference=ref)
        end
        u_prev = u
    end
    # Same simplices; the two loops visit the cells in a different order.
    bycentroid(v) = sort(v; by=e -> Tuple(e.centroid))
    @test length(out) == length(reference_result)
    @test all(a.vertices == b.vertices && a.dsigma == b.dsigma && a.fields == b.fields
              for (a, b) in zip(bycentroid(out), bycentroid(reference_result)))
end

@testset "Fold, callback and in-place forms agree" begin
    T = Tuple(1.0 - sum(abs2, (t, x, y, z)) for t in 0:1, x in 0:1, y in 0:1, z in 0:1)
    cell = Cell((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    q = Tuple(2.0 + t + x for t in 0:1, x in 0:1, y in 0:1, z in 0:1)
    simplices = surface_simplices(T, cell, 0.5; fields=(q=q,))
    total = sum(e.dsigma for e in simplices)
    # fold with an immutable SVector accumulator
    @test surface_simplices((acc, e) -> acc + e.dsigma, zero(SVector{4,Float64}), T, cell, 0.5) ≈ total
    # fold with a field-weighted integrand
    @test surface_simplices((acc, e) -> acc + e.fields.q * e.dsigma[1], 0.0, T, cell, 0.5; fields=(q=q,)) ≈
          sum(e.fields.q * e.dsigma[1] for e in simplices)
    # callback form sees every element once
    seen = Ref(0)
    surface_simplices(T, cell, 0.5) do e; seen[] += 1; end
    @test seen[] == length(simplices)
    # in-place form appends to an existing vector
    out = simplex_type(4, (:q,))[]; surface_simplices!(out, T, cell, 0.5; fields=(q=q,)); surface_simplices!(out, T, cell, 0.5; fields=(q=q,))
    @test out == [simplices; simplices]
    # a cell the level does not reach returns init unchanged
    @test surface_simplices((acc, e) -> acc + 1, 0, T, cell, 5.0) == 0
end

@testset "Merged simplices" begin
    # The unit square with T = 1.5 - t: two segments merge into one, flat, element.
    square = Tuple(1.5 - t for t in 0:1, x in 0:1)
    m = surface_element(square, Cell((0.0, 0.0), (1.0, 1.0)), 1.0)
    @test m.centroid ≈ [0.5, 0.5] && m.dsigma ≈ [1.0, 0.0] && m.area ≈ 1.0 && flatness(m) ≈ 1.0
    @test iszero(surface_element(square, Cell((0.0, 0.0), (1.0, 1.0)), 5.0))

    # A curved isotherm on a non-uniform 3D grid, with an affine field.
    t = range(0.0, 1.0; length=9); x = [-1.0, -0.7, -0.5, -0.2, 0.0, 0.3, 0.6, 1.0]; y = range(-1.0, 1.0; length=7)
    T = [1.0 - ti - 0.4 * (xi^2 + yi^2) for ti in t, xi in x, yi in y]
    affine(p) = 0.3 + 2p[1] - p[2] + 0.5p[3]
    q = [affine((ti, xi, yi)) for ti in t, xi in x, yi in y]
    simplices = surface_simplices(T, (t, x, y), 0.45; fields=(q=q,))

    for block in (1, 2, (3, 1, 2), 100)
        merged = surface_elements(T, (t, x, y), 0.45; fields=(q=q,), block)
        # dΣ and areas add up exactly, whatever the blocks
        @test sum(e.dsigma for e in merged) ≈ sum(e.dsigma for e in simplices)
        @test sum(e.area for e in merged) ≈ sum(norm(e.dsigma) for e in simplices)
        # an affine field is the field at the merged centroid
        @test all(e.fields.q ≈ affine(e.centroid) for e in merged)
        @test all(0 < flatness(e) <= 1 + 1e-12 for e in merged)
        @test integrate_surface(merged, e -> e.dsigma[1]) ≈ integrate_surface(simplices, e -> e.dsigma[1])
    end
    per_cell = surface_elements(T, (t, x, y), 0.45; fields=(q=q,))
    whole = only(surface_elements(T, (t, x, y), 0.45; fields=(q=q,), block=100))
    @test length(per_cell) < length(simplices)
    @test flatness(whole) < 0.99                       # the whole surface is visibly curved

    # Merging is associative: merging merged cells equals merging every piece at once.
    remerged = foldl(merge_element, per_cell; init=zero(element_type(3, (:q,))))
    @test remerged.centroid ≈ whole.centroid && remerged.dsigma ≈ whole.dsigma &&
          remerged.area ≈ whole.area && remerged.fields.q ≈ whole.fields.q

    # A strict flatness threshold splits curved blocks back into single cells.
    @test length(surface_elements(T, (t, x, y), 0.45; fields=(q=q,), block=100, min_flatness=1.0)) == length(per_cell)
end

println("All simplicial freeze-out geometry tests passed.")
