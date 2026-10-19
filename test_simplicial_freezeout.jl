using Test
include("SimplicialFreezeOut.jl")
using .SimplicialFreezeOut

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
        elements = isosurface(temperature, axes, 0.37; fields=(q=affine_field,))
        @test !isempty(elements)
        @test all(abs(e.centroid[1] - 0.37) < 1e-12 for e in elements)
        for e in elements
            @test isapprox(e.fields.q,
                2.0+sum(k*e.centroid[k] for k in 1:D); atol=1e-11)
        end
        area = ntuple(mu -> sum(e.dsigma[mu] for e in elements),D)
        @test isapprox(area[1],-1.0;atol=1e-10)
        @test all(abs(area[mu]) < 1e-10 for mu in 2:D)

        # Oblique affine temperature: every produced point is on the
        # corresponding exact plane and is oriented hot -> cold.
        g = ntuple(k -> (-1)^k * (0.2+k*0.11), D)
        temp2 = [sum(g[k]*axes[k][I[k]] for k in 1:D)
                 for I in CartesianIndices(sizes)]
        level = (minimum(temp2)+maximum(temp2))/2
        elements2 = isosurface(temp2, axes, level)
        @test !isempty(elements2)
        for e in elements2
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
    elements = isosurface(T, (x, x, x), 0.5)
    @test all(allunique(e.vertices) for e in elements)
    @test all(sum(e.centroid .* e.dsigma) > 0 for e in elements)   # outward
    # Closed surface: every edge is shared by exactly two triangles.
    edges = Dict{Any,Int}()
    for e in elements, drop in 1:3
        key = Tuple(sort([round.(e.vertices[i]; digits=8) for i in 1:3 if i != drop]))
        edges[key] = get(edges, key, 0) + 1
    end
    @test all(==(2), values(edges))
    # Divergence theorem: (1/3) ∮ x·dΣ ≈ volume of the ball of radius √0.5.
    volume = sum(sum(e.centroid .* e.dsigma) for e in elements) / 3
    @test isapprox(volume, 4π/3 * 0.5^1.5; rtol=0.02)
end

@testset "Single hypercube" begin
    # One unit cube, T = 1.5 - t, isotherm t = 0.5 (hot at t < 0.5).
    T = [1.5 - t for t in 0:1, x in 0:1, y in 0:1]          # 2×2×2 array
    q = [3.0 + t + 2x + 3y for t in 0:1, x in 0:1, y in 0:1]
    elements = cell_isosurface(T, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0; fields=(q=q,))
    @test length(elements) == 8
    @test all(abs(e.centroid[1] - 0.5) < 1e-12 for e in elements)
    @test sum(e.dsigma[1] for e in elements) ≈ 1.0
    @test all(e.fields.q ≈ 3.0 + sum((1,2,3) .* e.centroid) for e in elements)
    # Corner tuples give the same result as the 2×2×2 array.
    @test cell_isosurface(Tuple(T), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0) ==
          cell_isosurface(T, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0)
    # The grid driver on the same cube agrees.
    @test isosurface(T, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]), 1.0; fields=(q=q,)) == elements
    @test isempty(cell_isosurface(T, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2.0))
end

@testset "Streaming from two time slices" begin
    using StaticArrays
    t = collect(range(0.0, 1.0; length=6)); x = collect(range(-1.0, 1.0; length=9)); y = x
    # Hydro-like state: an array of state vectors (energy density, ux, uy) per time slice.
    state(ti) = [SVector(1.3 - ti - 0.15 * (xi^2 + yi^2), 0.1xi, 0.1yi) for xi in x, yi in y]
    temperature(u) = u[1]^(1/4) * 1.1                     # a fake equation of state
    T = [temperature(state(ti)[i, j]) for ti in t, i in eachindex(x), j in eachindex(y)]
    ux = [0.1xi for ti in t, xi in x, yi in y]
    reference_result = isosurface(T, (t, x, y), 0.95; fields=(ux=ux,))

    @test slab_corners((10, 20, 30, 40), (11, 21, 31, 41)) == (10, 11, 20, 21, 30, 31, 40, 41)
    ref = ReferenceGrid(Val(3))
    out = element_vector(Val(3), (ux=ux,))
    u_prev = state(t[1])
    for k in 2:length(t)
        u = state(t[k])
        for I in CartesianIndices(size(u) .- 1)
            cell_isosurface!(out, slab_corners(temperature, u_prev, u, I), Cell((t[k-1], t[k]), (x, y), I), 0.95;
                             fields=(ux=slab_corners(v -> v[2], u_prev, u, I),), reference=ref)
        end
        u_prev = u
    end
    # Same elements; the two loops visit the cells in a different order.
    bycentroid(v) = sort(v; by=e -> Tuple(e.centroid))
    @test length(out) == length(reference_result)
    @test all(a.vertices == b.vertices && a.dsigma == b.dsigma && a.fields == b.fields
              for (a, b) in zip(bycentroid(out), bycentroid(reference_result)))
end

@testset "Fold, callback and in-place forms agree" begin
    using StaticArrays, LinearAlgebra
    T = Tuple(1.0 - sum(abs2, (t, x, y, z)) for t in 0:1, x in 0:1, y in 0:1, z in 0:1)
    cell = Cell((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    q = Tuple(2.0 + t + x for t in 0:1, x in 0:1, y in 0:1, z in 0:1)
    elements = cell_isosurface(T, cell, 0.5; fields=(q=q,))
    total = sum(e.dsigma for e in elements)
    # fold with an immutable SVector accumulator
    @test cell_isosurface((acc, e) -> acc + e.dsigma, zero(SVector{4,Float64}), T, cell, 0.5) ≈ total
    # fold with a field-weighted integrand
    @test cell_isosurface((acc, e) -> acc + e.fields.q * e.dsigma[1], 0.0, T, cell, 0.5; fields=(q=q,)) ≈
          sum(e.fields.q * e.dsigma[1] for e in elements)
    # callback form sees every element once
    seen = Ref(0)
    cell_isosurface(T, cell, 0.5) do e; seen[] += 1; end
    @test seen[] == length(elements)
    # in-place form appends to an existing vector
    out = element_vector(Val(4), (q=q,)); cell_isosurface!(out, T, cell, 0.5; fields=(q=q,)); cell_isosurface!(out, T, cell, 0.5; fields=(q=q,))
    @test out == [elements; elements]
    # a cell the level does not reach returns init unchanged
    @test cell_isosurface((acc, e) -> acc + 1, 0, T, cell, 5.0) == 0
end

println("All simplicial freeze-out geometry tests passed.")
