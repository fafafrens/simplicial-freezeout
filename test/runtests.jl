using SimplicialFreezeOut
using SimplicialFreezeOut: from_bits, to_bits, linear_index, corners, case_index, hot_pattern,
                           vertex_values, lerp, intersection_pieces, area_covector
using Test, StaticArrays, LinearAlgebra, Random

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

    per_cell = surface_elements(T, (t, x, y), 0.45; fields=(q=q,))
    # dΣ and areas add up exactly
    @test sum(e.dsigma for e in per_cell) ≈ sum(e.dsigma for e in simplices)
    @test sum(e.area for e in per_cell) ≈ sum(norm(e.dsigma) for e in simplices)
    # an affine field is the field at the merged centroid
    @test all(e.fields.q ≈ affine(e.centroid) for e in per_cell)
    @test all(0 < flatness(e) <= 1 + 1e-12 for e in per_cell)
    @test integrate_surface(per_cell, e -> e.dsigma[1]) ≈ integrate_surface(simplices, e -> e.dsigma[1])
    @test length(per_cell) < length(simplices)

    # Merging is associative: merging the merged cells equals merging every piece at once.
    whole = foldl(merge_element, simplices; init=zero(element_type(3, (:q,))))
    remerged = foldl(merge_element, per_cell; init=zero(element_type(3, (:q,))))
    @test remerged.centroid ≈ whole.centroid && remerged.dsigma ≈ whole.dsigma &&
          remerged.area ≈ whole.area && remerged.fields.q ≈ whole.fields.q
    @test flatness(whole) < 0.99                       # the whole surface is visibly curved

    # A block of cells merged at once, one at a time, or from its per-cell elements.
    block = CartesianIndices((2:4, 3:4, 1:2))
    cornerTs = [cell_corners(T, I) for I in block]
    cells = [Cell((t, x, y), I) for I in block]
    cell_fields = [(q=cell_corners(q, I),) for I in block]
    together = surface_element(cornerTs, cells, 0.45; fields=cell_fields)
    one_by_one = foldl((acc, i) -> surface_element(acc, cornerTs[i], cells[i], 0.45; fields=cell_fields[i]),
                       eachindex(cells); init=zero(element_type(3, (:q,))))
    @test together == one_by_one
    from_elements = foldl(merge_element, (surface_element(cornerTs[i], cells[i], 0.45; fields=cell_fields[i]) for i in eachindex(cells));
                          init=zero(element_type(3, (:q,))))
    @test from_elements.centroid ≈ together.centroid && from_elements.dsigma ≈ together.dsigma &&
          from_elements.area ≈ together.area && from_elements.fields.q ≈ together.fields.q
    @test together.fields.q ≈ affine(together.centroid)
    @test !iszero(together) && iszero(surface_element(cornerTs, cells, 5.0; fields=cell_fields))
    @test surface_element(cornerTs, Tuple(cells), 0.45).dsigma ≈ together.dsigma        # tuples of cells, no fields
    @test_throws DimensionMismatch surface_element(cornerTs[1:end-1], cells, 0.45)
    @test_throws DimensionMismatch surface_element(cornerTs, cells, 0.45; fields=cell_fields[1:end-1])
    @test iszero(surface_element(cornerTs[1:0], cells[1:0], 0.45; fields=cell_fields[1:0]))
    # the docstring example: two unit squares stacked along x
    two = surface_element([Tuple(1.5 - t for t in 0:1, x in 0:1) for _ in 1:2],
                          [Cell((0.0, 0.0), (1.0, 1.0)), Cell((0.0, 1.0), (1.0, 1.0))], 1.0)
    @test two.centroid ≈ [0.5, 1.0] && two.dsigma ≈ [2.0, 0.0] && two.area ≈ 2.0
end

@testset "Corner indexing and reference tables" begin
    # The examples in the docstrings.
    @test from_bits((1, 0, 1)) == 5 && to_bits(5, 4) == (true, false, true, false)
    @test linear_index(Corner((1, 0, 1))) == 6 && Corner(6, Val(3)).bits == (1, 0, 1)
    A = [10 30; 20 40]
    @test A[Corner((1, 0))] == 20 && Corner(4, Val(2)).bits == (1, 1)
    @test map(c -> c.bits, corners(Val(2))) == ((0, 0), (1, 0), (0, 1), (1, 1))
    @test cell_corners(A, CartesianIndex(1, 1)) == (10.0, 20.0, 30.0, 40.0)
    @test case_index((false, true, true, false)) == 7 && case_index((true, true, true, true)) == 16
    triangle = (Corner((0, 0)), Corner((1, 0)), Corner((1, 1)))
    @test vertex_values((10, 20, 30, 40), triangle) == (10.0, 20.0, 40.0)
    @test lerp(map(SimplicialFreezeOut.coordinates, triangle), (1, 3), 0.25) ≈ [0.25, 0.25]
    cell = Cell(([0.0, 0.5, 1.0], [-1.0, 0.0, 1.0]), CartesianIndex(2, 1))
    @test cell.origin == [0.5, -1.0] && cell.width == (0.5, 1.0)
    @test SimplicialFreezeOut.position(cell, SVector(1, 1)) == [1.0, 0.0]
    @test intersection_pieces((true, true, false, false)) == [((1, 3), (2, 3), (2, 4)), ((1, 3), (1, 4), (2, 4))]

    for D in 2:5
        N = D + 1
        reference = ReferenceGrid(Val(D))
        @test all(hot_pattern(case_index(p), Val(D)) == p for p in map(i -> hot_pattern(i, Val(D)), 1:2^N))
        @test all(linear_index(Corner(i, Val(D))) == i for i in 1:2^D)
        # Every simplex is a staircase from 0…0 to 1…1, and half of them are left-handed.
        @test allunique(reference.simplices)
        @test all(s[1].bits == ntuple(_ -> 0, D) && s[end].bits == ntuple(_ -> 1, D) &&
                  all(sum(s[a+1].bits .- s[a].bits) == 1 for a in 1:D) for s in reference.simplices)
        @test count(reference.left_handed) == factorial(D) ÷ 2
        # A k-vs-(N-k) cut has binomial(N-2, k-1) pieces, each joining hot to cold vertices.
        for i in 1:2^N
            is_hot = hot_pattern(i, Val(D))
            k = min(count(is_hot), N - count(is_hot))
            pieces = reference.cases[i]
            @test length(pieces) == (k == 0 ? 0 : binomial(N - 2, k - 1))
            @test all(is_hot[a] != is_hot[b] for piece in pieces for (a, b) in piece)
            @test allunique(map(Set, pieces))
        end
    end
    @test_throws ArgumentError ReferenceGrid(Val(1))
end

@testset "Area covector" begin
    rng = Xoshiro(1)
    for D in 2:5, _ in 1:20
        v = ntuple(_ -> SVector{D}(randn(rng, D)), D)
        dσ = area_covector(v)
        E = hcat(ntuple(a -> v[a+1] - v[1], D - 1)...)
        @test all(abs(dot(dσ, E[:, a])) < 1e-12 * norm(E)^D for a in 1:D-1)     # normal to the simplex
        @test norm(dσ) ≈ sqrt(det(E' * E)) / factorial(D - 1)                  # its (D-1)-volume
        @test det([dσ E]) > 0                                                  # the sign convention
        # the written-out D = 3 and 4 forms agree with the general one
        @test invoke(area_covector, Tuple{NTuple{n,SVector{n,Float64}} where n}, v) ≈ dσ
    end
    @test area_covector((SVector(0.0, 0.0), SVector(2.0, 0.0))) == [0.0, -2.0]
    @test area_covector((SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))) == [0.0, 0.0, 0.5]
end

@testset "Affine temperature: orientation in every dimension and cell orientation" begin
    # An affine T is cut by one plane. Every piece must lie on it and point from
    # hot to cold, whatever the dimension and the signs of the cell widths
    # (a negative width mirrors the cell). Random gradients reach every
    # hot/cold pattern of every simplex.
    rng = Xoshiro(2)
    for D in 2:5
        reference = ReferenceGrid(Val(D))
        for signs in Iterators.product(ntuple(_ -> (1, -1), D)...), _ in 1:3
            g = SVector{D}(randn(rng, D)); c = randn(rng)
            q(x) = 1.0 + dot(SVector{D}(1:D), x)                                 # an affine field
            cell = Cell(SVector{D}(randn(rng, D)), signs .* (0.5 .+ rand(rng, D)))
            X = map(k -> SimplicialFreezeOut.position(cell, SimplicialFreezeOut.coordinates(k)), corners(Val(D)))
            cornerT = map(x -> c + dot(g, x), X)
            Tfo = c + dot(g, cell.origin + SVector(cell.width) / 2) + 0.2 * randn(rng) * norm(g)
            pieces = surface_simplices(cornerT, cell, Tfo; fields=(q=map(q, X),), reference)
            minimum(cornerT) < Tfo < maximum(cornerT) || continue
            @test !isempty(pieces)
            @test all(abs(c + dot(g, v) - Tfo) < 1e-10 for e in pieces for v in e.vertices)
            @test all(e.dsigma ≈ -norm(e.dsigma) * g / norm(g) for e in pieces)
            @test all(e.fields.q ≈ q(e.centroid) for e in pieces)
            e = surface_element(cornerT, cell, Tfo; fields=(q=map(q, X),), reference)
            @test flatness(e) ≈ 1 && e.fields.q ≈ q(e.centroid)
        end
    end
end

# Closed surface checks: every (D-2)-face shared by exactly two pieces, and Σ dΣ = 0.
function face_counts(simplices)
    faces = Dict{Any,Int}()
    D = length(first(simplices).vertices)
    for e in simplices, drop in 1:D
        key = Tuple(sort([round.(e.vertices[i]; digits=8) for i in 1:D if i != drop]))
        faces[key] = get(faces, key, 0) + 1
    end
    values(faces)
end

@testset "Closed surfaces in 3+1D" begin
    x = collect(range(-1.0, 1.0; length=17))
    # A Gaussian with a cross term, not affine inside cells: its isotherm is the
    # ellipsoid rᵀAr = R² with A = 1 - 0.3(e_t e_xᵀ + e_x e_tᵀ), all cut patterns occur.
    R² = 0.4863
    gaussian(x) = [exp(-(t^2 + a^2 + b^2 + c^2) + 0.6t * a) for t in x, a in x, b in x, c in x]
    simplices = surface_simplices(gaussian(x), (x, x, x, x), exp(-R²))
    @test all(==(2), face_counts(simplices))
    @test norm(sum(e.dsigma for e in simplices)) < 1e-12
    # The enclosed volume, (1/4)∮ x·dΣ, converges to the ellipsoid's at second order in h.
    volume(x) = sum(dot(e.centroid, e.dsigma) for e in surface_simplices(gaussian(x), (x, x, x, x), exp(-R²))) / 4
    exact = π^2 / 2 * R²^2 / sqrt(1 - 0.09)
    coarse, fine = volume(collect(range(-1.0, 1.0; length=9))) - exact, volume(x) - exact
    @test abs(fine / exact) < 0.04 && 3.5 < coarse / fine < 4.5

    # T = 1 - r² with grid nodes exactly at Tfo: zero-area pieces dropped, still closed.
    x = collect(range(-1.0, 1.0; length=9))
    T = [1.0 - (t^2 + a^2 + b^2 + c^2) for t in x, a in x, b in x, c in x]
    simplices = surface_simplices(T, (x, x, x, x), 0.5)
    @test any(==(0.5), T)
    @test all(allunique(e.vertices) && norm(e.dsigma) > 0 for e in simplices)
    @test all(==(2), face_counts(simplices))
    @test norm(sum(e.dsigma for e in simplices)) < 1e-12
    @test all(dot(e.centroid, e.dsigma) > 0 for e in simplices)
end

@testset "Isotherms through grid nodes and along cell faces" begin
    # A node at T == Tfo is hot: a cell whose face is at Tfo and that is colder
    # beyond it holds the face; the cell on the hot side holds nothing.
    cell = Cell((0.0, 0.0), (1.0, 1.0))
    on_face = surface_simplices((1.0, 0.5, 1.0, 0.5), cell, 1.0)
    @test all(v[1] == 0 for e in on_face for v in e.vertices) && sum(e.dsigma for e in on_face) ≈ [1.0, 0.0]
    @test isempty(surface_simplices((1.5, 1.0, 1.5, 1.0), cell, 1.0))
    # Affine temperatures whose isotherms run through grid nodes, along faces or
    # across them: the total dΣ is continuous in Tfo, so no piece is lost or doubled.
    for D in 2:4, g in (ntuple(k -> k == 1 ? -1.0 : 0.0, D), ntuple(k -> 0.5k - 1.5, D), ntuple(k -> isodd(k) ? -1.0 : 2.0, D))
        x = collect(0.0:0.5:2.0)
        T = [dot(g, Tuple(I)) / 2 for I in CartesianIndices(ntuple(_ -> length(x), D))]
        total(Tfo) = sum((e.dsigma for e in surface_simplices(T, ntuple(_ -> x, D), Tfo)); init=zero(SVector{D,Float64}))
        for Tfo in unique(T)[2:end-1]
            @test isapprox(total(Tfo), total(Tfo + 1e-9); atol=1e-6) && isapprox(total(Tfo), total(Tfo - 1e-9); atol=1e-6)
        end
    end
end

@testset "Streaming in 3+1D" begin
    rng = Xoshiro(3)
    prev, next = rand(rng, 3, 4, 3), rand(rng, 3, 4, 3)
    stacked = permutedims(cat(prev, next; dims=4), (4, 1, 2, 3))                 # (2, nx, ny, nz)
    for I in CartesianIndices(size(prev) .- 1)
        @test slab_corners(identity, prev, next, I) == cell_corners(stacked, CartesianIndex(1, Tuple(I)...))
    end
    # The grid form on a two-slice array is one step of a full extraction.
    t = [0.0, 0.3, 0.7]; x = [0.0, 1.0, 2.0]; y = [0.0, 0.5, 1.5, 2.0]; z = [-1.0, 0.0, 1.0]
    T = [1.2 - ti + 0.1 * (xi - yi * zi) for ti in t, xi in x, yi in y, zi in z]
    full = surface_simplices(T, (t, x, y, z), 0.7)
    steps = [surface_simplices(T[k:k+1, :, :, :], (t[k:k+1], x, y, z), 0.7) for k in 1:2]
    @test vcat(steps...) == full
end

@testset "Merging by hand" begin
    # Two unit segments in 2D with normals an angle θ apart: flatness cos(θ/2).
    segment(θ) = SurfaceSimplex((SVector(0.0, 0.0), SVector(0.0, 0.0)), SVector(θ, 0.0), SVector(cos(θ), sin(θ)), (q=θ,))
    zero2 = zero(element_type(2, (:q,)))
    @test iszero(zero2)
    for θ in (0.0, 0.3, 1.0, π)
        e = merge_element(merge_element(zero2, segment(0.0)), segment(θ))
        @test e.area ≈ 2 && e.centroid ≈ [θ / 2, 0] && e.fields.q ≈ θ / 2
        @test isapprox(flatness(e), cos(θ / 2); atol=1e-15)                   # 0 for opposite sheets
    end
    # zero area changes nothing; an element merges like its simplices
    e = merge_element(zero2, segment(0.3))
    @test merge_element(e, zero2) == e
    @test merge_element(zero2, e) == e
    @test merge_element(zero2, zero2) == zero2
    nothing_there = SurfaceSimplex(segment(0.0).vertices, SVector(5.0, 5.0), SVector(0.0, 0.0), (q=9.0,))
    @test merge_element(e, nothing_there) == e
    # integrate_surface with a measure
    @test integrate_surface([segment(0.0), segment(2.0)], s -> s.dsigma[1]; measure=x -> 1 + x[1]) ≈
          1 + 3 * cos(2.0)
end

@testset "Input checks and empty results" begin
    t = [0.0, 1.0, 2.0]; x = [0.0, 0.5, 1.0]
    T = [1.0 - ti for ti in t, xi in x]
    for extract in (surface_simplices, surface_elements)
        @test_throws DimensionMismatch extract(T, (t, [0.0, 1.0]), 0.5)
        @test_throws ArgumentError extract(T, (t, [0.0, 1.0, 0.5]), 0.5)
        @test_throws ArgumentError extract(T, (t, [0.0, 0.5, 0.5]), 0.5)
        @test_throws ArgumentError extract(T[1:1, :], ([0.0], x), 0.5)
        @test_throws DimensionMismatch extract(T, (t, x), 0.5; fields=(q=zeros(2, 3),))
        @test isempty(extract(T, (t, x), 5.0))
    end
end

println("All simplicial freeze-out geometry tests passed.")
