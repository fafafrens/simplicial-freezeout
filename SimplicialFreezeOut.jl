module SimplicialFreezeOut

using LinearAlgebra
using StaticArrays
using Combinatorics: multiset_permutations

export SurfaceElement, ReferenceGrid, Corner, Cell, isosurface, cell_isosurface, cell_isosurface!, element_vector, cell_corners, slab_corners, integrate_surface

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"An edge of a simplex, as a sorted pair of local vertex labels (1-based)."
const Edge = NTuple{2,Int}

"A point in D-dimensional coordinate space."
const Point{D} = SVector{D,Float64}
"A covector in D-dimensional coordinate space, e.g. an area element dΣ_μ."
const Covector{D} = SVector{D,Float64}

"""
    Corner{D}

A corner of the unit cell `[0,1]^D`, given by its 0/1 coordinates `bits`
(axis 1 is time); `coordinates(c)` returns them as a vector. A corner also
indexes a cell's corner data: `A[corner]` works on a `2×…×2` array or on a
tuple of `2^D` values, using the column-major linear index
`1 + Σ_μ bits_μ 2^(μ-1)` (see `linear_index`). For D = 2:

    corner (t, x)   (0,0)  (1,0)  (0,1)  (1,1)
    linear index      1      2      3      4

so with `A = [10 30; 20 40]` (rows are t, columns x), `A[Corner((1, 0))]`
is `20` and `Corner(4, Val(2))` is the corner `(1, 1)`.
"""
struct Corner{D}
    bits::NTuple{D,Int}
end
coordinates(c::Corner) = SVector(c.bits)

"""
    RefSimplex{D,N}

One of the `D!` simplices of the Kuhn–Freudenthal subdivision of the unit
cell `[0,1]^D`: its `N = D+1` vertices, as corners, in local order. The two
triangles of the unit square are

    ((0,0), (1,0), (1,1))      switch on t, then x
    ((0,0), (0,1), (1,1))      switch on x, then t

Local vertex labels 1…N refer to positions in this tuple; the edge `(1, 3)`
of either triangle is the diagonal.
"""
const RefSimplex{D,N} = NTuple{N,Corner{D}}

"""
    ReferenceGrid{D,N}

Everything that is independent of the data: the reference simplices and, for
each of the `2^N` hot/cold vertex patterns, the list of surface pieces it
produces. `cases[case_index(is_hot)]` is a list of pieces; each piece is the
`D`-tuple of crossed edges whose intersection points form one `(D-1)`-simplex.
"""
struct ReferenceGrid{D,N}
    simplices::Vector{RefSimplex{D,N}}
    cases::Vector{Vector{NTuple{D,Edge}}}
end

"""
    Cell{D}

A hypercube cell `[origin, origin + width]` of the grid. `Cell(origin, width)`
takes the lowest corner and the per-axis sizes; `Cell(coords, I)` builds the
cell whose lowest corner is grid point `I` of the coordinate vectors `coords`:

    t = [0.0, 0.5, 1.0];  x = [-1.0, 0.0, 1.0]
    Cell((t, x), CartesianIndex(2, 1))      # origin (0.5, -1.0), width (0.5, 1.0)

`position(cell, s)` maps unit-cell coordinates to physical ones, e.g.
`s = (1, 1)` to `(1.0, 0.0)` for this cell.
"""
struct Cell{D}
    origin::Point{D}
    width::NTuple{D,Float64}      # per-axis cell sizes
end
Cell(origin, width) = Cell(Point{length(origin)}(origin), NTuple{length(width),Float64}(width))
function Cell(coords::Tuple{Vararg{AbstractVector,D}}, I::CartesianIndex{D}) where {D}
    origin = ntuple(μ -> Float64(coords[μ][I[μ]]), Val(D))
    width = ntuple(μ -> Float64(coords[μ][I[μ]+1]) - origin[μ], Val(D))
    Cell(origin, width)
end
"""
    Cell((t0, t1), coords, I)

The spacetime cell between times `t0` and `t1` above the spatial cell whose
lowest corner is grid point `I` of the spatial coordinate vectors `coords`.
"""
function Cell((t0, t1)::Tuple{Real,Real}, coords::Tuple{Vararg{AbstractVector,d}}, I::CartesianIndex{d}) where {d}
    origin = (Float64(t0), ntuple(μ -> Float64(coords[μ][I[μ]]), Val(d))...)
    width = (Float64(t1 - t0), ntuple(μ -> Float64(coords[μ][I[μ]+1]) - origin[μ+1], Val(d))...)
    Cell(origin, width)
end
"Physical position of the point with unit-cell coordinates `s ∈ [0,1]^D`."
position(cell::Cell, s) = cell.origin + cell.width .* s

"""
    SurfaceElement{D,F}

A flat `(D-1)`-simplex of the freeze-out surface with its `D` vertices, its
centroid, the integrated oriented area covector `dsigma` (pointing from
`T ≥ Tfo` to `T < Tfo`, no metric factor included) and the fluid `fields`
interpolated at the centroid. In the unit square with `T = 1.5 - t` and
`Tfo = 1`, the isotherm is the line `t = 0.5`, cut into one segment per
triangle:

    vertices                   centroid     dsigma
    ((0.5, 0.0), (0.5, 0.5))   (0.5, 0.25)  (0.5, 0.0)
    ((0.5, 0.5), (0.5, 1.0))   (0.5, 0.75)  (0.5, 0.0)

Each `dsigma` has the segment's length 0.5 and points to larger `t`, where
it is colder; they sum to `(1, 0)`, the oriented area of the whole line.
"""
struct SurfaceElement{D,F}
    vertices::NTuple{D,Point{D}}
    centroid::Point{D}
    dsigma::Covector{D}
    fields::F
end

# ---------------------------------------------------------------------------
# Reference geometry (computed once per D)
# ---------------------------------------------------------------------------

"""
    from_bits(bits)
    to_bits(n, len)

Binary encoding of a tuple of 0/1 (or Bool) values, least significant first,
and its inverse: `from_bits((1, 0, 1))` is `5` and `to_bits(5, 4)` is
`(true, false, true, false)`. Used for corner indices and for hot/cold
patterns. `to_bits` reads bit `k` as the parity of `n` shifted right by `k`
places; unlike `digits` it allocates nothing, so corner lists fold to constants.
"""
from_bits(bits) = evalpoly(2, bits)
to_bits(n::Int, len::Int) = ntuple(k -> isodd(n >> (k - 1)), len)

"""
    linear_index(c::Corner)
    Corner(index, Val(D))

A corner's linear index into a `2×…×2` array of corner data, and back:
`linear_index(Corner((1, 0, 1)))` is `6` and `Corner(6, Val(3))` is the
corner `(1, 0, 1)`.
"""
linear_index(c::Corner) = 1 + from_bits(c.bits)
Corner(index::Int, ::Val{D}) where {D} = Corner{D}(Tuple(to_bits(index - 1, D)))
"All 2^D corners of the unit cell, in linear-index order: (0,0), (1,0), (0,1), (1,1) for D = 2."
corners(::Val{D}) where {D} = ntuple(i -> Corner(i, Val(D)), Val(2^D))
Base.getindex(A::AbstractArray, c::Corner) = A[linear_index(c)]
Base.getindex(A::Tuple, c::Corner) = A[linear_index(c)]

"""
    case_index(is_hot)

Index into `ReferenceGrid.cases` of the hot/cold pattern `is_hot`, a tuple of
`D+1` Bools, one per simplex vertex in local order. The pattern is read as a
binary number with vertex 1 as the least significant bit, plus one:

    (false, false, false, false) → 1      all cold
    (true,  false, false, false) → 2      vertex 1 hot
    (false, true,  true,  false) → 7      vertices 2 and 3 hot
    (true,  true,  true,  true)  → 16     all hot

`hot_pattern(index, Val(D))` is the inverse, used to build the table.
"""
case_index(is_hot) = 1 + from_bits(is_hot)
hot_pattern(index::Int, ::Val{D}) where {D} = Tuple(to_bits(index - 1, D + 1))

edge(a::Int, b::Int) = minmax(a, b)

"""
    vertex_values(A, simplex)

Values of a cell's corner data `A` at the vertices of `simplex`, as a tuple in
local vertex order 1…D+1. For the unit-square triangle `((0,0), (1,0), (1,1))`
and corner data `(10, 20, 30, 40)` in linear-index order, the result is
`(10.0, 20.0, 40.0)`.
"""
vertex_values(A, simplex::RefSimplex) = map(c -> Float64(A[c]), simplex)

"The unit vectors e₁ … eₙ of an n-dimensional space, as a tuple of static vectors."
unit_vectors(::Val{n}) where {n} = ntuple(μ -> SMatrix{n,n,Float64}(I)[:, μ], Val(n))

"""
    lerp(X, (a, b), t)

Linear interpolation between entries `a` and `b` of `X`, a fraction `t` of the
way from `a` to `b`: with `X = ((0,0), (1,0), (1,1))`, `lerp(X, (1, 3), 0.25)`
is `(0.25, 0.25)`, a quarter of the way along the diagonal.
"""
lerp(X, (a, b)::Edge, t) = (1 - t) * X[a] + t * X[b]

"""
    staircases(steps)

All staircases made of the given `steps`: take the steps in every distinct
order, and for each order walk from the origin, recording the point after
each step. With one step `down = (1,0)` and one `right = (0,1)`:

    down, right:  (0,0) → (1,0) → (1,1)
    right, down:  (0,0) → (0,1) → (1,1)

Both triangulations below are of this kind (Kuhn–Freudenthal / staircase
triangulations).
"""
function staircases(steps)
    origin = zero(first(steps))
    map(multiset_permutations(steps, length(steps))) do order
        points = [origin]
        for step in order
            push!(points, points[end] + step)
        end
        points
    end
end

# The unit cell is split by the staircases from 0…0 to 1…1 that switch on one
# coordinate at a time; each visits D+1 corners, which span one simplex (see
# the example in `RefSimplex`).
function reference_simplices(::Val{D}) where {D}
    simplices = RefSimplex{D,D + 1}[]
    for path in staircases(unit_vectors(Val(D)))  # D+1 points from 0…0 to 1…1
        vertices = ntuple(a -> Corner(Tuple(Int.(path[a]))), D + 1)
        push!(simplices, vertices)
    end
    simplices
end

"""
    intersection_pieces(is_hot)

For a hot/cold pattern of the D+1 simplex vertices, the surface pieces, each
as the D edges it crosses. The crossed edges always join a vertex of the
smaller group to one of the larger group:

    (true, false, false)             D=2, one segment:   [((1,2), (1,3))]
    (true, true, false, false)       D=3, 2-vs-2, a quadrilateral split in two triangles:
                                     [((1,3), (2,3), (2,4)), ((1,3), (1,4), (2,4))]
    (false, true, true, false, false) D=4, 2-vs-3, a prism split in three tetrahedra:
                                     [((1,2), (1,3), (3,4), (3,5)), ((1,2), (2,4), (3,4), (3,5)),
                                      ((1,2), (2,4), (2,5), (3,5))]
"""
function intersection_pieces(is_hot::NTuple{N,Bool}) where {N}
    D = N - 1
    hot, cold = findall(is_hot), findall(.!is_hot)
    (isempty(hot) || isempty(cold)) && return NTuple{D,Edge}[]
    small, large = length(hot) <= length(cold) ? (hot, cold) : (cold, hot)
    # The crossed edges form a grid: x[i, j] joins small[i] to large[j].
    x = [edge(s, l) for s in small, l in large]
    triangulate(x)
end

"""
    triangulate(x::Matrix{Edge})

Split the cut polytope into (D-1)-simplices. The crossing points `x[i, j]`
form a product of two simplices (one per group of vertices), and its
staircase triangulation has one simplex per staircase from `x[1, 1]` to
`x[end, end]`. For the 2-vs-2 grid

    x = [(1,3) (1,4)
         (2,3) (2,4)]

the two staircases down-right and right-down give the triangles
`((1,3), (2,3), (2,4))` and `((1,3), (1,4), (2,4))`, sharing the diagonal
`(1,3)`–`(2,4)`. A 1×D grid has a single staircase and a 2×3 grid three.
Shared faces of neighbouring simplices are triangulated identically because
a staircase restricted to a face is again a staircase.
"""
function triangulate(x::Matrix{Edge})
    m, n = size(x)
    D = m + n - 1                                  # points per staircase
    down, right = CartesianIndex(1, 0), CartesianIndex(0, 1)
    steps = [fill(down, m - 1); fill(right, n - 1)]
    pieces = NTuple{D,Edge}[]
    for path in staircases(steps)                  # from x[1,1] to x[m,n]
        cells = [CartesianIndex(1, 1) + step for step in path]
        push!(pieces, Tuple(x[cells]))
    end
    pieces
end

"""
    ReferenceGrid(Val(D))

Build the reference geometry for `D ≥ 2` spacetime dimensions (`D!` simplices
and `2^(D+1)` cut patterns). It is data independent and cheap for the physical
cases `D = 2, 3, 4`; `isosurface` builds it on every call.
"""
function ReferenceGrid(::Val{D}) where {D}
    D >= 2 || throw(ArgumentError("spacetime dimension D must be at least 2"))
    cases = [intersection_pieces(hot_pattern(i, Val(D))) for i in 1:2^(D+1)]
    ReferenceGrid(reference_simplices(Val(D)), cases)
end

# ---------------------------------------------------------------------------
# Surface elements
# ---------------------------------------------------------------------------

"""
    area_covector(vertices)

Integrated area covector of the (D-1)-simplex with the given vertices: the
generalized cross product of its edge vectors, `dσ_μ = det[e_μ | E] / (D-1)!`,
where `E` has the edge vectors as columns. The segment from `(0,0)` to `(2,0)`
gives `(0, -2)`, length 2 and normal to the segment; the right triangle
`(0,0,0), (1,0,0), (0,1,0)` gives `(0, 0, 0.5)`, its area times the normal.
The sign depends on the vertex order and is fixed by the caller.
"""
function area_covector(vertices::NTuple{D,Point{D}}) where {D}
    E = hcat(ntuple(a -> vertices[a+1] - vertices[1], Val(D - 1))...)
    e = unit_vectors(Val(D))
    Covector{D}(ntuple(μ -> det([e[μ] E]) / factorial(D - 1), Val(D)))
end

# One piece of the isosurface inside one reference simplex of `cell`, or
# `nothing` if it is degenerate. `cornerT` and each entry of the NamedTuple
# `fields` hold the cell's corner data, indexable by `Corner`.
function surface_element(edges::NTuple{D,Edge}, simplex::RefSimplex{D}, cornerT, Tfo,
                         cell::Cell{D}, fields) where {D}
    T = vertex_values(cornerT, simplex)             # in local vertex order 1…D+1
    X = map(coordinates, simplex)                   # vertex positions in the unit cell
    # Fraction of the way from vertex a to vertex b at which the linear
    # interpolant reaches Tfo, for each crossed edge.
    θ = map(edges) do (a, b)
        (Tfo - T[a]) / (T[b] - T[a])
    end
    vertices = map((e, t) -> position(cell, lerp(X, e, t)), edges, θ)
    # A vertex with T == Tfo exactly puts two crossings on the same point.
    allunique(vertices) || return nothing
    centroid = sum(vertices) / D

    # Orient from hot to cold: toward the coldest simplex vertex, which has
    # T < Tfo strictly and so lies strictly on the cold side of the piece.
    dsigma = area_covector(vertices)
    if dot(dsigma, position(cell, X[argmin(T)]) - centroid) < 0
        dsigma = -dsigma
        vertices = (vertices[2], vertices[1], vertices[3:end]...)
    end

    # Barycentric coordinates of the centroid w.r.t. the D+1 simplex vertices:
    # the same interpolation applied to the unit vectors, averaged.
    basis = unit_vectors(Val(D + 1))
    weights = sum(map((e, t) -> lerp(basis, e, t), edges, θ)) / D
    sampled = map(A -> dot(weights, vertex_values(A, simplex)), fields)
    SurfaceElement(vertices, centroid, dsigma, sampled)
end

"""
    element_vector(Val(D), fields)

An empty, concretely typed vector of the `SurfaceElement`s that
`cell_isosurface` produces in `D` dimensions for these `fields`, e.g.
`SurfaceElement{3, @NamedTuple{ux::Float64}}[]` for `fields = (ux=…,)`.
"""
element_vector(::Val{D}, fields) where {D} = SurfaceElement{D,typeof(map(_ -> 0.0, fields))}[]

"""
    cell_isosurface(f, init, cornerT, cell::Cell, Tfo; fields=NamedTuple(), reference=ReferenceGrid(Val(D)))
    cell_isosurface(f, cornerT, cell::Cell, Tfo; ...)
    cell_isosurface!(elements, cornerT, cell::Cell, Tfo; ...)
    cell_isosurface(cornerT, cell::Cell, Tfo; ...)
    cell_isosurface(cornerT, origin, width, Tfo; ...)

Isosurface `T = Tfo` inside one hypercube cell. `cornerT` holds the `2^D`
corner temperatures, indexable by `Corner`: a `2×…×2` array or a tuple in
linear-index order (see `cell_corners`, `slab_corners`). `fields` is a
NamedTuple of corner values in the same form.

The base form is a fold over the pieces, like `foldl`: starting from `init`,
the running value `acc` is replaced by `f(acc, element)` for each piece in
the order it is built, and the final value is returned. It allocates nothing,
so surface integrals can be accumulated on the fly:

    flux = cell_isosurface((acc, e) -> acc + dot(J, e.dsigma), 0.0, cornerT, cell, Tfo)

The other forms are special cases: the callback form calls `f(element)` for
its effect; the in-place form pushes onto `elements` (see `element_vector`)
and returns it; the remaining forms return a fresh vector of
`SurfaceElement`s, empty if the surface does not cross the cell.
"""
function cell_isosurface(f, init, cornerT, cell::Cell{D}, Tfo::Real;
                         fields::NamedTuple=NamedTuple(),
                         reference::ReferenceGrid{D}=ReferenceGrid(Val(D))) where {D}
    acc = init
    Tmin, Tmax = extrema(c -> cornerT[c], corners(Val(D)))
    Tmin <= Tfo <= Tmax || return acc                                  # cheap rejection
    for simplex in reference.simplices
        is_hot = map(c -> cornerT[c] >= Tfo, simplex)
        for edges in reference.cases[case_index(is_hot)]
            element = surface_element(edges, simplex, cornerT, Tfo, cell, fields)
            isnothing(element) || (acc = f(acc, element))
        end
    end
    acc
end
cell_isosurface(f, cornerT, cell::Cell, Tfo::Real; kwargs...) =
    cell_isosurface((_, e) -> (f(e); nothing), nothing, cornerT, cell, Tfo; kwargs...)
cell_isosurface!(elements::AbstractVector, cornerT, cell::Cell, Tfo::Real; kwargs...) =
    cell_isosurface(push!, elements, cornerT, cell, Tfo; kwargs...)
cell_isosurface(cornerT, cell::Cell{D}, Tfo::Real; fields::NamedTuple=NamedTuple(), kwargs...) where {D} =
    cell_isosurface!(element_vector(Val(D), fields), cornerT, cell, Tfo; fields, kwargs...)
cell_isosurface(cornerT, origin, width, Tfo::Real; kwargs...) =
    cell_isosurface(cornerT, Cell(origin, width), Tfo; kwargs...)

# ---------------------------------------------------------------------------
# Grid driver
# ---------------------------------------------------------------------------

function check_inputs(temperature, coords, fields)
    for (μ, x) in enumerate(coords)
        length(x) == size(temperature, μ) ||
            throw(DimensionMismatch("axis $μ has $(length(x)) points, temperature has $(size(temperature, μ))"))
        length(x) >= 2 || throw(ArgumentError("axis $μ needs at least 2 points"))
        all(x[i] < x[i+1] for i in 1:length(x)-1) ||
            throw(ArgumentError("axis $μ must be strictly increasing"))
    end
    for (name, A) in pairs(fields)
        size(A) == size(temperature) ||
            throw(DimensionMismatch("field $name must have the same shape as temperature"))
    end
end

"""
    cell_corners([f,] A, I::CartesianIndex)

The `2^D` values of the grid array `A` at the corners of the cell whose lowest
corner is `I`, as a tuple in linear-index order, ready for `cell_isosurface`:
with `A = [10 30; 20 40]`, `cell_corners(A, CartesianIndex(1, 1))` is
`(10.0, 20.0, 30.0, 40.0)`, the same as `Tuple(A[1:2, 1:2])`. With `f`, each
grid value is mapped through `f` first, e.g. an equation of state applied to
a state vector.
"""
cell_corners(f, A::AbstractArray{<:Any,D}, I::CartesianIndex{D}) where {D} =
    map(c -> Float64(f(A[I + CartesianIndex(c.bits)])), corners(Val(D)))
cell_corners(A::AbstractArray{<:Real,D}, I::CartesianIndex{D}) where {D} = cell_corners(identity, A, I)

"""
    slab_corners(prev, next)
    slab_corners(f, prev, next, I::CartesianIndex)

Corner values of a spacetime cell between two time slices, in linear-index
order (time is the fastest index, so earlier and later values interleave).
The first form takes the `2^(D-1)` spatial corner values at the earlier time
`prev` and at the later time `next`, e.g. from `cell_corners` on each slice:

    slab_corners((10, 20, 30, 40), (11, 21, 31, 41))   # (10, 11, 20, 21, 30, 31, 40, 41)

The second form reads them from two spatial grid arrays at the cell whose
lowest spatial corner is `I`, applying `f` to each grid value (e.g. an
equation of state, or `u -> u.ux` for a field). With a hydro state kept as
two slices `u_prev` and `u`, one time step of extraction is

    for I in CartesianIndices(size(u) .- 1)
        cell_isosurface!(out, slab_corners(temperature, u_prev, u, I), Cell((t_prev, t), (x, y), I), Tfo;
                         fields=(ux=slab_corners(u -> u.ux, u_prev, u, I),), reference)
    end
"""
slab_corners(prev::NTuple{n}, next::NTuple{n}) where {n} =
    map(corners(Val(ndims_of(n) + 1))) do c
        (c.bits[1] == 0 ? prev : next)[1 + from_bits(c.bits[2:end])]
    end
slab_corners(f, prev::AbstractArray{<:Any,d}, next::AbstractArray{<:Any,d}, I::CartesianIndex{d}) where {d} =
    slab_corners(cell_corners(f, prev, I), cell_corners(f, next, I))
ndims_of(n) = round(Int, log2(n))        # 2^(D-1) spatial corners → D-1

"""
    isosurface(temperature, coords, Tfo; fields=NamedTuple())

Extract the surface `T = Tfo` from a Cartesian-product spacetime grid.
`temperature` is a `D`-dimensional array (`D = 2, 3, 4`; first axis is time)
and `coords` a tuple of `D` strictly increasing coordinate vectors, uniform or
not. `fields` is a NamedTuple of arrays shaped like `temperature`, e.g.
`(ux=ux, uy=uy)`, whose linearly interpolated values are returned at each
element centroid.

Returns a vector of `SurfaceElement`s: segments for `D=2`, triangles for
`D=3`, tetrahedra for `D=4`, each oriented from hot to cold. The extraction
can be streamed by passing two consecutive time slices as a `(2, nx, …)` array.

Vertices with `T == Tfo` count as hot; the resulting zero-area pieces are
dropped. Extended plateaus with `T == Tfo` are not supported.
"""
function isosurface(temperature::AbstractArray{<:Real,D},
                    coords::Tuple{Vararg{AbstractVector,D}}, Tfo::Real;
                    fields::NamedTuple=NamedTuple()) where {D}
    check_inputs(temperature, coords, fields)
    reference = ReferenceGrid(Val(D))
    elements = element_vector(Val(D), fields)

    for I in CartesianIndices(size(temperature) .- 1)
        cell_isosurface!(elements, cell_corners(temperature, I), Cell(coords, I), Tfo;
                         fields=map(A -> cell_corners(A, I), fields), reference)
    end
    elements
end

"""
    integrate_surface(elements, integrand; measure = x -> 1.0)

Centroid quadrature `Σ measure(centroid) * integrand(element)`. The integrand
must already contain the contraction with `element.dsigma`; `measure` supplies
`√(-g)`, e.g. `x -> x[1]` (τ) in Milne coordinates.
"""
integrate_surface(elements, integrand; measure=x -> 1.0) =
    sum(measure(e.centroid) * integrand(e) for e in elements; init=0.0)

end # module
