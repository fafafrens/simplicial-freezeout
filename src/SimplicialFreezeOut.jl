module SimplicialFreezeOut

using LinearAlgebra
using StaticArrays
using Combinatorics: multiset_permutations

export SurfaceSimplex, SurfaceElement, ReferenceGrid, Corner, Cell,
       surface_simplices, surface_simplices!, simplex_type, element_type, surface_element, surface_elements,
       merge_element, flatness, cell_corners, slab_corners, integrate_surface

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
    index::Int      # the linear index, stored: corners are read in the inner loop
    Corner{D}(bits) where {D} = new{D}(bits, 1 + from_bits(bits))
end
Corner(bits::NTuple{D,Integer}) where {D} = Corner{D}(bits)
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
`D`-tuple of crossed edges whose intersection points form one `(D-1)`-simplex,
in the order that makes its area covector point from hot to cold in a
right-handed simplex. `left_handed[k]` is true if simplex `k` is left-handed
(negatively oriented), so that its pieces need two edges swapped.
"""
struct ReferenceGrid{D,N}
    simplices::Vector{RefSimplex{D,N}}
    cases::Vector{Vector{NTuple{D,Edge}}}
    left_handed::Vector{Bool}
end

"""
    Cell{D}

A hypercube cell `[origin, origin + width]` of the grid. `Cell(origin, width)`
takes the lowest corner and the per-axis sizes; `Cell(coords, I)` builds the
cell whose lowest corner is grid point `I` of the coordinate vectors `coords`:

    t = [0.0, 0.5, 1.0];  x = [-1.0, 0.0, 1.0]
    Cell((t, x), CartesianIndex(2, 1))      # origin (0.5, -1.0), width (0.5, 1.0)

`position(cell, s)` maps unit-cell coordinates, a static vector, to physical
ones: `SVector(1, 1)` to `(1.0, 0.0)` for this cell.
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
    SurfaceSimplex{D,F}

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
struct SurfaceSimplex{D,F}
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
linear_index(c::Corner) = c.index
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
cases `D = 2, 3, 4`; `surface_simplices` builds it on every call.
"""
function ReferenceGrid(::Val{D}) where {D}
    D >= 2 || throw(ArgumentError("spacetime dimension D must be at least 2"))
    cases = [oriented_pieces(hot_pattern(i, Val(D))) for i in 1:2^(D+1)]
    simplices = reference_simplices(Val(D))
    ReferenceGrid(simplices, cases, map(is_left_handed, simplices))
end

"The tuple `t` with its first two entries swapped: reverses the orientation of a piece."
swap_first_two(t::Tuple) = (t[2], t[1], t[3:end]...)

# The pieces of a cut, each with its edges ordered so that its area covector
# points from hot to cold in a right-handed simplex. T is affine in a
# simplex, so this direction depends only on the hot/cold pattern: a piece
# cannot turn over without first degenerating. It is found once, with the
# crossings at the edge midpoints of the right-handed staircase simplex 0, e₁, e₁+e₂, …
function oriented_pieces(is_hot::NTuple{N,Bool}) where {N}
    D = N - 1
    X = ntuple(k -> Point{D}(ntuple(μ -> Float64(μ < k), D)), N)
    map(intersection_pieces(is_hot)) do edges
        vertices = map(((a, b),) -> (X[a] + X[b]) / 2, edges)
        cold = X[findfirst(!, is_hot)]
        dot(area_covector(vertices), cold - sum(vertices) / D) > 0 ? edges : swap_first_two(edges)
    end
end

# Whether a reference simplex is left-handed (negatively oriented): the
# determinant of its edge vectors, from the first vertex, is negative.
function is_left_handed(simplex::RefSimplex)
    X = map(coordinates, simplex)
    det(hcat(map(x -> x - X[1], Base.tail(X))...)) < 0
end

# ---------------------------------------------------------------------------
# Surface simplices
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
    # det[e_μ | E] expanded along its first column: the signed minor of E
    # without row μ, a (D-1)×(D-1) determinant.
    Covector{D}(ntuple(Val(D)) do μ
        rows = SVector(ntuple(k -> k < μ ? k : k + 1, Val(D - 1)))
        (isodd(μ) ? 1 : -1) * det(E[rows, :]) / factorial(D - 1)
    end)
end

# The same for the physical cases, written out. In D = 3 it is the cross
# product. In D = 4 the four 3×3 minors share the six 2×2 minors of the last
# two edges, which are computed once: about a third of the work.
function area_covector(v::NTuple{3,Point{3}})
    cross(v[2] - v[1], v[3] - v[1]) / 2
end

function area_covector(v::NTuple{4,Point{4}})
    a, b, c = v[2] - v[1], v[3] - v[1], v[4] - v[1]
    m(i, j) = b[i] * c[j] - b[j] * c[i]
    m12, m13, m14, m23, m24, m34 = m(1, 2), m(1, 3), m(1, 4), m(2, 3), m(2, 4), m(3, 4)
    Covector{4}( a[2] * m34 - a[3] * m24 + a[4] * m23,
                -a[1] * m34 + a[3] * m14 - a[4] * m13,
                 a[1] * m24 - a[2] * m14 + a[4] * m12,
                -a[1] * m23 + a[2] * m13 - a[3] * m12) / 6
end

# One piece of the isosurface inside one reference simplex of `cell`, or
# `nothing` if it is degenerate. `cornerT` and each entry of the NamedTuple
# `fields` hold the cell's corner data, indexable by `Corner`.
function surface_simplex(edges::NTuple{D,Edge}, simplex::RefSimplex{D}, cornerT, Tfo,
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
    # Crossings on different edges can meet only at a shared simplex vertex,
    # so compare them only when one lies at, or within roundoff of, an end.
    near_end(t) = !(sqrt(eps()) < t < 1 - sqrt(eps()))
    any(near_end, θ) && !allunique(vertices) && return nothing
    centroid = sum(vertices) / D
    # The edges come ordered so that this points from hot to cold.
    dsigma = area_covector(vertices)

    # Fields at the centroid: the mean of their values at the crossing points,
    # exact for the affine interpolant, as the centroid is the mean of the vertices.
    crossings(V) = map((e, t) -> lerp(V, e, t), edges, θ)
    sampled = map(A -> sum(crossings(vertex_values(A, simplex))) / D, fields)
    SurfaceSimplex(vertices, centroid, dsigma, sampled)
end

# The type of the interpolated `fields` of a simplex or element: one Float64
# per field name. Takes the names, or the NamedTuple of fields itself.
field_names(names::Tuple{Vararg{Symbol}}) = names
field_names(fields::NamedTuple) = keys(fields)
field_type(names) = NamedTuple{field_names(names),NTuple{length(field_names(names)),Float64}}

"""
    simplex_type(D, names=())

The concrete `SurfaceSimplex` type that `surface_simplices` produces in `D`
dimensions with fields called `names`, a tuple of symbols or the NamedTuple of
fields itself: `simplex_type(3, (:ux, :uy))` is
`SurfaceSimplex{3, @NamedTuple{ux::Float64, uy::Float64}}`. An empty vector to
collect simplices with `surface_simplices!` is

    out = simplex_type(3, (:ux, :uy))[]
"""
simplex_type(D::Integer, names=()) = SurfaceSimplex{D,field_type(names)}

"""
    surface_simplices(f, init, cornerT, cell::Cell, Tfo; fields=NamedTuple(), reference=ReferenceGrid(Val(D)))
    surface_simplices(f, cornerT, cell::Cell, Tfo; ...)
    surface_simplices!(simplices, cornerT, cell::Cell, Tfo; ...)
    surface_simplices(cornerT, cell::Cell, Tfo; ...)
    surface_simplices(cornerT, origin, width, Tfo; ...)

The surface `T = Tfo` inside one hypercube cell, as flat `SurfaceSimplex`
pieces, one or more per reference simplex it crosses. `cornerT` holds the `2^D`
corner temperatures, indexable by `Corner`: a `2×…×2` array or a tuple in
linear-index order (see `cell_corners`, `slab_corners`). `fields` is a
NamedTuple of corner values in the same form.

The base form is a fold over the simplices, like `foldl`: starting from `init`,
the running value `acc` is replaced by `f(acc, s)` for each simplex `s` in
the order it is built, and the final value is returned. It allocates nothing,
so surface integrals can be accumulated on the fly:

    flux = surface_simplices((acc, e) -> acc + dot(J, e.dsigma), 0.0, cornerT, cell, Tfo)

The other forms are special cases: the callback form calls `f(s)` for
its effect; the in-place form pushes onto `simplices` (see `simplex_type`)
and returns it; the remaining forms return a fresh vector of
`SurfaceSimplex`es, empty if the surface does not cross the cell. To merge
them into one element per cell, see `surface_element`.
"""
function surface_simplices(f, init, cornerT, cell::Cell{D}, Tfo::Real;
                           fields::NamedTuple=NamedTuple(),
                           reference::ReferenceGrid{D}=ReferenceGrid(Val(D))) where {D}
    acc = init
    Tmin, Tmax = extrema(c -> cornerT[c], corners(Val(D)))
    Tmin <= Tfo <= Tmax || return acc                                  # cheap rejection
    # Pieces are reversed in left-handed simplices, and in a cell made left-handed by negative widths.
    left_handed_cell = isodd(count(<(0), cell.width))
    for (simplex, left_handed) in zip(reference.simplices, reference.left_handed)
        is_hot = map(c -> cornerT[c] >= Tfo, simplex)
        for edges in reference.cases[case_index(is_hot)]
            left_handed ⊻ left_handed_cell && (edges = swap_first_two(edges))
            piece = surface_simplex(edges, simplex, cornerT, Tfo, cell, fields)
            isnothing(piece) || (acc = f(acc, piece))
        end
    end
    acc
end
surface_simplices(f, cornerT, cell::Cell, Tfo::Real; kwargs...) =
    surface_simplices((_, e) -> (f(e); nothing), nothing, cornerT, cell, Tfo; kwargs...)
surface_simplices!(simplices::AbstractVector, cornerT, cell::Cell, Tfo::Real; kwargs...) =
    surface_simplices(push!, simplices, cornerT, cell, Tfo; kwargs...)
surface_simplices(cornerT, cell::Cell{D}, Tfo::Real; fields::NamedTuple=NamedTuple(), kwargs...) where {D} =
    surface_simplices!(simplex_type(D, fields)[], cornerT, cell, Tfo; fields, kwargs...)
surface_simplices(cornerT, origin, width, Tfo::Real; kwargs...) =
    surface_simplices(cornerT, Cell(origin, width), Tfo; kwargs...)

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
corner is `I`, as a tuple in linear-index order, ready for `surface_simplices`:
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
        surface_simplices!(out, slab_corners(temperature, u_prev, u, I), Cell((t_prev, t), (x, y), I), Tfo;
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
    surface_simplices(temperature, coords, Tfo; fields=NamedTuple())

Extract the surface `T = Tfo` from a Cartesian-product spacetime grid.
`temperature` is a `D`-dimensional array (`D = 2, 3, 4`; first axis is time)
and `coords` a tuple of `D` strictly increasing coordinate vectors, uniform or
not. `fields` is a NamedTuple of arrays shaped like `temperature`, e.g.
`(ux=ux, uy=uy)`, whose linearly interpolated values are returned at each
simplex centroid.

Returns a vector of `SurfaceSimplex`es: segments for `D=2`, triangles for
`D=3`, tetrahedra for `D=4`, each oriented from hot to cold. The extraction
can be streamed by passing two consecutive time slices as a `(2, nx, …)` array.

Vertices with `T == Tfo` count as hot; the resulting zero-area pieces are
dropped. Extended plateaus with `T == Tfo` are not supported. For one merged
element per cell, see `surface_elements`.
"""
function surface_simplices(temperature::AbstractArray{<:Real,D},
                           coords::Tuple{Vararg{AbstractVector,D}}, Tfo::Real;
                           fields::NamedTuple=NamedTuple()) where {D}
    check_inputs(temperature, coords, fields)
    reference = ReferenceGrid(Val(D))
    simplices = simplex_type(D, fields)[]

    for I in CartesianIndices(size(temperature) .- 1)
        surface_simplices!(simplices, cell_corners(temperature, I), Cell(coords, I), Tfo;
                           fields=map(A -> cell_corners(A, I), fields), reference)
    end
    simplices
end

# ---------------------------------------------------------------------------
# Surface elements: the simplices of a cell, or of several cells, merged
# ---------------------------------------------------------------------------

"""
    SurfaceElement{D,F}

Several `SurfaceSimplex`es merged into one surface element, like the one
element per cell that Cornelius returns. `dsigma` is the sum of their area
covectors, `area` the sum of their areas `|dσ|` (Euclidean norm of the
components), and `centroid` and `fields` are area-weighted means. In the unit
square with `T = 1.5 - t` and `Tfo = 1`, the two segments shown for
`SurfaceSimplex` merge into

    centroid     dsigma      area   fields
    (0.5, 0.5)   (1.0, 0.0)  1.0    area-weighted means

`dsigma` is summed rather than averaged because it is additive: the flux of a
constant current through the merged simplices is exactly `J ⋅ dsigma`. Fields are
averaged component by component, so rebuild nonlinear quantities from the
means: recompute `u⁰` from averaged spatial components, since a mean of unit
four-velocities is not a unit four-velocity. Merging simplices that lie on one
flat sheet is exact for affine integrands; see `flatness` for how far from flat
a merged element is.
"""
struct SurfaceElement{D,F}
    centroid::Point{D}
    dsigma::Covector{D}
    area::Float64
    fields::F
end

"""
    element_type(D, names=())

The concrete `SurfaceElement` type in `D` dimensions with fields called
`names`, like `simplex_type`: an empty vector of elements is
`element_type(3, (:ux, :uy))[]`.
"""
element_type(D::Integer, names=()) = SurfaceElement{D,field_type(names)}

"""
    zero(element_type(D, names))
    iszero(e::SurfaceElement)

The `SurfaceElement` with zero area: the neutral element of `merge_element`
and the starting value for merging. `iszero` tells whether an element is
empty, e.g. a `surface_element` of a cell the surface does not cross.
"""
Base.zero(::Type{SurfaceElement{D,F}}) where {D,F} =
    SurfaceElement(zero(Point{D}), zero(Covector{D}), 0.0, F(ntuple(_ -> 0.0, fieldcount(F))))
Base.iszero(e::SurfaceElement) = iszero(e.area)

area(e::SurfaceSimplex) = norm(e.dsigma)
area(e::SurfaceElement) = e.area

"""
    merge_element(acc::SurfaceElement, e)

Absorb `e`, a `SurfaceSimplex` or another `SurfaceElement`, into
`acc`: the area covectors and areas add, the centroid and fields become the
area-weighted means. It is the step function of a fold, so merging the
simplices of one cell is

    surface_simplices(merge_element, zero(element_type(D, fields)), cornerT, cell, Tfo; fields)

and passing the result as `init` for the next cell merges a block of cells.
Merging is associative: merged elements can be merged again with the same
result as merging all their simplices at once.
"""
@inline function merge_element(acc::SurfaceElement, e)
    # Inlined into the fold, the merge costs almost nothing; called, it costs a third of the pass.
    w = area(e)
    iszero(w) && return acc
    total = acc.area + w
    a, b = acc.area / total, w / total
    SurfaceElement(a * acc.centroid + b * e.centroid, acc.dsigma + e.dsigma, total,
                   map((p, q) -> a * p + b * q, acc.fields, e.fields))
end

"""
    flatness(e::SurfaceElement)

`|dsigma| / area`, between 0 and 1. It is 1 when every merged simplex has the
same normal, and about `1 - θ²/8` for two equal simplices bent by an angle `θ`.
Simplices of two opposite sheets partly cancel in `dsigma`, so a low value also
flags a merge across separate parts of the surface.
"""
flatness(e::SurfaceElement) = norm(e.dsigma) / e.area

"""
    surface_element(cornerT, cell::Cell, Tfo; fields=NamedTuple(), reference=ReferenceGrid(Val(D)))
    surface_element(acc::SurfaceElement, cornerT, cell::Cell, Tfo; ...)
    surface_element(cornerTs, cells, Tfo; fields=nothing, reference=ReferenceGrid(Val(D)))

The surface `T = Tfo` inside one cell as a single `SurfaceElement`: all the
cell's `SurfaceSimplex`es merged. It takes the same arguments as
`surface_simplices`; if the surface does not cross the cell the result `iszero`.

The second form merges the cell into the element `acc`, so cells can be added
one at a time, e.g. while streaming. The third form merges several cells at
once: `cells` is an array or tuple of `Cell`s, `cornerTs` a collection with
the corner values of each, in the same order, and `fields`, if given, a
collection of one NamedTuple of corner values per cell.
All three forms give the same element for the same cells. In two unit squares
stacked along `x`, with `T = 1.5 - t` and `Tfo = 1`:

    cornerT = (1.5, 0.5, 1.5, 0.5)
    cells = [Cell((0.0, 0.0), (1.0, 1.0)), Cell((0.0, 1.0), (1.0, 1.0))]
    e = surface_element([cornerT, cornerT], cells, 1.0)
    e.centroid, e.dsigma, e.area     # (0.5, 1.0), (2.0, 0.0), 2.0
"""
surface_element(cornerT, cell::Cell{D}, Tfo::Real; fields::NamedTuple=NamedTuple(), kwargs...) where {D} =
    surface_element(zero(element_type(D, fields)), cornerT, cell, Tfo; fields, kwargs...)
surface_element(acc::SurfaceElement{D}, cornerT, cell::Cell{D}, Tfo::Real; kwargs...) where {D} =
    surface_simplices(merge_element, acc, cornerT, cell, Tfo; kwargs...)

const Cells{D} = Union{AbstractArray{<:Cell{D}},Tuple{Vararg{Cell{D}}}}

function surface_element(cornerTs, cells::Cells{D}, Tfo::Real; fields=nothing,
                         reference::ReferenceGrid{D}=ReferenceGrid(Val(D))) where {D}
    cell_fields = something(fields, Iterators.repeated(NamedTuple()))
    acc = zero(element_type(D, first(cell_fields)))
    for (cornerT, cell, f) in zip(cornerTs, cells, cell_fields)
        acc = surface_element(acc, cornerT, cell, Tfo; fields=f, reference)
    end
    acc
end

"""
    surface_elements(temperature, coords, Tfo; fields=NamedTuple())

Like `surface_simplices` on a grid, but returns one `SurfaceElement` per cell
crossed by the surface: the cell's simplices merged, like the element
Cornelius returns per cell. For larger elements, merge blocks of cells with
the multi-cell `surface_element`, or merge elements with `merge_element`;
merging is associative, so both give the same element. The geometry still
comes from the full grid, and the quadrature error grows like the square of
the block width.
"""
function surface_elements(temperature::AbstractArray{<:Real,D},
                           coords::Tuple{Vararg{AbstractVector,D}}, Tfo::Real;
                           fields::NamedTuple=NamedTuple()) where {D}
    check_inputs(temperature, coords, fields)
    reference = ReferenceGrid(Val(D))
    elements = element_type(D, fields)[]

    for I in CartesianIndices(size(temperature) .- 1)
        e = surface_element(cell_corners(temperature, I), Cell(coords, I), Tfo;
                            fields=map(A -> cell_corners(A, I), fields), reference)
        iszero(e) || push!(elements, e)
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
