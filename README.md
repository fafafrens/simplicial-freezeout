# Simplicial freeze-out: geometry + fully commented Julia code

This package is a tutorial companion to the standalone generic-grid Julia
module discussed for Fluidum.jl. It **does not depend on Fluidum**.

## Files

- **`Simplicial_Freezeout_Explained.pdf`**: illustrated 8-page mathematical and algorithmic notes.
- **`Simplicial_Freezeout_Explained.tex`**: editable LaTeX source of the notes.
- **`NOTES.md`**: Markdown source of the notes.
- **`SimplicialFreezeOut.jl`**: the reference algorithm, with docstrings carrying worked examples. Entry points:
  - `isosurface(T, coords, Tfo; fields)` on a full spacetime grid;
  - `cell_isosurface` on one hypercube, as a fold over its pieces (`cell_isosurface(f, init, cornerT, cell, Tfo)`), a callback, an in-place push, or a returned vector;
  - `cell_corners`, `slab_corners` and `Cell` to gather a cell's corner values and geometry, e.g. from two consecutive hydro time slices;
  - `ReferenceGrid(Val(D))` for inspecting the reference geometry; `integrate_surface` for centroid quadrature.
- **`test_simplicial_freezeout.jl`**: analytic tests (planar isotherms, affine field interpolation), a closed spherical isotherm (watertightness, orientation, divergence theorem), the single-cell API, streaming from two time slices, and the fold forms.
- **`compare_cornelius.jl`**: comparison with Cornelius (P. Huovinen, H. Petersen, arXiv:1206.3371) on the same 4D field. The Cornelius source is not included: place `cornelius.f90` next to the script and it is compiled with gfortran on first use. On a spherical isotherm both give the same integrated dΣ and enclosed volume; Cornelius returns one element per crossing cell and is about twice as fast at extraction.
- **`geometry_rectangle.png`, `geometry_tetrahedron.png`**: explanatory geometry figures.

## Run

From this folder:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # once, installs dependencies
julia --project=. test_simplicial_freezeout.jl
julia --project=. compare_cornelius.jl   # needs cornelius.f90 and gfortran
```

The non-standard dependencies are StaticArrays.jl and Combinatorics.jl, declared in `Project.toml`.

## Main geometry idea

In `D` spacetime dimensions:

1. A regular Cartesian cell has `2^D` grid corners.
2. Its reference Freudenthal subdivision has `D!` simplices, precomputed once.
3. Each simplex has `D+1` vertices: check their signs relative to `Tfo`.
4. A precomputed table, indexed by the hot/cold pattern of the vertices, lists the crossed edges and how to join them.
5. Interpolate the crossing points and join them into a `(D-1)`-simplex.
6. Compute an oriented **integrated** area covector and locally sample fluid fields.

`D=2` returns line segments, `D=3` triangles, `D=4` tetrahedra.

The input simplices and output surface elements must not be confused. In
particular, `D=3` input tetrahedra give triangle output elements, whereas
`D=4` input 4-simplices give tetrahedral output elements.

**Limitations:** This is a readable, piecewise-affine reference algorithm. It
uses centroid quadrature, does not handle extended exactly-on-isotherm
plateaus, and does not compute Cooper--Frye particle spectra itself (the fold
form of `cell_isosurface` lets you accumulate them on the fly). The notes in
`NOTES.md` and the PDF describe an earlier version of the code: orientation
there is taken from the temperature gradient and the cut cases are tabulated
by hand, whereas the module now orients toward the coldest vertex and
generates all cases from staircase triangulations.
