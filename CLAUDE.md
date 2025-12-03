# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Filthy.jl is a Julia package for Linear State Space models (Kalman filtering and smoothing). It supports both exact and inexact diffuse initialization, with implementations optimized for scalar, matrix, and static array types.

## Build and Test Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run tests interactively
julia --project test/runtests.jl

# Load package in REPL
julia --project -e 'using Filthy'

# Run feature-specific tests (when using TestItems.jl)
TI_TAGS=filter,smoke julia --project -e 'using Pkg; Pkg.test()'
```

**Note:** Tests require `Nile.csv` data file in the test directory and depend on CSV.jl.

## Architecture

### Core Types (`src/types.jl`)

- **`LinearStateSpace`** - Main model container holding parameters, filtered/smoothed states, initial values, and caches
- **`KFParms`** - State space parameters (Z, H, T, R, Q matrices)
- **`KFFiltered`** - Filtered state estimates (a, P, Pinf, Pstar for diffuse case)
- **`KFSmoothed`** - Smoothed state estimates
- **`KFCache`** - Preallocated workspace for filter computations
- **`Masked`** - Parameter masking for optimization (tracks free vs fixed parameters)
- **`SSMOptim`** - MathProgBase evaluator for maximum likelihood estimation

### Module Organization

- `src/Filthy.jl` - Main module, constructors, high-level API (`fit`, `simulate`, `smooth!`)
- `src/filter.jl` - Kalman filter implementation (`filter!`, `filterstep`, diffuse variants)
- `src/methods.jl` - Helper functions, type utilities, likelihood computation
- `src/mpb.jl` - MathProgBase interface for optimization

### Key Design Patterns

**Multiple dispatch on array types:** The package dispatches on `AbstractFloat`, `AbstractArray`, and `StaticArrays.StaticMatrix` to provide type-stable, allocation-free paths for different use cases.

**Diffuse initialization:** Handled via `Val{true/false}` dispatch in `onlinefilterstep!` to separate diffuse and non-diffuse filter paths.

**GrowableArrays:** Filter results stored in `GrowableArray` containers for efficient online filtering.

### State Space Model Formulation

```
Observation:  yₜ = Z αₜ + εₜ,  εₜ ~ N(0, H)
State:        αₜ₊₁ = T αₜ + R ηₜ,  ηₜ ~ N(0, Q)
```

Constructor supports three parametrizations:
1. Scalar (all parameters are `Float64`)
2. Matrix (`Array{Float64}`)
3. Static (`StaticArrays.SMatrix/SVector`)

### Dependencies

- `StaticArrays` - Fixed-size arrays for performance
- `GrowableArrays` - Dynamically resizable arrays for online filtering
- `ForwardDiff` - Automatic differentiation for ML estimation
- `MathProgBase` - Optimization interface
- `Parameters` - `@unpack` macro for struct field extraction

---

## Julia Code Style Guidelines

This project values clarity **and** speed. When proposing changes, follow these rules, with short justifications and minimal examples when you make edits.

## Structure & type stability

- **Put hot code in functions; avoid global work.** Globals hurt type inference and force allocations. Prefer passing values as arguments. If a global truly is constant, mark it `const`.  

- **Make functions type-stable.** Ensure a function returns the **same concrete type** for all input paths. Use `zero(x)`, `oneunit(x)`, or `oftype(x,y)` to avoid mixing `Int`/`Float64`, etc.

- **Don’t change local variable types mid-function.** Initialize with the final type (`x = 1.0` not `1` if it will become float).

- **Avoid abstractly-typed containers and fields.**  
  Bad: `Vector{Real}`, `a::AbstractFloat`, `f::Function`.  
  Good: parametric concrete fields/containers: `Vector{Float64}`, `struct W{F}; f::F; end`.

- **Know when Julia won’t specialize.** Methods that take plain `Type`, `Function`, or varargs may avoid specialization. Add explicit type parameters (e.g. `g(t::Type{T}) where T`).

- **Use function barriers.** Separate setup (where types may be uncertain) from tight kernels that operate on concretely-typed arguments.

- **Diagnose types.** Use `@code_warntype` and fix red (non-concrete) spots first; unions with `missing`/`nothing` are common culprits.

## Allocations & arrays

- **Preallocate outputs** for repeated work; prefer `foo!(out, ...)` forms. For elementwise ops use in-place fused broadcasting: `y .= 3 .* x .+ 2`.

- **Use views for light slicing.** `@views` or `view(A, ...)` avoids copying slices when you only do a little work on them. If you’ll reuse heavily, a copy can be faster than a non-contiguous view.

- **Fuse broadcasts (dots) judiciously.** Write `@. 3x^2 + 4x + 7x^3` or sprinkle dots to avoid temporaries. But “unfuse” subexpressions that repeat unnecessarily (cache them in a temp).

- **Access in column-major order.** Make the first index the fastest-varying in loops; prefer column operations to row operations.

- **Small fixed-size arrays:** consider `StaticArrays.jl` to avoid heap allocations and enable unrolled code.

## Tools & measurement

- **Measure with `@time`, `@allocated`, and BenchmarkTools.jl** (for robust timing). Watch **allocations**; unexpected ones usually mean type instability or accidental temporaries.

- **Profile before optimizing.** Use the profiler (and tools like ProfileView) to find real bottlenecks. JET.jl can flag performance pitfalls.

## Annotations (use carefully)

- **`@inbounds`** for bounds-check-free loops when indices are provably safe.  
- **`@simd`** only when loop iterations are independent.  
- **`@fastmath`** allows re-association & non-IEEE shortcuts; results may change (avoid in numerically delicate code).  
- Prefer `eachindex(x)` or `LinearIndices(x)` over `1:n` for generic arrays.

## Miscellaneous micro-optimizations

- Prefer `x + y + z` over `sum([x, y, z])`.  
- Use `abs2(z)` over `(abs(z))^2` for complex numbers.  
- Prefer `div/fld/cld` family for integer division semantics.  
- Avoid splatting large tuples/varargs in hot paths.  
- For I/O, avoid string interpolation that creates temporaries; use multiple `print/println` arguments.

## Patterns to avoid (common footguns)

- Globals without `const`.
- Containers/fields typed as `Abstract...`, `Any`, or `Function` unless truly necessary.
- Returning different concrete types across branches.
- Heavy slicing without `@views`, or repeated broadcasts that recompute invariant parts.
- Row-major access patterns on `Array`s.
- Excessive “values-as-type-parameters” that explode method specializations.

## Testing strategy (TestItems.jl + feature-tagged tests)

We use **TestItems.jl** to author granular tests that can run **individually** or **by feature tag**. This enables fast, targeted feedback for humans and AI agents.

### 8.1 Authoring tests

- Write tests as `@testitem "Descriptive name" tags=[...feature tags...] begin ... end`.
- Each `@testitem` must be **independently executable**; it runs in a temporary module with `using Test` and `using MyPackage` already in scope (unless you set `default_imports=false`).
- Keep `@testitem`s colocated with code (inline in `src/…`) **or** in `test/`. Both are supported.

**Example (inline test with feature tags):**

```julia
module MyPackage
using TestItems
export foo

foo(x) = String(x)

@testitem "foo returns String for AbstractString" tags=[:api, :string, :smoke] begin
    @test foo("bar") == "bar"
    @test foo('x') == "x"
end

end
```

**Example (tests in `test/` with shared setup):**

```julia
# test/setup.jl
using TestItems
@testsnippet DBSetup begin
    # expensive fixture or common helpers
    const TEST_DSN = "memory://"
end
```

```julia
# test/foo_tests.jl
using TestItems, Test
include("setup.jl")

@testitem "db-backed foo path" tags=[:api, :db] setup=[DBSetup] begin
    @test !isempty(TEST_DSN)
    # … assertions …
end
```

### 8.2 Tag taxonomy (conventions)

Use **feature-oriented** tags so subsets are meaningful and stable:

* Domain features: `:filter`, `:smoother`, `:diffuse`, `:likelihood`, `:optimization`, `:constructor`
* Cross-cutting: `:smoke` (fast canary), `:regression`, `:slow`, `:perf`, `:gpu`, `:random`
* CI control: `:skipci` (opt-out of default CI), `:nightly` (run on nightly schedule only)

A test may carry multiple tags; prefer **small**, **orthogonal** tag sets.

### 8.3 Running tests (humans, CI, and AI agents)

We integrate **TestItemRunner.jl** with `Pkg.test` and expose simple environment variables for filtering.

**`Project.toml` (test dependencies):**

```toml
[extras]
TestItemRunner = "f7a19654-30c6-45f8-84f8-3e4bd9b59689"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test", "TestItemRunner"]
```

**Typical invocations:**

* Run everything (excluding `:skipci` by default):

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

* Run only **feature-specific** tests (e.g., filter and smoother):

```bash
TI_TAGS=filter,smoother julia --project -e 'using Pkg; Pkg.test()'
```

* Run a named subset:

```bash
TI_NAME="diffuse" julia --project -e 'using Pkg; Pkg.test()'
```

* Include `:skipci` tests or run a single file:

```bash
TI_SKIPCI=false TI_FILE="foo_tests.jl" julia --project -e 'using Pkg; Pkg.test()'
```

**How the AI agent should act:**

* Determine relevant **feature tags** from the task (e.g., editing filter code → use `TI_TAGS=filter,smoke`).
* Execute `Pkg.test()` with `ENV` filters as above.
* If adding tests, tag them consistently and run only those tags to validate.

### 8.5 Sharing fixtures across items

* Use `@testsnippet` for per-item setup that runs **each time** the item runs.
* Use `@testmodule` for heavier, build-once fixtures reused across items/processes.

**Example:**

```julia
@testmodule HeavyData begin
    const REF = rand(10_000)  # built once per test process
end

@testitem "correlator uses shared ref" tags=[:signal, :perf] setup=[HeavyData] begin
    @test length(HeavyData.REF) == 10_000
end
```

### One-paragraph rule-of-thumb
**Make types and shapes obvious to the compiler, keep hot loops allocation-free, and measure everything.** Most slowdowns in Julia trace back to type instability or accidental allocations.
