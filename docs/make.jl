using Documenter
using Filthy

makedocs(
    modules = [Filthy],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://gragusa.github.io/Filthy.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting_started.md",
            "Custom Models" => "tutorials/custom_models.md",
            "Dynamic Factor Models" => "tutorials/dynamic_factor.md",
        ],
        "API Reference" => [
            "Core Functions" => "api/core.md",
            "DSL & Templates" => "api/dsl.md",
            "Matrix Helpers" => "api/matrix_helpers.md",
            "Optimization & Bayesian" => "api/optimization.md",
        ],
    ],
    sitename = "Filthy.jl",
    authors = "Giuseppe Ragusa",
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/gragusa/Filthy.jl.git",
    devbranch = "master",
    push_preview = true,
)
