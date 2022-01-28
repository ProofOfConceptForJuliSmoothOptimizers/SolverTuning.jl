using SolverTuning
using Documenter

DocMeta.setdocmeta!(
  SolverTuning,
  :DocTestSetup,
  :(using SolverTuning);
  recursive = true,
)

makedocs(;
  modules = [SolverTuning],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Abel Soares Siqueira <abel.s.siqueira@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/SolverTuning.jl/blob/{commit}{path}#{line}",
  sitename = "SolverTuning.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/SolverTuning.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/SolverTuning.jl",
  push_preview = true,
  devbranch = "main",
)
