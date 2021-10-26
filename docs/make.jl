using ParameterOptimization
using Documenter

DocMeta.setdocmeta!(ParameterOptimization, :DocTestSetup, :(using ParameterOptimization); recursive = true)

makedocs(;
  modules = [ParameterOptimization],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Abel Soares Siqueira <abel.s.siqueira@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/ParameterOptimization.jl/blob/{commit}{path}#{line}",
  sitename = "ParameterOptimization.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/ParameterOptimization.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/ParameterOptimization.jl",
  push_preview = true,
  devbranch = "main",
)
