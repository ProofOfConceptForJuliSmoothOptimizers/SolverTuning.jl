include(joinpath(@__DIR__, "constants.jl"))
include(joinpath(@__DIR__,"types.jl"))
include(joinpath(@__DIR__, "parameter_search.jl"))

params = Vector{Parameter}()
for param_dict ∈ get_parameters("trunk.jl")
    push!(params, Parameter(param_dict))
end

for p ∈ params
    show(p)
end