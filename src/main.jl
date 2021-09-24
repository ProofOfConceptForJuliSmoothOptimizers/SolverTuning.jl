import Base.show
using JSON

include(joinpath(@__DIR__, "constants.jl"))
include(joinpath(@__DIR__, "parameters.jl"))
include(joinpath(@__DIR__, "constraints.jl"))
include(joinpath(@__DIR__, "parameter_search.jl"))


params = Vector{Parameter}()
for param_dict ∈ get_parameters("trunk.jl")
    push!(params, Parameter(param_dict))
end

for p ∈ params
    show(p)
end

dict = Dict{Any,Any}()
open("constraints.json") do f
    global dict = JSON.parse(read(f, String))
end

for constraint in dict
    show(LinearConstraint(constraint))
end
