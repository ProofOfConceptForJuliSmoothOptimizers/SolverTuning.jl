include(joinpath(@__DIR__, "..", "..", "src", "domains.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "parameters.jl"))

# Defining one parameter at a time
# See here for the range: https://en.wikipedia.org/wiki/Limited-memory_BFGS
# mem_domain = IntegerRange(4, 10)
mem = AlgorithmicParameter(5, IntegerRange(4, 10), "mem")

lbfgs_params = AlgorithmicParameters([mem])
