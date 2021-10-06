include("domains.jl")
include("parameters.jl")

mem = AlgorithmicParameter(5, IntegerRange(4, 10), "mem")
tr_radius = AlgorithmicParameter(eps(Float64), RealInterval(0.0, 10.0), "tr_radius")
cts = AlgorithmicParameter(5.0, RealInterval(0.0, 10.0), "cts")

yes_no = BinarySet(Set([1, 0]))
# # testing with on param
vec = [mem, tr_radius]

params = Dict(p.name => p for p in vec)

other_param = AlgorithmicParameters([cts])
other_param = AlgorithmicParameters(vec)
