include("domains.jl")
include("parameters.jl")

mem = AlgorithmicParameter(5, IntegerSet([1, 2, 3, 4, 5]), "mem")
tr_radius = AlgorithmicParameter(eps(Float64), RealInterval(0.0, 10.0), "tr_radius")

dci_params = AlgorithmicParameters([mem, tr_radius])
