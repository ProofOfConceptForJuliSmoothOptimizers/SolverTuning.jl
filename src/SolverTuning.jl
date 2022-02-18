module SolverTuning

# stdlib.
using Pkg, LinearAlgebra, Printf, Distributed, Random

# JSO packages.
using NLPModels, SolverCore, SolverParameters
using JSOSolvers: AbstractOptSolver

# Nomad.
using NOMAD
using NOMAD: NomadOptions

# Specific imports.
using ClusterManagers
using BenchmarkTools
using BenchmarkTools: Trial
import BenchmarkTools.hasevals
import BenchmarkTools.prunekwargs
import BenchmarkTools.Benchmark
import BenchmarkTools.Parameters
import BenchmarkTools.run_result

# global vector containing problems for nomad.
worker_problems = Vector{AbstractNLPModel}()
# seed for random numbers:
Random.seed!(2017)

# include module components.
include("workers.jl")
include("benchmark_macros.jl")
include("black_box.jl")
include("load_balancer.jl")
include("nomad_interface.jl")
end
