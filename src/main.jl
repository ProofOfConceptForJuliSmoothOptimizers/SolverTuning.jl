# Distributed computing
using ClusterManagers, Distributed

include("setup_workers.jl")

# define params:
mem = AlgorithmicParameter(5, IntegerRange(1, 100), "mem")
τ₁ = AlgorithmicParameter(Float64(0.99), RealInterval(Float64(1.0e-4), 1.0), "τ₁")
scaling = AlgorithmicParameter(true, BinaryRange(), "scaling")
bk_max = AlgorithmicParameter(25, IntegerRange(10, 30), "bk_max")
lbfgs_params = [mem, τ₁, scaling, bk_max]

"""
Define blackbox:
1. problems
2. function (default function available)
3. black box positional args
4. black box keyword arguments
5. solver
"""
# define solver:

#define problems
problem_generator = (MathOptNLPModel(eval(problem)(), name=string(problem)) for problem ∈ filter(x -> x != :PureJuMP, names(OptimizationProblems.PureJuMP)))
problems = collect(Iterators.filter(x -> unconstrained(x) && get_nvar(x) ≥ 1 && get_nvar(x) ≤ 100, problem_generator))
solver = LBFGSSolver(first(problems), lbfgs_params)
args = []
kwargs = Dict{Symbol, Any}()
black_box = BlackBox(solver, args, kwargs, problems)

# define problem suite
param_optimization_problem =
  ParameterOptimizationProblem(black_box)

# named arguments are options to pass to Nomad
create_nomad_problem!(
  param_optimization_problem;
  display_all_eval = true,
  max_time = 18000,
)

# Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
println(result)

rmprocs(workers())
