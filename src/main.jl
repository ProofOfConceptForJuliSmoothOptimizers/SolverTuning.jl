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

@everywhere function get_problem(problem_name::Symbol)
  MathOptNLPModel(eval(problem_name)(),name=string(problem_name))
end

#define problems
problems = Dict{Symbol, Union{Nothing,AbstractExecutionStats}}()
for p in filter(x -> x != :PureJuMP, names(OptimizationProblems.PureJuMP))
  nlp = MathOptNLPModel(eval(p)(), name=string(p))
  if unconstrained(nlp) && get_nvar(nlp) ≥ 1 && get_nvar(nlp) ≤ 100
    problems[p] = nothing
  end
end
solver = LBFGSSolver(get_problem(first(keys(problems))), lbfgs_params)
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
