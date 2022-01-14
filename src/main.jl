# Distributed computing
using ClusterManagers, Distributed

include("setup_workers.jl")

# define params:
# LBFGS:
mem = AlgorithmicParameter(5, IntegerRange(1, 100), "mem")
τ₁ = AlgorithmicParameter(Float64(0.99), RealInterval(Float64(1.0e-4), 1.0), "τ₁")
scaling = AlgorithmicParameter(true, BinaryRange(), "scaling")
bk_max = AlgorithmicParameter(25, IntegerRange(10, 30), "bk_max")
lbfgs_params = [mem, τ₁, scaling, bk_max]

# TRUNK
# initial_radius = AlgorithmicParameter(1.0, RealInterval(0.5, 5.0), "initial_radius")
# acceptance_threshold = AlgorithmicParameter(1.0e-4, RealInterval(0.0, 0.5, true, true), "acceptance_threshold")
# increase_threshold = AlgorithmicParameter(0.95, RealInterval(0.5, 1.0, false, true), "increase_threshold")
# monotone = AlgorithmicParameter(true, BinaryRange(), "monotone")
# nm_itmax = AlgorithmicParameter(25, IntegerRange(10, 50), "nm_itmax")
# trunk_params = [initial_radius, acceptance_threshold, increase_threshold, monotone, nm_itmax]
"""
Define blackbox:
1. problems
2. function (default function available)
3. black box positional args
4. black box keyword arguments
5. solver
"""

#define problems
problems = (MathOptNLPModel(eval(p)(),name=string(p)) for p ∈ filter(x -> x != :PureJuMP, names(OptimizationProblems.PureJuMP)))

problems = Iterators.filter(p -> unconstrained(p) &&  5 ≤ get_nvar(p) ≤ 1000 && get_minimize(p), problems )

set_worker_problems(problems)

# Define solver
solver = LBFGSSolver(first(problems), lbfgs_params)
# solver = TrunkSolver(first(problems), trunk_params)

# define user's blackbox:
function my_black_box(args...;kwargs...)
  solver_results = eval_solver(lbfgs, args...;kwargs...)
  bmark_results = Dict(i=> b for (i,(b,s)) in solver_results)
  stats_results = Dict(i => s for (i,(b,s)) in solver_results)
  times = sum((median(bmark).time/1.0e9) for bmarks ∈ values(bmark_results) for bmark in bmarks)
  return [times]
end
kwargs = Dict{Symbol, Any}(:verbose => false)
black_box = BlackBox(solver, my_black_box, kwargs)

# define problem suite
param_optimization_problem =
  ParameterOptimizationProblem(black_box)

# named arguments are options to pass to Nomad
create_nomad_problem!(
  param_optimization_problem;
  display_all_eval = true,
  max_time = 180000,
  max_bb_eval = 2,
  display_stats = ["BBE", "EVAL", "SOL", "OBJ"],
)

# Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
println(result)
println("Best feasible parameters: $(result.x_best_feas)")
rmprocs(workers())
