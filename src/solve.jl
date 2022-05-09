export solve, solve_with_nomad

function solve_with_nomad!(
  problem::ParameterOptimizationProblem{T, S, B, L},
) where {T, S, B <: AbstractBBModel, L <: AbstractLoadBalancer}
  check_problem(problem)
  return solve(problem.nomad, [Float64(xᵢ) for xᵢ in problem.x])
end

function solve_with_nomad(bbmodel::AbstractBBModel; kwargs...)
  param_optimization_problem = ParameterOptimizationProblem(bbmodel)
  let result = nothing, best_params=nothing
    try

      # named arguments are options to pass to Nomad
      create_nomad_problem!(param_optimization_problem; kwargs...)
      result = solve_with_nomad!(param_optimization_problem)
      result = result.x_best_feas
      best_params = (; zip(keys(param_opt_problem.nlp.meta.x_n), result)...)
      @info "Best feasible parameters: $best_params"
    catch e
      @error "Error occured while running NOMAD: $e"
      best_params = (; zip(keys(param_opt_problem.nlp.meta.x_n), param_optimization_problem.x)...)
    finally
      rmprocs(workers())
      return best_params
    end
  end
end
