export solve, solve_with_nomad

function solve_with_nomad!(
  problem::ParameterOptimizationProblem{T, S, B, L},
) where {T, S, B <: AbstractBBModel, L <: AbstractLoadBalancer}
  check_problem(problem)
  return solve(problem.nomad, [Float64(xᵢ) for xᵢ in problem.x])
end

function solve_with_nomad(bbmodel::AbstractBBModel; kwargs...)
  param_optimization_problem = ParameterOptimizationProblem(bbmodel)
  let result = nothing
    try

      # named arguments are options to pass to Nomad
      create_nomad_problem!(param_optimization_problem; kwargs...)
      result = solve_with_nomad!(param_optimization_problem)
      result = result.x_best_feas
    catch e
      @error "Error occured while running NOMAD: $e"
    finally
      @info ("Best feasible parameters: $(param_optimization_problem.x)")
      rmprocs(workers())
      return result
    end
  end
end
