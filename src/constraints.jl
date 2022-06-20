# Format l ≤ c(x) ≤ u to c(x) ≤ 0:
function format_constraints(
  param_opt_problem::ParameterOptimizationProblem{T, S, B, L},
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  nlp = param_opt_problem.nlp
  lcon = nlp.meta.lcon
  ucon = nlp.meta.ucon
  v = convert(Vector{Float64}, param_opt_problem.x)
  cons_values = nlp.c(v)

  # constraints that are from lower_bound:
  lower_cons = [l - c_value for (l, c_value) in zip(lcon, cons_values)]
  upper_cons = [c_value - u for (u, c_value) in zip(ucon, cons_values)]

  return [lower_cons..., upper_cons...]
end
