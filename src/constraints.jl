# Format l ≤ c(x) ≤ u to c(x) ≤ 0:
function format_constraints(
  param_opt_problem::ParameterOptimizationProblem{B, L},
) where {B <: BBModel, L <: AbstractLoadBalancer}
  nlp = param_opt_problem.nlp
  lcon = nlp.meta.lcon
  ucon = nlp.meta.ucon
  v = values(nlp.parameter_set)
  cons_values = nlp.c(v)

  # constraints that are from lower_bound:
  lower_cons = [l - c_value for (l, c_value) in zip(lcon, cons_values)]
  upper_cons = [c_value - u for (u, c_value) in zip(ucon, cons_values)]

  return [lower_cons..., upper_cons...]
end
