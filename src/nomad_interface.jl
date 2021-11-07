"""
Struct that defines a problem that will be sent to NOMAD.jl.
Example: p = ParameterOptimizationProblem(solver)
Nomad options can also be provided
p = ParameterOptimizationProblem(solver; max_time=10, display_unsuccesful=true)
See NOMAD.jl documentation for more options.
"""
mutable struct ParameterOptimizationProblem{S,F1}
    nomad::NomadProblem
    solver::S
    black_box::F1
    substitute_model::Union{Nothing,F1}
    problems::AbstractVector{String}
end

# TODO: Add Parametric type to solver (e.g Abstract Solver)
function ParameterOptimizationProblem(solver::S,
                                    problems::AbstractVector{String},
                                    black_box::F1 = default_black_box,
                                    substitute_model::Union{Nothing,F1} = nothing;
                                    kwargs...) where {S<:LBFGSSolver, F1}
    parameters = solver.parameters
    # define eval function here: 
    function eval_fct(v::AbstractVector{Float64}; algorithmic_params::AbstractVector{P} = parameters) where {P<:AbstractHyperParameter}
        [set_default!(param, param_value) for (param, param_value) in zip(algorithmic_params, v)]
        println("current_param_values: $(current_param_values(algorithmic_params))")
        success = false
        count_eval = false
        black_box_output = [typemax(Float64)]
        try
            black_box_output = black_box(algorithmic_params, problems)
            success = true
            count_eval = true
        catch exception
            println("exception occured while solving:\t $exception")
        finally
            return success, count_eval, black_box_output 
        end
    end

    nomad = NomadProblem(
        length(parameters),
        1,
        ["OBJ"],
        eval_fct;
        input_types = input_types(parameters),
        granularity = granularities(parameters),
        lower_bound = lower_bounds(parameters),
        upper_bound = upper_bounds(parameters),
    )
    set_nomad_options!(nomad.options; kwargs...)
    return ParameterOptimizationProblem(nomad, solver, black_box, substitute_model, problems)
end

function set_nomad_options!(options::NomadOptions; kwargs...)
    for (field, value) in Dict(kwargs)
        setfield!(options, field, value)
    end
end

function default_black_box(solver_params::AbstractVector{P}, problems::AbstractVector{String}) where {P<:AbstractHyperParameter}
    max_time = 0.0
    for problem in problems
        nlp = CUTEstModel(problem)
        time_per_problem = @elapsed lbfgs(nlp, solver_params)
        max_time += time_per_problem
        finalize(nlp)
    end
    return [max_time]
end

# Nomad:
function solve_with_nomad!(problem::ParameterOptimizationProblem)
    println("Entering NOMAD!")
    solve(problem.nomad, current_param_values(problem.solver.parameters))
end
