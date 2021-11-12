"""
Struct that defines a problem that will be sent to NOMAD.jl.
TODO: Docs string
"""
mutable struct ParameterOptimizationProblem{S,F1,F2}
    nomad::Union{Nothing,NomadProblem}
    solver::S
    black_box::F1
    surrogate_model::Union{Nothing,F2}
    use_surrogate::Bool
end

# TODO: Add Parametric type to solver (e.g Abstract Solver)
function ParameterOptimizationProblem(
    solver::S,
    black_box::F1,
    surrogate_model::Union{Nothing,F2} = nothing,
    use_surrogate = false,
) where {S<:LBFGSSolver,F1,F2}
    ParameterOptimizationProblem(nothing, solver, black_box, surrogate_model, use_surrogate)
end

function create_nomad_problem!(
    param_opt_problem::ParameterOptimizationProblem{S,F1,F2},
    black_box_kwargs::Dict{Symbol,T};
    kwargs...,
) where {S<:LBFGSSolver,F1,F2,T}
    # eval function:
    function eval_function(
        v::AbstractVector{Float64};
        problem = param_opt_problem,
        bb_kwargs = black_box_kwargs,
    )
        eval_fct(v, problem, [problem.solver.parameters]; bb_kwargs...)
    end

    solver_params = param_opt_problem.solver.parameters
    nomad = NomadProblem(
        length(solver_params),
        1,
        ["OBJ"],
        eval_function;
        input_types = input_types(solver_params),
        granularity = granularities(solver_params),
        lower_bound = lower_bounds(solver_params),
        upper_bound = upper_bounds(solver_params),
    )
    set_nomad_options!(nomad.options; kwargs...)
    param_opt_problem.nomad = nomad
end

# define eval function here: 
function eval_fct(
    v::AbstractVector{Float64},
    problem::ParameterOptimizationProblem{S,F1,F2},
    bb_args::Vector;
    kwargs...,
) where {S<:LBFGSSolver,F1,F2}
    !isempty(bb_args) ||
        error("No algorithmic parameters given to pass as arguments to black box.")
    algorithmic_params = problem.solver.parameters
    use_surrogate = problem.use_surrogate
    black_box = problem.black_box
    surrogate_model = problem.surrogate_model
    success = false
    count_eval = false
    black_box_output = [typemax(Float64)]
    try
        [set_default!(param, param_value) for(param, param_value) in zip(algorithmic_params, v)]
        if use_surrogate
            black_box_output = surrogate_model(bb_args...; kwargs...)
        else
            black_box_output = black_box(bb_args...; kwargs...)
        end
        success = true
        count_eval = true
    catch exception
        println("exception occured while solving:\t $exception")
    finally
        return success, count_eval, black_box_output
    end
end

function set_nomad_options!(options::NomadOptions; kwargs...)
    for (field, value) in Dict(kwargs)
        setfield!(options, field, value)
    end
end

# Function that validates a parameter optimization problem
function check_problem(p::ParameterOptimizationProblem{S,F1,F2}) where {S,F1,F2}
    @assert !p.use_surrogate || !isnothing(p.surrogate_model) "error: cannot use surrogate model if no surrogate model is defined"
end

function solve_with_nomad!(problem::ParameterOptimizationProblem)
    check_problem(problem)
    solve(problem.nomad, current_param_values(problem.solver.parameters))
end
