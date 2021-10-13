using NOMAD
abstract type AbstractParameter{T} end

abstract type AbstractSolverParameter{T} <: AbstractParameter{T} end
abstract type AbstractHyperParameter{T} <: AbstractParameter{T} end
mutable struct AlgorithmicParameter{T,D<:AbstractDomain{T}} <: AbstractHyperParameter{T}
    default::T
    domain::D
    name::String
    input_type::String
    granularity::Float64
    function AlgorithmicParameter(
        default::T,
        domain::AbstractDomain{T},
        name::String,
    ) where {T}
        check_default(domain, default)
        input_type = get_input_type(domain)
        granularity = default_granularity(input_type)
        new{T,typeof(domain)}(default, domain, name, input_type, granularity)
    end
end
default(parameter::AlgorithmicParameter{T}) where {T} = parameter.default
domain(parameter::AlgorithmicParameter{T}) where {T} = parameter.domain
name(parameter::AlgorithmicParameter{T}) where {T} = parameter.name
input_type(parameter::AlgorithmicParameter{T}) where {T} = parameter.input_type
granularity(parameter::AlgorithmicParameter{T}) where {T} = parameter.granularity

function check_default(domain::AbstractDomain{T},new_value::T) where T
    new_value ∈ domain || error("default value should be in domain")
end

function set_default!(parameter::AlgorithmicParameter{T}, new_value::Float64) where T
    new_value = !(input_type(parameter) == "R") ? convert(Int64, new_value) : new_value
    check_default(parameter.domain, new_value)
    parameter.default = new_value
end

function get_input_type(domain::AbstractDomain{T}) where {T}
    isa(domain, RealDomain) && return "R"
    # Check if domain is a BinaryRange:
    isa(domain, IntegerRange) && (domain.lower == zero(T) && domain.upper == one(T)) && return "B"
    return "I"
end

function default_granularity(input_type::String)
    (input_type == "R") && return zero(Float64)
    return one(Float64)
end

# function that counts the number of parameter to tune
nb_params(parameters::AbstractVector{P}) where {P<:AbstractHyperParameter} = length(parameters)

# find param by name
function find(parameters::AbstractVector{P}, param_name) where {P<:AbstractHyperParameter}
    idx = findfirst(p->p.name == param_name, parameters)
    isnothing(idx) || return parameters[idx]
    return nothing
end
# Function that returns lower bounds of each param:
function lower_bounds(parameters::AbstractVector{P}) where {P<:AbstractHyperParameter}
    return convert(Vector{Float64}, [lower_bound(p.domain) for p ∈ parameters])
end
# Function that returns upper bounds of each param:
function upper_bounds(parameters::AbstractVector{P}) where {P<:AbstractHyperParameter}
    return convert(Vector{Float64}, [upper_bound(p.domain) for p ∈ parameters])
end
# Function that returns a vector of current param values:
function current_param_values(parameters::AbstractVector{P}) where {P<:AbstractHyperParameter}
    return convert(Vector{Float64}, [p.default for p ∈ parameters])
end
# function that returns granularity vector (for all params depending on types):
function granularities(parameters::AbstractVector{P}) where {P<:AbstractHyperParameter}
    return [p.granularity for p ∈ parameters]
end
# function that returns a vector of input types
function input_types(parameters::AbstractVector{P}) where {P<:AbstractHyperParameter}
    return [input_type(p) for p ∈ parameters]
end

mutable struct ParameterOptimizationProblem
    nomad::NomadProblem
    solver::Any
    bb_output::Function
    obj::Function
end

function bb_output(solver_params::AbstractVector{P}) where {P<:AbstractHyperParameter}
    [@elapsed unconstrained_nlp(nlp ->lbfgs(nlp, solver_params))]
end

function ParameterOptimizationProblem(solver::Any)
    parameters = solver.p
    function obj(v::AbstractVector{Float64}; solver = solver)
        println("Updating params!")
        parameters = solver.p
        [set_default!(param, param_value) for (param, param_value) in zip(parameters, v)]
        return true, true, bb_output(parameters)   
    end
    nomad = NomadProblem(nb_params(parameters), 1, ["OBJ"], obj;
                input_types = input_types(parameters),
                granularity = granularities(parameters),
                lower_bound = lower_bounds(parameters),
                upper_bound = upper_bounds(parameters)
            )
    return ParameterOptimizationProblem(nomad, solver, bb_output, obj)
end

# Nomad:
function minimize_with_nomad!(problem::ParameterOptimizationProblem)
    println("Entering NOMAD!")
    solve(problem.nomad, current_param_values(problem.solver.p))
end