abstract type AbstractParameter{T} end
mutable struct AlgorithmicParameter{T,D<:AbstractDomain{T}} <: AbstractParameter{T}
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

function set_default!(parameter::AlgorithmicParameter{T}, new_value::T) where {T}
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

abstract type AbstractParameters{P} end
struct AlgorithmicParameters{P<:AlgorithmicParameter} <: AbstractParameters{P}
    params::Dict{String,P}
    function AlgorithmicParameters(vec::Vector{P}) where {P<:AlgorithmicParameter}
        new{P}(Dict{String,P}(p.name => p for p in vec))
    end
end
has_param(parameters::AlgorithmicParameters{P}, param_name::String) where {P <: AlgorithmicParameter} = haskey(parameters.params, param_name)
function get_param(parameters::AlgorithmicParameters{P}, param_name::String) where {P <: AlgorithmicParameter}
    has_param(parameters, param_name) || return
    return parameters.params[param_name]
end

function fetch(f::Function, parameters::AlgorithmicParameters{P}) where {P <: AlgorithmicParameter}
    return map(f, values(parameters.params))
end
# function that counts the number of parameter to tune
nb_params(parameters::AlgorithmicParameters{P}) where {P <: AlgorithmicParameter} = length(values(parameters.params))
# Function that returns lower bounds of each param:
function lower_bounds(parameters::AlgorithmicParameters{P}) where {P <: AlgorithmicParameter}
    return convert(Vector{Float64}, fetch(p-> lower_bound(p.domain), parameters))
end
# Function that returns upper bounds of each param:
function upper_bounds(parameters::AlgorithmicParameters{P}) where {P <: AlgorithmicParameter}
    return convert(Vector{Float64}, fetch(p-> upper_bound(p.domain), parameters))
end
# Function that returns a vector of current param values:
function current_param_values(parameters::AlgorithmicParameters{P}) where {P <: AlgorithmicParameter}
    return convert(Vector{Float64}, fetch(p-> default(p), parameters))
end
# function that returns granularity vector (for all params depending on types):
function granularities(parameters::AlgorithmicParameters{P}) where {P <: AlgorithmicParameter}
    return fetch(p->granularity(p), parameters)
end

# function that returns a vector of input types
function input_types(parameters::AlgorithmicParameters{P}) where {P <: AlgorithmicParameter}
    return fetch(p->input_type(p), parameters)
end


mutable struct ParameterOptimizationProblem
    
end