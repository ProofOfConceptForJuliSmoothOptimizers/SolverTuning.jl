abstract type AbstractParameter{T} end
abstract type AbstractSolverParameter{T} <: AbstractParameter{T} end
abstract type AbstractHyperParameter{T} <: AbstractParameter{T} end
mutable struct AlgorithmicParameter{T, D <: AbstractDomain{T}} <: AbstractHyperParameter{T}
  default::T
  domain::D
  name::String
  granularity::Float64
  function AlgorithmicParameter(default::T, domain::AbstractDomain{T}, name::String) where {T}
    check_default(domain, default)
    granularity = default_granularity(eltype(domain))
    new{T, typeof(domain)}(default, domain, name, granularity)
  end
end
default(parameter::AlgorithmicParameter{T}) where {T} = parameter.default
domain(parameter::AlgorithmicParameter{T}) where {T} = parameter.domain
name(parameter::AlgorithmicParameter{T}) where {T} = parameter.name
granularity(parameter::AlgorithmicParameter{T}) where {T} = parameter.granularity
nomad_type(::Type{T}) where {T <: Real} = "R"
nomad_type(::Type{T}) where {T <: Integer} = "I"
nomad_type(::Type{T}) where {T <: Bool} = "B"

function check_default(domain::AbstractDomain{T}, new_value::T) where {T}
  new_value ∈ domain || error("default value should be in domain")
end

function set_default!(parameter::AlgorithmicParameter{T}, new_value::Float64) where {T}
  if nomad_type(eltype(domain(parameter))) == "I"
    new_value = round(Int64, new_value)
  end
  if nomad_type(eltype(domain(parameter))) == "B"
    new_value = convert(Bool, round(Int64, new_value))
  end
  check_default(parameter.domain, new_value)
  parameter.default = new_value
end

function default_granularity(input_type::T) where {T}
  (nomad_type(input_type) == "R") && return zero(Float64)
  return one(Float64)
end

# find param by name
function find(parameters::AbstractVector{P}, param_name) where {P <: AbstractHyperParameter}
  idx = findfirst(p -> p.name == param_name, parameters)
  isnothing(idx) || return parameters[idx]
  return
end
# Function that returns lower bounds of each param:
function lower_bounds(parameters::AbstractVector{P}) where {P <: AbstractHyperParameter}
  [Float64(lower(domain(p))) for p ∈ parameters]
end
# Function that returns upper bounds of each param:
function upper_bounds(parameters::AbstractVector{P}) where {P <: AbstractHyperParameter}
  [Float64(upper(domain(p))) for p ∈ parameters]
end
# Function that returns a vector of current param values:
function current_param_values(parameters::AbstractVector{P}) where {P <: AbstractHyperParameter}
  [Float64(default(p)) for p ∈ parameters]
end
# function that returns granularity vector (for all params depending on types):
function granularities(parameters::AbstractVector{P}) where {P <: AbstractHyperParameter}
  [p.granularity for p ∈ parameters]
end
# function that returns a vector of input types
function input_types(parameters::AbstractVector{P}) where {P <: AbstractHyperParameter}
  return [nomad_type(eltype(domain(p))) for p ∈ parameters]
end
