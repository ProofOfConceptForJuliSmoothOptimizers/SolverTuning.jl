abstract type AbstractParameter{T} end
mutable struct AlgorithmicParameter{T,D<:AbstractDomain{T}} <: AbstractParameter{T}
    default::T
    domain::D
    name::String
    function AlgorithmicParameter(
        default::T,
        domain::AbstractDomain{T},
        name::String,
    ) where {T}
        default ∈ domain || error("default value should be in domain")
        new{T,typeof(domain)}(default, domain, name)
    end
end

# TODO: Check that this method should only apply for the AlgorithmicParameter type
function set_default!(parameter::AlgorithmicParameter{T}, new_value::T) where {T}
    new_value ∈ parameter.domain || error("default value should be in domain")
    parameter.default = new_value
end

abstract type AbstractParameters{P} end

mutable struct AlgorithmicParameters{P} <: AbstractParameters{P}
    params::Dict{String,P}
    function AlgorithmicParameters(vec::Vector{P}) where {P}
        new{P}(Dict{String,P}(p.name => p for p in vec))
    end
end
