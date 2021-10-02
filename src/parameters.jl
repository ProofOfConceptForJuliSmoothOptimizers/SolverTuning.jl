abstract type AbstractParameter{T} end
mutable struct AlgorithmicParameter{T<:Union{AbstractString,Real},D<:AbstractDomain{T}} <:
               AbstractParameter{T}
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
function set_default!(parameter::AlgorithmicParameter{T,D}, new_value::T) where {T,D}
    new_value ∈ parameter.domain || error("default value should be in domain")
    parameter.default = new_value
end


abstract type AbstractParameters end
mutable struct AlgorithmicParameters <: AbstractParameters
    params::Dict{AbstractString,AlgorithmicParameter}
end
function AlgorithmicParameters(parameters::Vector{AlgorithmicParameter})
    return AlgorithmicParameters(
        Dict{AbstractString,AlgorithmicParameter}(p.name => p for p ∈ parameters),
    )
end
