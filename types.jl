import Base.show

abstract type AbstractParameter end

mutable struct Parameter <: AbstractParameter
    name::String
    type::String
    default::Union{Real, Nothing}
end

""" Default constructor"""
Parameter() = Parameter("", Real, nothing)

"""Constructor with parameters"""
# function Parameter(name::String, type::String, default::Union{Real, Nothing}) 
#     return Parameter(name, type, default)
# end


"""Constructor given a dict representing the parameter"""
function Parameter(param::Dict{Symbol, String})
    name = haskey(param, :name) ? param[:name] : ""
    type = get_type(param[:type])

    default = get_default_value(type, param)

    return Parameter(name, type, default)
end

"""Returns the correct datatype given a string.
    Uses a dict that maps a String to a DataType"""
function get_type(type::String)
    haskey(DATA_TYPES, type) && return DATA_TYPES[type]

    return nothing
end

function get_default_value(type::String, param::Dict{Symbol, String})
    (!haskey(param, :default_value) ||  !(type ∈ INPUT_TYPES)) && return nothing
    (type == "R") && return parse(Float64, param[:default_value])
    (type == "I") && return parse(Int, param[:default_value])
    (type == "B") && return parse(Bool, param[:default_value])
end

function show(param::AbstractParameter)
    println("Parameter: name => $(param.name), type: => $(param.type), default => $(param.default)")
end