abstract type AbstractConstraint end

mutable struct LinearConstraint <: AbstractConstraint
    id::UInt
    lhs::Union{String, Real}
    operation::String
    rhs::Union{String, Real}
end

function LinearConstraint(dict::Dict{String, Any})
    id = dict["id"]
    lhs = dict["lhs"]
    operation = dict["operation"]
    rhs = dict["rhs"]

    return LinearConstraint(id, lhs, operation, rhs)
end

function show(constraint::AbstractConstraint)
    println("""
    Constraint: id      => $(constraint.id), 
            lhs:        => $(constraint.lhs),
            operation   => $(constraint.operation),
            rhs         => $(constraint.rhs)
    """)
end