import Base.in
abstract type AbstractDomain{T} end
eltype(::AbstractDomain{T}) where {T} = T

"""
Real Domain for continuous variables
"""
abstract type RealDomain{T<:Real} <: AbstractDomain{T} end
mutable struct RealInterval{T<:Real} <: RealDomain{T}
    lower::T
    upper::T
    lower_open::Bool
    upper_open::Bool
    function RealInterval(
        lower::T,
        upper::T,
        lower_open::Bool = false,
        upper_open::Bool = false,
    ) where {T<:Real}
        lower ≤ upper ||
            error("lower bound ($lower) must be less than upper bound ($upper)")
        new{T}(lower, upper, lower_open, upper_open)
    end
end
lower(D::RealInterval) = D.lower
upper(D::RealInterval) = D.upper

∈(x::T, D::RealInterval{T}) where {T<:Real} =
    (lower_open ? lower(D) < x : lower(D) ≤ x) && (upper_open ? x < upper(D) : x ≤ upper(D))
in(x::T, D::RealInterval{T}) where {T<:Real} = x ∈ D


"""
Integer Domain for discrete variables.
    1. Integer range
    2. Integer Set
"""
abstract type IntegerDomain{T<:Integer} <: AbstractDomain{T} end
mutable struct IntegerRange{T<:Integer} <: IntegerDomain{T}
    lower::T
    upper::T

    function IntegerRange(lower::T, upper::T) where {T<:Integer}
        lower ≤ upper ||
            error("lower bound ($lower) must be less than upper bound ($upper)")
        new{T}(lower, upper)
    end
end
lower(D::IntegerRange) = D.lower
upper(D::IntegerRange) = D.upper

∈(x::T, D::IntegerRange{T}) where {T<:Integer} = lower(D) ≤ x ≤ upper(D)
in(x::T, D::IntegerRange{T}) where {T<:Integer} = x ∈ D


mutable struct IntegerSet{T<:Integer} <: IntegerDomain{T}
    set::Set{T}

    function IntegerSet(values::Vector{T}) where {T<:Integer}
        new{T}(Set{T}(values))
    end
end
∈(x::T, D::IntegerSet{T}) where {T<:Integer} = x ∈ D.s
in(x::T, D::IntegerSet{T}) where {T<:Integer} = x ∈ D

mutable struct BinarySet{T<:Integer} <: IntegerDomain{T}
    set::Set{T}
    function BinarySet(set::Set{T}) where {T<:Integer}
        new{T}(set)
    end
    BinarySet() = BinarySet(Set(0:1))
end


"""
Categorical Domain for categorical variables.
"""
abstract type CategoricalDomain{T<:AbstractString} <: AbstractDomain{T} end
mutable struct CategoricalSet{T<:AbstractString} <: CategoricalDomain{T}
    set::Set{T}
    function CategoricalSet(categories::Vector{T}) where {T<:AbstractString}
        new{T}(Set{T}(categories))
    end
    CategoricalSet() = CategoricalSet(Vector{AbstractString}())
end
