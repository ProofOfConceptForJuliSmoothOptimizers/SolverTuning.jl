import Base.in
import Base.push!
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
(D.lower_open ? lower(D) < x : lower(D) ≤ x) &&
(D.upper_open ? x < upper(D) : x ≤ upper(D))
in(x::T, D::RealInterval{T}) where {T<:Real} = x ∈ D

lower_bound(D::RealInterval{T}) where {T<:Real} = lower(D)
upper_bound(D::RealInterval{T}) where {T<:Real} = upper(D)

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

    lower_bound(D::IntegerRange{T}) where {T<:Integer} = lower(D)
    upper_bound(D::IntegerRange{T}) where {T<:Integer} = upper(D)
    # Binary set is just an integer range with 0 and 1.
    BinaryRange(T::DataType = Int) = IntegerRange(zero(T), one(T))
    
mutable struct IntegerSet{T<:Integer} <: IntegerDomain{T}
    set::Set{T}

    function IntegerSet(values::Vector{T}) where {T<:Integer}
        new{T}(Set{T}(values))
    end
end
∈(x::T, D::IntegerSet{T}) where {T<:Integer} = in(x, D.set)
in(x::T, D::IntegerSet{T}) where {T<:Integer} = x ∈ D
Base.push!(D::IntegerSet{T}, x::T) where {T<:Integer} = push!(D.set, x)

lower_bound(D::IntegerSet{T}) where {T<:Integer} = min(D.set...)
upper_bound(D::IntegerSet{T}) where {T<:Integer} = max(D.set...)
"""
Categorical Domain for categorical variables.
"""
# Change this to a vector. use indices...
abstract type CategoricalDomain{T<:AbstractString} <: AbstractDomain{T} end
mutable struct CategoricalSet{T<:AbstractString} <: CategoricalDomain{T}
    categories::Vector{T}
end

function CategoricalSet(categories::Vector{T}) where {T<:AbstractString}
    return CategoricalSet(Set{T}(categories))
end
CategoricalSet() = CategoricalSet(Vector{AbstractString}())
∈(x::T, D::CategoricalSet{T}) where {T<:AbstractString} = x in D.categories
in(x::T, D::CategoricalSet{T}) where {T<:AbstractString} = x ∈ D
push!(D::CategoricalSet{T}, x::T) where {T<:AbstractString} = push!(D.categories, x)
lower_bound(D::CategoricalSet{T}) where {T<:AbstractString} = min(D.categories...)
upper_bound(D::CategoricalSet{T}) where {T<:AbstractString} = max(D.categories...)
