abstract type AbstractDomain{T} end
eltype(::AbstractDomain{T}) where T = T

abstract type RealDomain{T <: Real} <: AbstractDomain{T} end
mutable struct RealInterval{T <: Real} <: RealDomain{T}
  lower::T
  upper::T
  lower_open::Bool
  upper_open::Bool
  function RealInterval(lower::T, upper::T, lower_open::Bool = false, upper_open::Bool = false) where {T <: Real}
    lower ≤ upper || error("lower bound ($lower) must be less than upper bound ($upper)")
    new{T}(lower, upper, lower_open, upper_open)
  end
end
lower(D::RealInterval) = D.lower
upper(D::RealInterval) = D.upper

∈(x::T, D::RealInterval{T}) where {T <: Real} = (lower_open ? lower(D) < x : lower(D) ≤ x) && (upper_open ? x < upper(D) : x ≤ upper(D))
in(x::T, D::RealInterval{T}) where {T <: Real} = x ∈ D

abstract type IntegerDomain{T <: Integer} <: AbstractDomain{T} end
mutable struct IntegerRange{T <: Integer} <: IntegerDomain{T}
  lower::T
  upper::T
  function IntegerRange(lower::T, upper::T) where {T <: Integer}
    lower ≤ upper || error("lower bound ($lower) must be less than upper bound ($upper)")
    new{T}(lower, upper)
  end
end
lower(D::IntegerRange) = D.lower
upper(D::IntegerRange) = D.upper

∈(x::T, D::IntegerRange{T}) where {T <: Integer} = lower(D) ≤ x ≤ upper(D)
in(x::T, D::IntegerRange{T}) where {T <: Integer} = x ∈ D

mutable struct IntegerSet{T <: Integer} <: IntegerDomain{T}
  s::Set{T}
end
∈(x::T, D::IntegerSet{T}) where {T <: Integer} = x ∈ D.s
in(x::T, D::IntegerSet{T}) where {T <: Integer} = x ∈ D

abstract type AbstractParameter{T} end
mutable struct AlgorithmicParameter{T, D <: AbstractDomain{T}} <: AbstractParameter{T}
  default::T
  domain::D
  name::String
  function AlgorithmicParameter(default::T, domain::AbstractDomain{T}, name::String) where T
    default ∈ domain || error("default value should be in domain")
    new{T,typeof(domain)}(default, domain, name)
  end
end
