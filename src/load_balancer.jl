export AbstractLoadBalancer, GreedyLoadBalancer, RoundRobinLoadBalancer

abstract type AbstractLoadBalancer end

mutable struct GreedyLoadBalancer <: AbstractLoadBalancer
  problems::Dict{Int, Problem}
  method::Function
  iteration::Int

  function GreedyLoadBalancer(
    problems::Dict{Int, Problem},
    method::Function,
    iteration::Int,
  )
    obj = new(problems, method, iteration)
    obj.method = nb_partitions -> (method(obj, nb_partitions))
    return obj
  end
end

function GreedyLoadBalancer(problems::Dict{Int, Problem})
  return GreedyLoadBalancer(problems, LPT, 0)
end

mutable struct RoundRobinLoadBalancer <: AbstractLoadBalancer
  problems::Dict{Int, Problem}
  method::Function
  iteration::Int

  function RoundRobinLoadBalancer(
    problems::Dict{Int, Problem},
    method::Function,
    iteration::Int,
  )
    obj = new(problems, method, iteration)
    obj.method = nb_partitions -> (method(obj, nb_partitions))
    return obj
  end
end

function RoundRobinLoadBalancer(problems::Dict{Int, Problem})
  return RoundRobinLoadBalancer(problems, round_robin_partition, 0)
end

mutable struct CombineLoadBalancer <: AbstractLoadBalancer
  problems::Dict{Int, Problem}
  method::Function
  iteration::Int

  function CombineLoadBalancer(
    problems::Dict{Int, Problem},
    method::Function,
    iteration::Int,
  )
    obj = new(problems, method, iteration)
    obj.method = nb_partitions -> (method(obj, nb_partitions))
    return obj
  end
end

function CombineLoadBalancer(problems::Dict{Int, Problem})
  return CombineLoadBalancer(problems, COMBINE, 0)
end

function execute(lb::L; iteration_threshold = 1) where {L <: AbstractLoadBalancer}
  isempty(values(lb.problems)) && return
  bb_iteration = lb.iteration
  # to make sure we load balance after the first iteration
  (bb_iteration != 0) && ((bb_iteration - 1) % iteration_threshold != 0) && return

  nb_partitions = length(workers())
  partitions, status = lb.method(nb_partitions)
  problem_def_future = Future[]
  for (i, worker_id) in enumerate(workers())
    remotecall_fetch(clear_worker_problems, worker_id)
    push!(problem_def_future, remotecall(push_worker_problems, worker_id, partitions[i]))
  end
  @sync for problem_future in problem_def_future
    @async fetch(problem_future)
  end
  lb.iteration += 1
end

function LPT(lb::L, nb_partitions::Int) where {L <: Union{GreedyLoadBalancer,CombineLoadBalancer}}
  partitions = [Vector{Problem}() for _ ∈ 1:nb_partitions]
  problems = sort(collect(values(lb.problems)); by = p -> p.weight, rev = true)
  σ = sum(problem.weight for problem ∈ problems) / nb_partitions
  penalties = [-σ for _ ∈ 1:nb_partitions]
  for pb ∈ problems
    p_idx = argmin(penalties)
    push!(partitions[p_idx], pb)
    penalties[p_idx] += pb.weight
  end
  return partitions, true
end

function round_robin_partition(lb::RoundRobinLoadBalancer, nb_partitions::Int)
  problems = collect(Problem, values(lb.problems))
  problems = shuffle(problems)
  partitions = [problems[i:nb_partitions:length(problems)] for i ∈ 1:nb_partitions]
  return partitions, true
end

function first_fit(v::Vector{Problem}, max_capacity::S, nb_bin::Int) where S <: Real
  nb_bin > 0 || error("cannot have a negative number of partitions")
  partitions = [Vector{Problem{T}}() for _ in 1:nb_bin]
  capacities = [zero(S) for _ in 1:nb_bin]
  for p in v
    for (c_idx, c) in enumerate(capacities)
      if c + p.weight ≤ max_capacity
        push!(partitions[c_idx], p)
        capacities[c_idx] += p.weight
        break
      end
    end
  end
  status = sum(length(i) for i in partitions) == length(v)
  return partitions, status
end

function FFD(v::Vector{Problem}, max_capacity::S, nb_bin::Int) where {S <: Real}
sorted_v = sort(v; by = p -> p.weight, rev = true)
return first_fit(sorted_v, max_capacity, nb_bin)
end

function multifit(problems::Vector{Problem}, nb_bin::Int; L::S= max(sum(p.weight for p in problems)/nb_bin, max([p.weight for p in problems]...)), U::V= max(2*sum(problems)/nb_bin, max([p.weight for p in problems]...)), nm_itmax::Int=6, atol::Float64=0.005) where {S <: Real, V <: Real}
  nb_bin > 0 || error("nb of partitions must be greater than 0.")
  partitions = nothing
  σ = sum(problem.weight for problem ∈ problems) / nb_bin
  while (atol*σ < U-L) && nm_itmax > 0
    C = (L+U)/2
    partitions, status = FFD(problems, C, nb_bin)
    U = status ? C : U
    L = !status ? C : L
    nm_itmax -= 1
  end
  partitions, status = FFD(problems, U, nb_bin)
  return partitions, status
end

function COMBINE(lb::CombineLoadBalancer, nb_partitions::Int; atol::Float64=0.0005)
  nb_partitions > 0 || error("nb of partitions must be greater than 0.")
  problems = collect(values(lb.problems))
  partitions, status = LPT(lb, nb_partitions)
  σ = sum(problem.weight for problem ∈ problems) / nb_partitions
  M = max([sum(p.weight for p in p_i) for p_i in partitions]...)
  M ≥ 1.5σ && (return partitions, status)
  return multifit(problems, nb_partitions; L=max(M/(4/3 - 1/(3M)), max([p.weight for p in problems]...), σ), U=M, atol=atol)
end
