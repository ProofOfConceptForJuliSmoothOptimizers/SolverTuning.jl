export AbstractLoadBalancer, GreedyLoadBalancer, RoundRobinLoadBalancer

abstract type AbstractLoadBalancer{T} end

mutable struct GreedyLoadBalancer{N <: AbstractNLPModel, T <: Real} <: AbstractLoadBalancer{T}
  problems::Dict{N, T}
  method::Function

  function GreedyLoadBalancer(
    problems::Dict{N, T},
    method::Function,
  ) where {N <: AbstractNLPModel, T <: Real}
    obj = new{N, T}(problems, method)
    obj.method = nb_partitions -> (method(obj, nb_partitions))
    return obj
  end
end

function GreedyLoadBalancer(problems)
  problem_dict = generate_problem_dict(problems)
  return GreedyLoadBalancer(problem_dict, greedy_problem_partition)
end

mutable struct RoundRobinLoadBalancer{N <: AbstractNLPModel, T <: Real} <: AbstractLoadBalancer{T}
  problems::Dict{N, T}
  method::Function

  function RoundRobinLoadBalancer(
    problems::Dict{N, T},
    method::Function,
  ) where {N <: AbstractNLPModel, T <: Real}
    obj = new{N, T}(problems, method)
    obj.method = nb_partitions -> (method(obj, nb_partitions))
    return obj
  end
end

function RoundRobinLoadBalancer(problems)
  problem_dict = generate_problem_dict(problems)
  return RoundRobinLoadBalancer(problem_dict, round_robin_partition)
end

generate_problem_dict(g) = Dict(nlp => 100 * rand(Float64) for nlp ∈ g)

function execute(
  lb::L,
  bb_iteration::Int;
  iteration_threshold = 1,
) where {L <: AbstractLoadBalancer}
  isempty(values(lb.problems)) && return
  # to make sure we load balance after the first iteration
  (bb_iteration != 0) && ((bb_iteration - 1) % iteration_threshold != 0) && return

  @info "Load Balancing problems for iteration: $bb_iteration"
  nb_partitions = length(workers())
  partitions = lb.method(nb_partitions)

  @info "distributing problems"
  problem_def_future = Future[]
  for (i, worker_id) in enumerate(workers())
    remotecall_fetch(clear_worker_problems, worker_id)
    push!(problem_def_future, remotecall(push_worker_problems, worker_id, partitions[i]))
  end
  @sync for problem_future in problem_def_future
    @async fetch(problem_future)
  end
  @info "finished distributing"
end

function greedy_problem_partition(
  lb::GreedyLoadBalancer{N, T},
  nb_partitions::Int,
) where {N <: AbstractNLPModel, T <: Real}
  partitions = [Vector{N}() for _ ∈ 1:nb_partitions]
  problems = sort(collect(lb.problems); by = p -> p[2], rev = true)
  σ = sum(pb_value for (pb, pb_value) ∈ problems) / nb_partitions
  penalties = [-σ for _ ∈ 1:nb_partitions]
  for (pb, pb_value) ∈ problems
    p_idx = argmin(penalties)
    push!(partitions[p_idx], pb)
    penalties[p_idx] += pb_value
  end
  return partitions
end

# TODO create new time called RoundRobinLoadBalancer
function round_robin_partition(lb::L, nb_partitions::Int) where {L <: AbstractLoadBalancer}
  problems = collect(keys(lb.problems))
  partitions = [problems[i:nb_partitions:length(problems)] for i ∈ 1:nb_partitions]
  return partitions
end

function update_problems(
  lb::L,
  problem_data::Dict{N, T},
) where {L <: AbstractLoadBalancer, N <: AbstractNLPModel, T <: Real}
  lb.problems = problem_data
end
