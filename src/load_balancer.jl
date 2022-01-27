abstract type AbstractLoadBalancer{T} end

mutable struct GreedyLoadBalancer{N <: AbstractNLPModel, T<:Real} <: AbstractLoadBalancer{T}
    problems::Dict{N,T}
    method::Function

    function GreedyLoadBalancer(problems::Dict{N,T}, method::Function) where {N <: AbstractNLPModel, T<:Real}
        obj = new{N,T}(problems, method)
        obj.method = nb_partitions -> (method(obj, nb_partitions))
        return obj
    end
end

function GreedyLoadBalancer(problems::Dict{N,T}) where {N <: AbstractNLPModel, T<:Real}
    return GreedyLoadBalancer(problems, greedy_problem_partition)
end

function execute(lb::L, bb_iteration::Int; iteration_threshold=1) where L <: AbstractLoadBalancer
    isempty(values(lb.problems)) && return
    # to make sure we load balance after the first iteration
    (bb_iteration != 0) && ((bb_iteration - 1)  % iteration_threshold != 0) && return
    
    @info "Load Balancing problems for iteration: $bb_iteration"
    nb_partitions = length(workers())
    partitions = lb.method(nb_partitions)

    @info "distributing problems"
    problem_def_future = Future[]
    for (i,worker_id) in enumerate(workers())
        remotecall_fetch(clear_worker_problems, worker_id)
        push!(problem_def_future, remotecall(push_worker_problems, worker_id, partitions[i]))
    end
    @sync for problem_future in problem_def_future
        @async fetch(problem_future)
    end
    @info "finished distributing"
end

function greedy_problem_partition(lb::GreedyLoadBalancer{N ,T}, nb_partitions::Int) where {N <: AbstractNLPModel, T<:Real}
    partitions = [Vector{N}() for _ ∈ 1:nb_partitions]
    problems = sort(collect(lb.problems);by=p->p[2], rev=true)
    σ = sum(pb_value for (pb,pb_value) ∈ problems)/nb_partitions
    penalties = [-σ for _ ∈ 1:nb_partitions]
    for (pb,pb_value) ∈ problems
        p_idx = argmin(penalties)
        push!(partitions[p_idx], pb)
        penalties[p_idx] += pb_value
    end
    return partitions
end

function update_problems(lb::L, problem_data::Dict{N, T}) where {L <: AbstractLoadBalancer, N <: AbstractNLPModel, T <: Real}
    lb.problems = problem_data
end