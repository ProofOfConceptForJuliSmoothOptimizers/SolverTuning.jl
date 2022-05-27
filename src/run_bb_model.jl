function run_bb_model(nlp::B, x::S) where {T, S, B <: AbstractBBModel{T, S}}
  return get_bb_output(nlp, x)
end

function get_bb_output(nlp::B, x::S) where {T, S, B <: AbstractBBModel{T, S}}
  futures = Dict{Int, Future}()
  @sync for worker_id in workers()
    @async futures[worker_id] = @spawnat worker_id let bb_output = 0.0, metrics=ProblemMetrics[]
      global worker_problems
      for pb in worker_problems
        f, p_metric = obj!(nlp, x, pb)
        bb_output += f
        push!(metrics, p_metric)
      end
      return bb_output, metrics
    end
  end
  bb_output_per_worker = Dict{Int, Tuple{Float64, Vector{ProblemMetrics}}}()
  @sync for worker_id in workers()
    @async bb_output_per_worker[worker_id] = fetch(futures[worker_id])
  end
  new_worker_data = Dict{Int, Vector{ProblemMetrics}}(
    w_id => p_metrics for (w_id, (_, p_metrics)) in bb_output_per_worker
  )
  return sum(f for (f, _) in values(bb_output_per_worker)), new_worker_data
end
