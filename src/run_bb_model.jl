function run_bb_model(nlp::B, x::S) where {T, S, B <: AbstractBBModel{T, S}}
  bb_output = get_bb_output(nlp, x)
  return bb_output
end

function get_bb_output(nlp::B, x::S) where {T, S, B <: AbstractBBModel{T, S}}
  futures = Dict{Int, Future}()
  @sync for worker_id in workers()
    @async futures[worker_id] = @spawnat worker_id let bb_output = 0.0
      global worker_problems
      for pb in worker_problems
        bb_output += obj!(nlp, x, pb)
      end
      return bb_output
    end
  end
  bb_output_per_worker = Dict{Int, Float64}()
  @sync for worker_id in workers()
    @async bb_output_per_worker[worker_id] = fetch(futures[worker_id])
  end
  return sum(values(bb_output_per_worker))
end
