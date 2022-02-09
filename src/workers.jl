export init_workers

const MAX_NUMBER_SGE_NODES = 24
const QSUB_FLAGS = `-q hs22 -V`
const EXE_FLAGS = "--project=$(joinpath(@__DIR__, ".."))"
const WORKING_DIRECTORY = joinpath(ENV["HOME"], "julia_worker_logs")

function init_workers(nb_retries = 5; kwargs...)
  try
    setup_workers(nb_retries; kwargs...)
  catch worker_exception
    if isa(worker_exception, CompositeException)
      @error "This is a CompositeException"
      broadcast(e -> showerror(stdout, e), worker_exception.exceptions)
    else
      showerror(stdout, worker_exception)
    end
    @info "Killing workers..."
    rmprocs(workers())
    exit()
  end
end

function setup_workers(
  nb_tries::Int;
  nb_nodes = MAX_NUMBER_SGE_NODES,
  qsubflags = QSUB_FLAGS,
  exec_flags = EXE_FLAGS,
  working_dir = WORKING_DIRECTORY,
)
  @assert nb_tries ≥ 1 "Number of tries must be ≥ 1"
  @assert nb_nodes ≤ MAX_NUMBER_SGE_NODES "Number of nodes requested exceeds $MAX_NUMBER_SGE_NODES"
  for i = 1:nb_tries
    @info "Starting workers on SGE: try #$i"
    let task = Task(
        () ->
          addprocs_sge(nb_nodes; qsub_flags = qsubflags, exeflags = exec_flags, wd = working_dir),
      )
      try
        schedule(task)
        wait(task)
        break
      catch exception
        @warn "Worker initialization failed on try #$i. $(nb_tries-i) tries left."
        sleep(30)
      end
    end
  end
end

function clear_worker_problems()
  global worker_problems
  worker_problems = Vector{AbstractNLPModel}()
  return worker_problems
end

function push_worker_problems(problems::Vector{P}) where {P <: AbstractNLPModel}
  global worker_problems
  push!(worker_problems, problems...)
end
