export init_workers

const MAX_NUMBER_SGE_NODES = 24
const QSUB_FLAGS = `-q hs22 -V`
const EXE_FLAGS = "--project=."
const WORKING_DIRECTORY = joinpath(ENV["HOME"], "julia_worker_logs")

function setup_workers(nb_tries::Int;
  nb_nodes = MAX_NUMBER_SGE_NODES,
  qsubflags = QSUB_FLAGS,
  exec_flags = EXE_FLAGS,
  working_dir = WORKING_DIRECTORY,
)
  @assert nb_tries ≥ 1 "Number of tries must be ≥ 1"
  @assert nb_nodes ≤ MAX_NUMBER_SGE_NODES "Number of nodes requested exceeds $MAX_NUMBER_SGE_NODES"
  let task = Task(() -> addprocs_sge(nb_nodes; qsub_flags = qsubflags, exeflags = exec_flags, wd = working_dir))
    @info "Starting workers on SGE: 1"
    for i = 1:nb_tries
      try
        schedule(task)
        wait(task)
        break
      catch exception
        @warn "Worker initialization failed on try #$i. $(nb_tries-i) tries left."
      end
    end
    istaskfailed(task) && throw("Couldn't spawn workers on cluster.")
  end
end

function dispatch_modules()
  module_names = (Symbol(m) for m ∈ keys(Pkg.installed()))
  for m ∈ module_names
    @everywhere @eval using $m
  end
end

function init_workers(nb_retries = 5; kwargs...)
  try
    setup_workers(nb_retries;kwargs...)
    # using essential modules in all workers:
    using_modules_quote = quote
      using Pkg, Distributed
      using NLPModels
    end
    
    @everywhere eval(using_modules_quote)

    #TODO: dispatch modules found in current environment:

    # Define function to instantiate problems on workers: 
    dispatch_worker_quote = quote
      worker_problems = Vector{AbstractNLPModel}()
    end
    @everywhere workers() eval(dispatch_worker_quote)

    @everywhere function push_worker_problems(problems::Vector{P}) where {P <: AbstractNLPModel}
      global worker_problems
      push!(worker_problems, problems...)
    end

    @everywhere function clear_worker_problems()
      global worker_problems
      worker_problems = Vector{AbstractNLPModel}()
      return worker_problems
    end
  catch worker_exception
    if isa(worker_exception, CompositeException)
      @warn "This is a CompositeException"
      broadcast(e -> showerror(stdout, e), worker_exception.exceptions)
    else
      showerror(stdout, e)
    end
    @info "Killing workers..."
    rmprocs(workers())
  end
end