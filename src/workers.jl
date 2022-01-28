
export init_workers

const MAX_NUMBER_SGE_NODES = 24
const QSUB_FLAGS = `-q hs22 -V`
const EXE_FLAGS = "--project=."
const WORKING_DIRECTORY = joinpath(ENV["HOME"], "julia_worker_logs")

function setup_workers(;nb_nodes=MAX_NUMBER_SGE_NODES, qsub_flags=QSUB_FLAGS, exe_flags=EXE_FLAGS, wd=WORKING_DIRECTORY)
  @assert nb_nodes ≤ MAX_NUMBER_SGE_NODES "Number of nodes requested exceeds $MAX_NUMBER_SGE_NODES"
  addprocs_sge(
    nb_nodes;
    qsub_flags=qsub_flags,
    exeflags=exe_flags,
    wd = wd,
  )
end

function dispatch_modules()
  module_names = (Symbol(m) for m ∈ keys(Pkg.installed()))
  for m ∈ module_names
    @everywhere @eval using $m
  end
end

function init_workers(nb_retries=5;kwargs...)
  let task = Task(() -> setup_workers(;kwargs...))
    @info "Starting workers on SGE: 1"
    for i in 1:nb_retries
      try 
        schedule(task)
        wait(task)
        break
      catch exception
        @warn "Worker initialization failed on try #$i. $(nb_retries-i) tries left."
      end
    end
    istaskfailed(task) && throw("Couldn't spawn workers on cluster.")
  end

  try
    # using essential modules in all workers:
    @everywhere begin
      using Pkg, Distributed
      using NLPModels
    end 
    
    #TODO: dispatch modules found in current environment:

    # Define function to instantiate problems on workers: 
    @everywhere workers() worker_problems = Vector{AbstractNLPModel}()
    
    @everywhere function push_worker_problems(problems:: Vector{P}) where {P <: AbstractNLPModel}
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
  finally
    @info "Killing workers..."
    rmprocs(workers())
  end
end
