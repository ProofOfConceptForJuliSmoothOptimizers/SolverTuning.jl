try
  nb_sge_nodes = 20
  # setup julia workers on SGE:
  addprocs_sge(
    nb_sge_nodes;
    qsub_flags = `-q hs22 -V`,
    exeflags = "--project=.",
    wd = joinpath(ENV["HOME"], "julia_worker_logs"),
  )

  println("Standard package definition:")
  @everywhere begin
    using Pkg, Distributed
    using LinearAlgebra, Logging, Printf, DataFrames
  end

  # Define JSO packages
  println("JSO package definition:")
  @everywhere begin
    using Krylov,
      LinearOperators,
      NLPModels,
      NLPModelsJuMP,
      OptimizationProblems,
      OptimizationProblems.PureJuMP,
      NLPModelsModifiers,
      SolverCore,
      SolverTools,
      ADNLPModels,
      SolverTest,
      SolverBenchmark,
      BenchmarkTools

    using BenchmarkTools:Trial
    import BenchmarkTools.hasevals
    import BenchmarkTools.prunekwargs
    import BenchmarkTools.Benchmark
    import BenchmarkTools.Parameters
    import BenchmarkTools.run_result
  end

  # 1. Define function to instantiate problems on  workers: 
  @everywhere workers() worker_problems = Vector{Symbol}()

  @everywhere function add_worker_problem(problem_name::Symbol)
    global worker_problems
    push!(worker_problems, problem_name)
  end

  # Define Nomad:
  println("Nomad package definition:")
  @everywhere begin
    using NOMAD
    using NOMAD: NomadOptions
  end

  @everywhere begin
    include("domains.jl")
    include("parameters.jl")
    include("lbfgs.jl")
    include("benchmark_macros.jl")
    include("black_box.jl")
    include("nomad_interface.jl")
  end
catch e
  println("error occured with nodes:")
  if isa(e, CompositeException)
    # println(e.exceptions)
    println("This is a composite exception:")
    showerror(stdout, first(e.exceptions))
  else
    showerror(stdout, e)
  end
  rmprocs(workers())
  exit()
end
