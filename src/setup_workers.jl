nb_sge_nodes = 20
try
    # setup julia workers on SGE:
    addprocs_sge(nb_sge_nodes; qsub_flags=`-q hs22 -V`, exeflags="--project=.", wd=joinpath(ENV["HOME"], "julia_worker_logs"))

    # Define std packages
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
        NLPModelsModifiers,
        SolverCore,
        SolverTools,
        ADNLPModels,
        SolverTest,
        CUTEst,
        SolverBenchmark,
        BenchmarkTools
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
        include("nomad_interface.jl")
    end
catch e
    println("error occured with nodes:")
    if isa(e, CompositeException)
        # println(e.exceptions)
        println("This is a composite exception:")
        broadcast(err -> showerror(stdout, err), e.exceptions)
    else
        showerror(stdout, e)
    end
    rmprocs(workers())
    exit()
end
