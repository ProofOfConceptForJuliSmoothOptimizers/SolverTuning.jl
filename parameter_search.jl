function get_parameters(file_name="trunk.jl")
    open(joinpath(@__DIR__, "solvers", file_name), "r") do file
        matches = eachmatch(CONSTRUCTOR_REGEX, read(file, String))
        params = Vector{Dict{Symbol, String}}()
        for match in matches
        #    println(match[:funcName])
           vars = eachmatch(PARAM_REGEX, match[:params])
           vars = [Dict{Symbol, String}(:name => x[:varName], :type => x[:varType], :default_value => x[:varDefault]) for x in vars]
           append!(params, vars)
        end
        return params
    end
end
