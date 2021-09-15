constructor_rgex = r"
(?(DEFINE)(?<parenthese>\(([^()] | (?P>parenthese) )*\))) # Define recursive parenthese regex
function\s(?<funcName>[a-z0-9_]+?) # Get Function Name
    \s?(?<params>(?P>parenthese))  # Get Parameters
    "mxsi

parameter_regex = r"(?<varName>[a-z0-9_]+?)::(?<varType>Int[0-9]{0,2}|Bool|String|Float[0-9]{0,2})\s?=\s?(?<varDefault>.+?)[,)]"mxsi
open(joinpath(@__DIR__, "trunk.jl"), "r") do file
    matches = eachmatch(constructor_rgex, read(file, String))
    for match in matches
       println(match[:funcName])
       vars = eachmatch(parameter_regex, match[:params])
       for var in vars
            println("########")
            println("\tVariable Name          => ", var[:varName]) 
            println("\tVariable Type          => ", var[:varType]) 
            println("\tVariable Default Value => ", var[:varDefault]) 
       end 
    end
end