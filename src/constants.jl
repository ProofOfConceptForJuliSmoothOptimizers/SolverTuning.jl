const CONSTRUCTOR_REGEX = r"
(?(DEFINE)(?<parenthese>\(([^()] | (?P>parenthese) )*\))) # Define recursive parenthese regex
function\s(?<funcName>[a-z0-9_]+?) # Get Function Name
    \s?(?<params>(?P>parenthese))  # Get Parameters
    "mxsi

const PARAM_REGEX = r"(?<varName>[a-z0-9_]+?)::(?<varType>Int[0-9]{0,2}|Bool|String|Float[0-9]{0,2})\s?=\s?(?<varDefault>.+?)[,)]"mxsi

const INPUT_TYPES = ["R", "I", "B"]

const DATA_TYPES = Dict{String, String}(
    "Int32"     => "I",
    "Int64"     => "I",
    "Int"       => "I",
    "UInt32"    => "I",
    "UInt64"    => "I",
    "UInt"      => "I",
    "Float32"   => "R",
    "Float64"   => "R",
    "Bool"      => "B"
)