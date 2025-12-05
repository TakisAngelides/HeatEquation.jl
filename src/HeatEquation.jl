module HeatEquation # This defines the module named HeatEquation

# This runs the simulation when the package is included
include("main.jl")

# The following are available to the user when using the package with "using HeatEquation"
export Field, simulate!, initialize, visualize, average_temperature

end
