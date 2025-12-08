module HeatEquation # This defines the module named HeatEquation

using ProgressMeter
using Plots
using BenchmarkTools
using Profile
using PProf

# This runs the simulation when the package is included, probably not the best practice because with precompile it runs the main file 
# but okay for this simple example
include("main.jl")

# The following are available to the user when using the package with "using HeatEquation"
export Field, simulate!, initialize, visualize, average_temperature

end
