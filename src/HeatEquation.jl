module HeatEquation # This defines the module named HeatEquation

using Plots
using BenchmarkTools
using Metal
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

# This runs the simulation when the package is included
include("main.jl")

end
