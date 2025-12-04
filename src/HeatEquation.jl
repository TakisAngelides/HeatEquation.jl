module HeatEquation

using ProgressMeter
using Plots
using BenchmarkTools

include("main.jl")

export Field, simulate!, initialize, visualize, average_temperature

end
