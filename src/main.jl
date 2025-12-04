using Plots
using BenchmarkTools


# Fixed grid spacing
const DX = 0.01
const DY = 0.01
# Default temperatures
const T_DISC = 5.0
const T_AREA = 120.0
const T_UPPER = 85.0
const T_LOWER = 5.0
const T_LEFT = 20.0
const T_RIGHT = 70.0
# Default problem size
const ROWS = 128
const COLS = 128
const NSTEPS = 10000

include("heat.jl")
include("core.jl")


"""
    visualize(curr::Field, filename=:none)

Create a heatmap of a temperature field. Optionally write png file. 
"""    
function visualize(curr::Field, filename=:none)
    background_color = :white
    plot = heatmap(
        curr.data,
        colorbar_title = "Temperature (C)",
        background_color = background_color
    )

    if filename != :none
        savefig(filename)
    else
        display(plot)
    end
end


# simulate temperature evolution for nsteps
benchmark_result = @benchmark simulate!(curr, prev, $NSTEPS) setup = curr, prev = initialize($COLS, $ROWS) samples = 5 evals = 1
display(benchmark_result)

# visualize final field, requires Plots.jl
# visualize(curr, "images/final.png")