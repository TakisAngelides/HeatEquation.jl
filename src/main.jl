# Fixed grid spacing
const DX = 0.01
const DY = 0.01
# Default temperatures
const T_DISC = 5.0
const T_AREA = 65.0
const T_UPPER = 85.0
const T_LOWER = 5.0
const T_LEFT = 20.0
const T_RIGHT = 70.0
# Default problem size
const ROWS = 128
const COLS = 128
const NSTEPS = 500

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

r = @allocated curr, prev = initialize(COLS, ROWS)
# println(r/10^6, " MB allocated for initializing fields of size $(COLS)x$(ROWS)")
simulate!(curr, prev, NSTEPS)

function run_profile_simulation()
    curr, prev = initialize(COLS, ROWS)
    simulate!(curr, prev, NSTEPS)
end

# @profile (for _ in 1:10 run_profile_simulation() end)
# Profile.print(format=:flat; sortedby = :count) # :flat or :tree
# Profile.clear() # clear the collected profiling data

# Profile.@profile_walltime run_profile_simulation() # this captures also tasks that are sleeping on a sync primitive
# pprof()
