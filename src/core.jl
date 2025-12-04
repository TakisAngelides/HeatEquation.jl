using ProgressMeter

"""
    evolve!(curr::Field, prev::Field, a, dt)

Calculate a new temperature field curr based on the previous 
field prev. a is the diffusion constant and dt is the largest 
stable time step.    
"""
function evolve!(curr::Field, prev::Field, a, dt)
    # The threads macro parallelizes the outer loop over j, julia is column-major so accessing rows at fixed column is more cache friendly
    # Modern CPUs have cache memory (L1, L2, L3) that is:
    # much faster than RAM (like 100× faster)
    # loaded in cache lines (chunks of 64 bytes)
    # When your program accesses an array element, the CPU loads the entire cache line containing that element.
    # If you then access nearby memory, it is already in cache → super fast. 
    # If you jump to far-away memory, the CPU must fetch a new cache line from RAM → slow.
    Threads.@threads for j = 2:curr.ny+1 # Outer loop j assigns different columns to different threads.
        for i = 2:curr.nx+1 # inner loop which is changing the fastest accesses memory contiguously
            @inbounds xderiv = (prev.data[i-1, j] - 2.0 * prev.data[i, j] + prev.data[i+1, j]) / curr.dx^2
            @inbounds yderiv = (prev.data[i, j-1] - 2.0 * prev.data[i, j] + prev.data[i, j+1]) / curr.dy^2
            @inbounds curr.data[i, j] = prev.data[i, j] + a * dt * (xderiv + yderiv)
        end 
    end
end

"""
    swap_fields!(curr::Field, prev::Field)

Swap the data of two fields curr and prev.    
"""    
function swap_fields!(curr::Field, prev::Field)
    tmp = curr.data
    curr.data = prev.data
    prev.data = tmp
end

""" 
    average_temperature(f::Field)

Calculate average temperature of a temperature field.        
"""
average_temperature(f::Field) = sum(f.data[2:f.nx+1, 2:f.ny+1]) / (f.nx * f.ny)

"""
    simulate!(current, previous, nsteps)

Run the heat equation solver on fields curr and prev for nsteps.
"""
function simulate!(curr::Field, prev::Field, nsteps)

    # println("Initial average temperature: $(average_temperature(curr))")

    # Diffusion constant
    a = 0.5
    # Largest stable time step
    dt = curr.dx^2 * curr.dy^2 / (2.0 * a * (curr.dx^2 + curr.dy^2))
    
    # display a nice progress bar
    # p = Progress(nsteps)

    for _ = 1:nsteps
        
        # calculate new state based on previous state
        evolve!(curr, prev, a, dt)

        # swap current and previous fields
        swap_fields!(curr, prev)

        # increment the progress bar
        # next!(p)
    end 

    # print final average temperature
    println("Final average temperature: $(average_temperature(curr))")
end