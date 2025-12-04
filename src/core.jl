"""
    evolve!(curr::Field, prev::Field, a, dt)

Calculate a new temperature field curr based on the previous 
field prev. a is the diffusion constant and dt is the largest 
stable time step.    
"""
function evolve!(currdata::SharedArray, prevdata::SharedArray, dt) # this function will modify the data of the curr and prev objects
    nx, ny = size(currdata) .- 2

    @sync @distributed for j = 2:ny+1 # the threading branch is 3 times faster than this branch
        for i = 2:nx+1 
            @inbounds xderiv = (prevdata[i-1, j] - 2.0 * prevdata[i, j] + prevdata[i+1, j]) / DX^2
            @inbounds yderiv = (prevdata[i, j-1] - 2.0 * prevdata[i, j] + prevdata[i, j+1]) / DY^2
            @inbounds currdata[i, j] = prevdata[i, j] + A * dt * (xderiv + yderiv)
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

    # Largest stable time step
    dt = DX^2 * DY^2 / (2.0 * A * (DX^2 + DY^2))
    
    # display a nice progress bar
    # p = Progress(nsteps)

    for _ = 1:nsteps
        
        # calculate new state based on previous state
        evolve!(curr.data, prev.data, dt)

        # swap current and previous fields
        swap_fields!(curr, prev)

        # increment the progress bar
        # next!(p)
    end 

    # print final average temperature
    println("Final average temperature: $(average_temperature(curr))")
end