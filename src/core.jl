"""
    evolve!(curr::Field, prev::Field, a, dt)

Calculate a new temperature field curr based on the previous 
field prev. a is the diffusion constant and dt is the largest 
stable time step.    
"""
function evolve!(currdata, prevdata, dx2, dy2, nx, ny, a, dt) # this function will modify the data of the curr and prev objects

    pos = thread_position_in_grid()
    i, j = pos.x, pos.y

    if i > 1 && j > 1 && i < nx+2 && j < ny+2
        @inbounds xderiv = (prevdata[i-1, j] - 2.0 * prevdata[i, j] + prevdata[i+1, j]) / dx2
        @inbounds yderiv = (prevdata[i, j-1] - 2.0 * prevdata[i, j] + prevdata[i, j+1]) / dy2
        @inbounds currdata[i, j] = prevdata[i, j] + a * dt * (xderiv + yderiv)
    end

    return nothing

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
    dx2, dy2 = DX^2, DY^2
    dt = dx2 * dy2 / (2.0 * A * (dx2 + dy2))
    nx, ny = curr.nx, curr.ny
    a = A
    
    # display a nice progress bar
    # p = Progress(nsteps)

    for _ = 1:nsteps
        
        threads = 16, 16
        groups = cld(nx, threads[1]), cld(ny, threads[2])
        # calculate new state based on previous state
        @sync @metal threads=threads groups=groups evolve!(curr.data, prev.data, dx2, dy2, nx, ny, a, dt)

        # swap current and previous fields
        swap_fields!(curr, prev)

        # increment the progress bar
        # next!(p)
    end 

    # print final average temperature
    println("Final average temperature: $(average_temperature(curr))")
end