const DX      = 0.01f0
const DY      = 0.01f0
const T_DISC  = 5.0f0
const T_AREA  = 65.0f0
const T_UPPER = 85.0f0
const T_LOWER = 5.0f0
const T_LEFT  = 20.0f0
const T_RIGHT = 70.0f0
const A       = 0.5f0

const ROWS = 2048
const COLS = 2048
const NSTEPS = 500

const DT = DX^2 * DY^2 / (2f0 * A * (DX^2 + DY^2))
const _DX2 = 1f0 / DX^2
const _DY2 = 1f0 / DY^2

function evolve!(currdata, prevdata, _dx2, _dy2, nx, ny, a, dt) # this function will modify the data of the curr and prev objects

    Threads.@threads for j = 2:ny+1
        for i = 2:nx+1
            @inbounds xderiv = (prevdata[i-1, j] - 2.0 * prevdata[i, j] + prevdata[i+1, j]) * _dx2
            @inbounds yderiv = (prevdata[i, j-1] - 2.0 * prevdata[i, j] + prevdata[i, j+1]) * _dy2
            @inbounds currdata[i, j] = prevdata[i, j] + a * dt * (xderiv + yderiv)
        end 
    end
    
    return nothing

end

function evolve_gpu!(currdata, prevdata, _dx2, _dy2, nx, ny, a, dt) # this function will modify the data of the curr and prev objects

    (i, j) = thread_position_in_grid_2d()

    if i > 1 && j > 1 && i < nx+2 && j < ny+2
        @inbounds xderiv = (prevdata[i-1, j] - 2.0f0 * prevdata[i, j] + prevdata[i+1, j]) * _dx2
        @inbounds yderiv = (prevdata[i, j-1] - 2.0f0 * prevdata[i, j] + prevdata[i, j+1]) * _dy2
        @inbounds currdata[i, j] = prevdata[i, j] + a * dt * (xderiv + yderiv)
    end

    return nothing

end

function simulate!(currdata, prevdata, _DX2, _DY2, COLS, ROWS, A, DT)

    # Time-stepping loop on CPU (for comparison)
    for t in 1:NSTEPS
        evolve!(currdata, prevdata, _DX2, _DY2, COLS, ROWS, A, DT)
        currdata, prevdata = prevdata, currdata
    end

end

function simulate_gpu!(currdata_gpu, prevdata_gpu, _DX2, _DY2, COLS, ROWS, A, DT)

    # Time-stepping loop on GPU
    for t in 1:NSTEPS
        threads = 32, 32 # this must not exceed 1024 threads per block
        groups = cld(ROWS, threads[1]), cld(COLS, threads[2])
        @sync @metal threads=threads groups=groups evolve_gpu!(currdata_gpu, prevdata_gpu, _DX2, _DY2, COLS, ROWS, A, DT)
        currdata_gpu, prevdata_gpu = prevdata_gpu, currdata_gpu
    end

end

function run()

    # Initialize state on cpu with boundary conditions
    nx, ny = COLS, ROWS
    prevdata = zeros(Float32, nx+2, ny+2)
    radius2 = (nx / 6.0)^2
    for j = 1:ny+2
        for i = 1:nx+2
            ds2 = (i - nx / 2)^2 + (j - ny / 2)^2
            if ds2 < radius2 
                prevdata[i,j] = T_DISC
            else
                prevdata[i,j] = T_AREA
            end
        end 
    end 
    prevdata[:,1] .= T_LEFT
    prevdata[:,ny+2] .= T_RIGHT
    prevdata[1,:] .= T_UPPER
    prevdata[nx+2,:] .= T_LOWER
    currdata = deepcopy(prevdata)

    # Transfer to GPU
    prevdata_gpu = MtlArray(prevdata)
    currdata_gpu = MtlArray(deepcopy(prevdata))

    # Benchmark CPU vs GPU
    # r_cpu = @benchmark simulate!($currdata, $prevdata, $_DX2, $_DY2, $COLS, $ROWS, $A, $DT) samples = 5 evals = 1
    r_gpu = @benchmark simulate_gpu!($currdata_gpu, $prevdata_gpu, Float32($_DX2), Float32($_DY2), Int32($COLS), Int32($ROWS), Float32($A), Float32($DT)) samples = 5 evals = 1

    # Display results
    # println("CPU Benchmark Results:")
    # display(r_cpu)
    println("GPU Benchmark Results:")
    display(r_gpu)

    # Visualize final temperature distributions
    # display(heatmap(currdata', title="Final Temperature Distribution (CPU)", xlabel="X", ylabel="Y", colorbar_title="Temperature"))
    display(heatmap(Array(prevdata_gpu)', title="Final Temperature Distribution (GPU)", xlabel="X", ylabel="Y", colorbar_title="Temperature"))

    # Print average temperatures
    # println("Average Temperature (CPU): ", sum(currdata[2:nx+1, 2:ny+1]) / (nx * ny))
    println("Average Temperature (GPU): ", sum(Array(prevdata_gpu)[2:nx+1, 2:ny+1]) / (nx * ny))

end

run()
