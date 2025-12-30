const USE_GPU = false # Set to true to use Metal

@static if USE_GPU
    @init_parallel_stencil(Metal, Float32, 2)
    const FT = Float32 # Metal performs better with Float32 - in fact Float64 is not supported
else
    @init_parallel_stencil(Threads, Float64, 2)
    const FT = Float64
end

# Use the defined Float Type (FT) for constants to avoid type-mismatch overhead
const DX      = FT(0.01)
const DY      = FT(0.01)
const T_DISC  = FT(5.0)
const T_AREA  = FT(65.0)
const T_UPPER = FT(85.0)
const T_LOWER = FT(5.0)
const T_LEFT  = FT(20.0)
const T_RIGHT = FT(70.0)
const A       = FT(0.5)

const ROWS = 2048
const COLS = 2048
const NSTEPS = 500

const DT   = FT(DX^2 * DY^2 / (2 * A * (DX^2 + DY^2)))
const _DX2 = FT(1 / DX^2)
const _DY2 = FT(1 / DY^2)

@parallel function evolve_step!(currdata, prevdata, _dx2, _dy2, a, dt)
    # @inn(A): Select the inner elements of A. Corresponds to A[2:end-1,2:end-1,2:end-1].
    # @d2_xi(A): Compute the 2nd order differences between adjacent elements of A along the dimension x and select the inner elements of A in the remaining dimensions. Corresponds to @inn_yz(@d2_xa(A)).
    @inn(currdata) = @inn(prevdata) + a * dt * (@d2_xi(prevdata)*_dx2 + @d2_yi(prevdata)*_dy2)
    return
end

function run()
    nx, ny = COLS, ROWS
    
    # Initialization using ParallelStencil macros for xPU compatibility
    prevdata = @zeros(nx+2, ny+2)
    currdata = @zeros(nx+2, ny+2)
    
    # Initial condition setup (CPU-side initialization then transfer if needed)
    h_prev = zeros(FT, nx+2, ny+2)
    radius2 = (nx / 6.0)^2
    for j = 1:ny+2, i = 1:nx+2
        ds2 = (i - nx / 2)^2 + (j - ny / 2)^2
        h_prev[i,j] = (ds2 < radius2) ? T_DISC : T_AREA
    end 
    
    # Boundary Conditions
    h_prev[:, 1]      .= T_LEFT
    h_prev[:, ny+2]   .= T_RIGHT
    h_prev[1, :]      .= T_UPPER
    h_prev[nx+2, :]   .= T_LOWER
    
    # Copy to device (Works for both CPU and Metal)
    copyto!(prevdata, h_prev)
    copyto!(currdata, h_prev)

    println("Starting simulation on $(USE_GPU ? "Metal GPU" : "CPU Threads")...")

    # Benchmark the loop
    t_start = time()
    for t in 1:NSTEPS
        # FIXED: Added @parallel at the call site
        @parallel evolve_step!(currdata, prevdata, _DX2, _DY2, A, DT)
        
        # Pointer swap (efficient, no data copy)
        currdata, prevdata = prevdata, currdata
    end
    t_end = time()
    
    avg_t = (t_end - t_start) / NSTEPS
    println("Average time per step: $(round(avg_t * 1000, sigdigits=4)) ms")
    
    # Bring data back to CPU for visualization
    final_data = Array(currdata) # bring array back to CPU if on GPU
    
    # Print average temperature of the interior
    println("Average Temperature: ", sum(final_data[2:nx+1, 2:ny+1]) / (nx * ny))
    
    # Visualize
    display(heatmap(final_data', title="Final Temperature Distribution", color=:thermal))
end

run()