"""
    Field(nx::Int64, ny::Int64, dx::Float32, dy::Float32, data::Matrix{Float32})

Temperature field type. nx and ny are the dimensions of the field. 
The array data contains also ghost layers, so it will have dimensions 
[nx+2, ny+2]
"""
mutable struct Field{T<:AbstractArray}
    nx::Int64
    ny::Int64
    # Size of the grid cells
    dx::Float32
    dy::Float32
    # The temperature values in the 2D grid
    data::T
end

# outer constructor with default cell sizes and initialized data
Field(nx::Int64, ny::Int64, data) = Field{typeof(data)}(nx, ny, DX, DY, data)

# extend deepcopy to new type
Base.deepcopy(f::Field) = Field(f.nx, f.ny, f.dx, f.dy, deepcopy(f.data))

"""
    initialize(rows::Int, cols::Int, arraytype = Matrix)

Initialize two temperature field with (nrows, ncols) number of 
rows and columns. If the arraytype is something else than Matrix,
create data on the CPU first to avoid scalar indexing errors.
"""
function initialize(nrows = 1000, ncols = 1000, arraytype = Matrix)

    data = arraytype(zeros(Float32, nrows+2, ncols+2))
    previous = Field(nrows, ncols, data)

    # generate a specific field with boundary conditions
    generate_field!(previous)
    current = Base.deepcopy(previous)

    return previous, current
end


"""
    generate_field!(field0::Field)

Generate a temperature field.  Pattern is disc with a radius
of nx / 6 in the center of the grid. Boundary conditions are 
(different) constant temperatures outside the grid.
"""
function generate_field!(field::Field)
    # Square of the disk radius
    radius2 = (field.nx / 6.0)^2

    for j = 1:field.ny+2
        for i = 1:field.nx+2
            ds2 = (i - field.nx / 2)^2 + (j - field.ny / 2)^2
            if ds2 < radius2 
                field.data[i,j] = T_DISC
            else
                field.data[i,j] = T_AREA
            end
        end 
    end 

    # Boundary conditions
    field.data[:,1] .= T_LEFT
    field.data[:,field.ny+2] .= T_RIGHT
    field.data[1,:] .= T_UPPER
    field.data[field.nx+2,:] .= T_LOWER
end