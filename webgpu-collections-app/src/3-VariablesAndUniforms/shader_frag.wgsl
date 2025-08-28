// ============================== //
struct ourStruct {
    color: vec4f,
    scale: vec2f,
    offset: vec2f
};

// ============================== //
struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(perspective) color: vec4f,
    @location(1) @interpolate(perspective) scale: vec2f,
    @location(2) @interpolate(perspective) offset: vec2f,
};

// ============================== //
@fragment
fn fs(input: OurVertexShaderOutput) -> @location(0) vec4f
{
    let white = vec4f(1.0, 1.0, 1.0, 1.0);
    let black = vec4f(0.0, 0.0, 0.0, 1.0);

    let grid = vec2f(input.position.xy) / 15.0; // Example of using the builtin position
    let checker = (i32(floor(grid.x)) + i32(floor(grid.y))) % 2 == 1;
    // Diagonal stripes:
    // let diagonal = i32(floor(grid.x + grid.y)) % 2 == 1;

    return select(white, black, checker) * input.color;
}