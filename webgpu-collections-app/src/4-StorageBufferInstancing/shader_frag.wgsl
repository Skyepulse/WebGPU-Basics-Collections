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
    return input.color;
}