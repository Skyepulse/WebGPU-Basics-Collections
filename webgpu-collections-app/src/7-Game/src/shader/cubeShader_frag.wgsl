// ============================== //
struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(perspective) color: vec4f
};

// ============================== //
@fragment
fn fs(input: OurVertexShaderOutput) -> @location(0) vec4f
{
    return input.color;
}