// ============================== //
struct vertexStruct {
    @location(0) position: vec2f,
    @location(1) color: vec4f,
    @location(2) offset: vec2f,
    @location(3) scale: vec2f,
    @location(4) perVertexColor: vec3f
};

// ============================== //
struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(perspective) color: vec4f, // Inter stage variable example
    @location(1) @interpolate(perspective) scale: vec2f,
    @location(2) @interpolate(perspective) offset: vec2f,
};

// ============================== //
@vertex
fn vs(
    vert: vertexStruct,
) -> OurVertexShaderOutput
{
    var Output: OurVertexShaderOutput;
    Output.position = vec4f(vert.position * vert.scale + vert.offset, 0.0, 1.0);
    Output.color = vert.color * vec4f(vert.perVertexColor.rgb, 1.0);
    Output.scale = vert.scale;
    Output.offset = vert.offset;
    return Output;
}