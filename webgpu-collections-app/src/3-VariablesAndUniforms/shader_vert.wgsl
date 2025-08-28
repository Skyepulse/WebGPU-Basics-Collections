// ============================== //
struct ourStruct {
    color: vec4f,
    offset: vec2f
};

struct scaleStruct {
    scale: vec2f
};

// ============================== //
struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(perspective) color: vec4f, // Inter stage variable example
    @location(1) @interpolate(perspective) scale: vec2f,
    @location(2) @interpolate(perspective) offset: vec2f,
};

// ============================== //
@group(0) @binding(0) var<uniform> ourUniform: ourStruct;
@group(0) @binding(1) var<uniform> scaleUniform: scaleStruct;

// ============================== //
@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> OurVertexShaderOutput
{
    let pos = array(
        vec2f(0.0, 0.5),
        vec2f(0.5, -0.5),
        vec2f(-0.5, -0.5)
    );

    var Output: OurVertexShaderOutput;
    Output.position = vec4f(pos[vertex_index] * scaleUniform.scale + ourUniform.offset, 0.0, 1.0);
    Output.color = ourUniform.color;
    Output.scale = scaleUniform.scale;
    Output.offset = ourUniform.offset;
    return Output;
}