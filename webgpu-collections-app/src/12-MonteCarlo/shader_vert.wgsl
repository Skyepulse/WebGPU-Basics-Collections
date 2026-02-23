// ============================== //
struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

// ============================== //
@vertex
fn vs(
    @builtin(vertex_index) vertex_index: u32
) -> OurVertexShaderOutput
{
    var positions = array<vec2f, 6>(
        vec2f(-1.0, -1.0),  // Triangle 1
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),  // Triangle 2
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );
    
    let pos = positions[vertex_index];
    
    var output: OurVertexShaderOutput;
    output.position = vec4f(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}