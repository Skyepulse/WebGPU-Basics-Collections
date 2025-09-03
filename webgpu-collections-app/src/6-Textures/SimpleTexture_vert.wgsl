// ============================== //
struct VertexShaderOutput
{
    @builtin(position) Position : vec4f,
    @location(0) texCoord : vec2f,
};

// ============================== //
struct Uniforms 
{
    matrix: mat4x4f
}

// ============================== //
@group(0) @binding(2) var<uniform> uni: Uniforms;

// ============================== //
@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexShaderOutput
{
    // Two triangles for a texture
    let pos: array<vec2f, 6> = array<vec2f, 6>(
        //T1
        vec2f( 0.0,  0.0),  // center
        vec2f( 1.0,  0.0),  // right, center
        vec2f( 0.0,  1.0),  // center, top

        // 2st triangle
        vec2f( 0.0,  1.0),  // center, top
        vec2f( 1.0,  0.0),  // right, center
        vec2f( 1.0,  1.0),  // right, top
    );

    var vsOutput: VertexShaderOutput;
    let xy = pos[vertex_index];
    vsOutput.Position = uni.matrix * vec4f(xy, 0.0, 1.0);
    vsOutput.texCoord = xy;
    return vsOutput;
}