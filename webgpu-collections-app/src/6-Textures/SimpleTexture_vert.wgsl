// ============================== //
struct vertexStruct {
    @location(0) position: vec2f,
    @location(1) offset: vec2f,
    @location(2) scale: vec2f,
};

// ============================== //
struct MVPBuffer {
    mvp: array<mat4x4f>,
};

@group(0) @binding(2)
var<storage, read> mvpBuffer: MVPBuffer;

// ============================== //
struct VertexShaderOutput
{
    @builtin(position) Position : vec4f,
    @location(0) texCoord : vec2f,
};

// ============================== //
@vertex
fn vs(vert: vertexStruct, @builtin(instance_index) instanceIdx: u32) -> VertexShaderOutput
{
    var vsOutput: VertexShaderOutput;
    let mvp = mvpBuffer.mvp[instanceIdx];
    let pos = vert.position * vert.scale;
    vsOutput.Position = mvp * vec4f(pos, 0.0, 1.0);
    let uv = vert.position * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
    vsOutput.texCoord = uv;
    return vsOutput;
}