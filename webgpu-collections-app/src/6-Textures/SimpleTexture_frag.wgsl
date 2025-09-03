// ============================== //
struct VertexShaderOutput
{
    @builtin(position) Position : vec4f,
    @location(0) texCoord : vec2f,
};

// ============================== //
@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var myTexture: texture_external; // mandatory for Video

// ============================== //
@fragment
fn fs(fsInput: VertexShaderOutput) -> @location(0) vec4f
{
    return textureSampleBaseClampToEdge(myTexture, mySampler, fsInput.texCoord);
}