// ============================== //
struct ourStruct {
    color: vec4f,
    offset: vec2f
};

struct scaleStruct {
    scale: vec2f
};

struct vertexStruct {
    position: vec2f
};

// ============================== //
struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(perspective) color: vec4f, // Inter stage variable example
    @location(1) @interpolate(perspective) scale: vec2f,
    @location(2) @interpolate(perspective) offset: vec2f,
};

// ============================== //
@group(0) @binding(0) var<storage, read> staticStorage: array<ourStruct>;
@group(0) @binding(1) var<storage, read> scaleStorage: array<scaleStruct>;
@group(0) @binding(2) var<storage, read> pos: array<vertexStruct>;

// ============================== //
@vertex
fn vs(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> OurVertexShaderOutput
{
    let staticInstance = staticStorage[instance_index];
    let scaleInstance = scaleStorage[instance_index];

    var Output: OurVertexShaderOutput;
    Output.position = vec4f(pos[vertex_index].position * scaleInstance.scale + staticInstance.offset, 0.0, 1.0);
    Output.color = staticInstance.color;
    Output.scale = scaleInstance.scale;
    Output.offset = staticInstance.offset;
    return Output;
}