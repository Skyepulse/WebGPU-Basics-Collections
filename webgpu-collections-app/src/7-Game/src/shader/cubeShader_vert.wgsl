// ============================== //
struct vertexStruct {
    @location(0) position: vec2f,
    @location(1) color: vec4f,
    @location(2) worldPos: vec2f,
    @location(3) scale: vec2f,
};

// ============================== //
struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(perspective) color: vec4f,
};

// ============================== //
struct ScreenInfo {
    worldSize : vec2f, // e.g. vec2f(100.0, 50.0)
};

@group(0) @binding(0) 
var<uniform> uScreen : ScreenInfo;

// ============================== //
@vertex
fn vs(
    vert: vertexStruct,
) -> OurVertexShaderOutput
{
    var out: OurVertexShaderOutput;

    let world : vec2f = vert.worldPos + vert.position * vert.scale;

    // Map world [0..worldSize] -> NDC [-1..1]
    let ndc : vec2f = vec2f(
        (world.x / uScreen.worldSize.x) * 2.0 - 1.0,
        (world.y / uScreen.worldSize.y) * 2.0 - 1.0
    );

    out.position = vec4f(ndc, 0.0, 1.0);
    out.color = vert.color;
    return out;
}