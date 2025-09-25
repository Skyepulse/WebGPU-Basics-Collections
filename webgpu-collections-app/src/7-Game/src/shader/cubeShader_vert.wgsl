// ============================== //
struct vertexStruct {
    @location(0) position: vec2f,
    @location(1) color: vec4f,
    @location(2) worldPos: vec3f,
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

    let rotation : f32 = vert.worldPos.z;
    let cosR : f32 = cos(rotation);
    let sinR : f32 = sin(rotation);

    let scaledPos : vec2f = vert.position * vert.scale;
    let rotatedX : f32 = scaledPos.x * cosR - scaledPos.y * sinR;
    let rotatedY : f32 = scaledPos.x * sinR + scaledPos.y * cosR;
    let worldRotated : vec2f = vert.worldPos.xy + vec2f(rotatedX, rotatedY);

    // Map world [0..worldSize] -> NDC [-1..1]
    let ndc : vec2f = vec2f(
        (worldRotated.x / uScreen.worldSize.x) * 2.0 - 1.0,
        (worldRotated.y / uScreen.worldSize.y) * 2.0 - 1.0
    );

    out.position = vec4f(ndc, 0.0, 1.0);
    out.color = vert.color;
    return out;
}