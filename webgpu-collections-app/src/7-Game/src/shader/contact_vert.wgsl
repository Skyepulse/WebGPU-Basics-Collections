// ============================== //
struct Screen {
    worldSize : vec2<f32>,
    pad : vec2<f32>,
};

// ============================== //
@group(0) @binding(0) var<uniform> screen : Screen;

// ============================== //
struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>,
};

// ============================== //
@vertex
fn vs(
    @location(0) position : vec2<f32>,     // Circle vertex local coordinates
    @location(1) instancePos : vec2<f32>   // Contact center (world-space)
) -> VSOut {
    var out : VSOut;

    let radius = 0.1;
    let posFixed = vec2<f32>(position.x, position.y);
    let world = instancePos + posFixed * radius;
    let ndc = vec2<f32>(
        (world.x / screen.worldSize.x) * 2.0 - 1.0,
        (world.y / screen.worldSize.y) * 2.0 - 1.0
    );

    out.position = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);

    out.color = vec3<f32>(1.0, 0.0, 0.0);
    return out;
}
