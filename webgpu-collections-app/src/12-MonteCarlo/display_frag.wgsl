struct VSOut 
{
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

struct DisplayUniforms
{
    frameCount: u32,
    denoiseRenderTexture: f32,
    padding1: u32,
    padding2: u32,
};

@group(0) @binding(0) var displayTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> displayUniforms: DisplayUniforms;


@fragment
fn fs(in: VSOut) -> @location(0) vec4f 
{
    let coord = vec2i(in.pos.xy);

    if (displayUniforms.denoiseRenderTexture < 0.5) 
    {
        let color = textureLoad(displayTex, coord, 0).rgb;
        return vec4f(color, 1.0);
    }
    else
    {
        // Simple gaussian filtering
        let kernel = array<f32, 25>(
            1.0,  4.0,  6.0,  4.0, 1.0,
            4.0, 16.0, 24.0, 16.0, 4.0,
            6.0, 24.0, 36.0, 24.0, 6.0,
            4.0, 16.0, 24.0, 16.0, 4.0,
            1.0,  4.0,  6.0,  4.0, 1.0
        );
        let kernelSum = 256.0;

        var color = vec3f(0.0);
        for (var y = -2; y <= 2; y++) // Go search up to two neighbors in each direction, 5x5 kernel
        {
            for (var x = -2; x <= 2; x++)
            {
                let idx = (y + 2) * 5 + (x + 2);
                let sample = textureLoad(displayTex, coord + vec2i(x, y), 0).rgb;
                color += sample * kernel[idx];
            }
        }

        return vec4f(color / kernelSum, 1.0);
    }
}