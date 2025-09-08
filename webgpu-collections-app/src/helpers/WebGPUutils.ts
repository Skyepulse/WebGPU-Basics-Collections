/// <reference types="@webgpu/types" />


//============== STRUCTS ==================//
export interface ShaderModule
{
    vertex: GPUShaderModule;
    fragment: GPUShaderModule;
}

export interface TimestampQuerySet
{
    querySet: GPUQuerySet;
    resolveBuffer: GPUBuffer;
    resultBuffer: GPUBuffer;
}

//============== METHODS ==================//

/*
 * Request access to WebGPU in browser.
 * @returns device
 */
export async function RequestWebGPUDevice(features: GPUFeatureName[] = []): Promise<GPUDevice | null> 
{
    if (!navigator.gpu) {
        alert("WebGPU is not supported in this browser.");
        console.error("WebGPU is not supported in this browser.");
        return null;
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        alert("This browser supports WebGPU, but it appears disabled.");
        console.error("This browser supports WebGPU, but it appears disabled.");
        return null;
    }

    const logFeatureSupport = (name: GPUFeatureName): boolean =>
    {
        const supported = adapter.features.has(name);
        if (!supported) console.warn(`WebGPU feature not supported: ${name}`);
        else console.log(`WebGPU feature supported: ${name}`);
        return supported;
    };
    features = features.filter(f => logFeatureSupport(f));

    const device = await adapter.requestDevice(
        {
            requiredFeatures: features
        }
    );
    device.lost.then((info) => {
        console.error(`WebGPU device was lost: ${info.message}`);
    });

    return device;
}

/*
 *
 * Creates vertex and fragment shaders from WGSL source code.
 *
 */
export function CreateShaderModule(device: GPUDevice, vertexSource: string, fragmentSource: string, labelName: string = "shader module"): ShaderModule | null {
    const vertexShaderModule = device.createShaderModule({
        label: `${labelName} - vertex`,
        code: vertexSource
    });

    const fragmentShaderModule = device.createShaderModule({
        label: `${labelName} - fragment`,
        code: fragmentSource
    });

    return {
        vertex: vertexShaderModule,
        fragment: fragmentShaderModule
    };
}