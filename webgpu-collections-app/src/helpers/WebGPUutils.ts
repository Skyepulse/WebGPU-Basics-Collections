/// <reference types="@webgpu/types" />


//============== STRUCTS ==================//
export interface ShaderModule
{
    vertex: GPUShaderModule;
    fragment: GPUShaderModule;
}

//============== METHODS ==================//

/*
 * Request access to WebGPU in browser.
 * @returns device
 */
export async function RequestWebGPUDevice(): Promise<GPUDevice | null> 
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

    const device = await adapter.requestDevice();
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