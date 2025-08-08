/// <reference types="@webgpu/types" />

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
