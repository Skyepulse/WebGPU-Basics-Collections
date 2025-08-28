//================================//
import vertWGSL from './redTriangle_vert.wgsl?raw';
import fragWGSL from './redTriangle_frag.wgsl?raw';

//================================//
// Pass WebGPU canvas as argument
export async function startup_1(canvas: HTMLCanvasElement)
{
    // Check if the navigator supports WebGPU
    const adaptor = await navigator.gpu?.requestAdapter();
    const device = await adaptor?.requestDevice();

    if (!device) 
    {
        console.log("WebGPU is not supported on this device.");
        return;
    } else {
        console.log("WebGPU is supported on this device.");
    }

    // webGPUcontext
    const webGPUcontext: GPUCanvasContext | null = canvas.getContext('webgpu');
    // Format will either be 'bgra8unorm' or 'rgba8unorm'
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    if (!webGPUcontext) {
        console.error("WebGPU context is not available.");
        return;
    }

    webGPUcontext.configure({
        device,
        format: presentationFormat,
        alphaMode: 'premultiplied'
    });
    const basicModule: GPUShaderModule = createBasicModule(device);
    const basicPipeline: GPURenderPipeline = createBasicPipeline(device, basicModule, basicModule, presentationFormat);

    const basicRenderPassDescriptor = 
        {
            label: 'basic canvas renderPass',
            colorAttachments: [{
                view: undefined,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1 }
            }]
        }

    // The correct resolution based on canvas size
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const canvas = entry.target as HTMLCanvasElement;
        const width = entry.contentBoxSize[0].inlineSize;
        const height = entry.contentBoxSize[0].blockSize;
        canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
        canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
      }
    
        render(device, webGPUcontext, basicPipeline, basicRenderPassDescriptor as GPURenderPassDescriptor);
    });

    observer.observe(canvas);
}

//================================//
function createBasicModule(device: GPUDevice) {
    return device.createShaderModule({
        label: 'hardcoded red triangle',
        code: `${vertWGSL}\n${fragWGSL}`
    });
}

//================================//
function createBasicPipeline(device: GPUDevice, Vmodule: GPUShaderModule, Fmodule: GPUShaderModule, format: GPUTextureFormat) {
    return device.createRenderPipeline({
        label: 'basic red triangle pipeline',
        layout: 'auto',
        vertex: {
            module: Vmodule,
            entryPoint: 'vs'
        },
        fragment: {
            module: Fmodule,
            entryPoint: 'fs',
            targets: [{
                format
            }]
        }
    });
}

//================================//
function render(device: GPUDevice, context: GPUCanvasContext, pipeline: GPURenderPipeline, renderPassDescriptor: GPURenderPassDescriptor) 
{
    // Set the texture we want to render to
    (renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = context.getCurrentTexture().createView();

    // Create a command buffer
    const encoder: GPUCommandEncoder = device.createCommandEncoder({ label: 'pass encoder' });

    // Begin a render pass
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);
    pass.draw(3); // 3 vertices
    pass.end();

    const commandBuffer: GPUCommandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
}