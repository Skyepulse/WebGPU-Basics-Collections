// TRIVIA:
// maximum size of a Uniform buffer is 16kiB (read only)
// maximum size of a Storage buffer is 128MiB (read/write)

//================================//
import vertWGSL from './shader_vert.wgsl?raw';
import fragWGSL from './shader_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice } from '@src/helpers/WebGPUutils';
import { rand } from '@src/helpers/MathUtils';

//================================//
const kColorOffset = 0;
const kOffsetOffset = 4;
const kScaleOffset = 0;

const knumObjects = 100;

//================================//
interface UniformInfo {
    uniformBindGroup: GPUBindGroup;
    uniformBuffer: GPUBuffer;
    uniformValues: Float32Array;
    scale: number;
}

//================================//
// Pass WebGPU canvas as argument
export async function startup_3(canvas: HTMLCanvasElement)
{
    const device: GPUDevice | null = await RequestWebGPUDevice();
    if (device === null || device === undefined) 
    {
        console.log("Was not able to acquire a WebGPU device.");
        return;
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

    const vsModule: GPUShaderModule = createBasicModule(device, 'hardcoded triangle', vertWGSL);
    const fsModule: GPUShaderModule = createBasicModule(device, 'hardcoded triangle', fragWGSL);
    const basicPipeline: GPURenderPipeline = createBasicPipeline(device, vsModule, fsModule, presentationFormat);

    // Buffer size
    const staticUniformBufferSize =
        4 * 4 + // color vec4f
        2 * 4 + // offset vec2f
        2 * 4;  // PADDING (needed since WGSL requires std430 layout)

    const scaleUniformBufferSize = 
        2 * 4; // 2 32bit floats

    // Object infos
    const objectInfos: UniformInfo[] = [];

    for(let i = 0; i < knumObjects; i++)
    {
        const staticUniformBuffer = createBasicUniformBuffer(device,staticUniformBufferSize);
        
        {
            const uniformValues = new Float32Array(staticUniformBufferSize / 4);
            uniformValues.set([rand(0.1), rand(0.1), rand(0.1), 1], kColorOffset);
            uniformValues.set([rand(-0.9, 0.9), rand(-0.9, 0.9)], kOffsetOffset);
            device.queue.writeBuffer(staticUniformBuffer, 0, uniformValues as BufferSource);
        }

        const uniformValues = new Float32Array(scaleUniformBufferSize / 4);
        const scaleUniformBuffer = createBasicUniformBuffer(device, scaleUniformBufferSize);
        const uniformBindGroup = createBasicUniformBindGroup(device, basicPipeline.getBindGroupLayout(0), staticUniformBuffer, scaleUniformBuffer);
        
        const uniformInfo: UniformInfo = {
            uniformBindGroup,
            uniformBuffer: scaleUniformBuffer,
            uniformValues,
            scale: rand(0.2, 0.5)
        };
        objectInfos.push(uniformInfo);
    }

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

        render(device, canvas, webGPUcontext, basicPipeline, basicRenderPassDescriptor as GPURenderPassDescriptor, objectInfos);
    });

    observer.observe(canvas);
}

//================================//
function createBasicModule(device: GPUDevice, label: string, code: string) {
    return device.createShaderModule({
        label: label,
        code: code
    });
}

//================================//
function createBasicPipeline(device: GPUDevice, Vmodule: GPUShaderModule, Fmodule: GPUShaderModule, format: GPUTextureFormat) {
    return device.createRenderPipeline({
        label: 'slightly more advanced pipeline',
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
function render(device: GPUDevice, canvas: HTMLCanvasElement, context: GPUCanvasContext, pipeline: GPURenderPipeline, renderPassDescriptor: GPURenderPassDescriptor, objectInfos: UniformInfo[]) 
{
    // Set the texture we want to render to
    (renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = context.getCurrentTexture().createView();

    // Create a command buffer
    const encoder: GPUCommandEncoder = device.createCommandEncoder({ label: 'pass encoder' });

    // Begin a render pass
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);

    const aspectRatio = canvas.width / canvas.height;

    for(const uniformInfo of objectInfos)
    {
        uniformInfo.uniformValues.set([uniformInfo.scale / aspectRatio, uniformInfo.scale], kScaleOffset);
        device.queue.writeBuffer(uniformInfo.uniformBuffer, 0, uniformInfo.uniformValues as BufferSource); // Here we are only writing the changing uniform buffer, a.k.a scale
        pass.setBindGroup(0, uniformInfo.uniformBindGroup);
        pass.draw(3);
    }
    pass.end();

    const commandBuffer: GPUCommandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
}

//================================//
function createBasicUniformBuffer(device: GPUDevice, size: number) {
    const buffer = device.createBuffer({
        label: 'uniform buffer',
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    return buffer;
}

//================================//
function createBasicUniformBindGroup(device: GPUDevice, layout: GPUBindGroupLayout, staticUniformBuffer: GPUBuffer, changingUniformBuffer: GPUBuffer) {
    return device.createBindGroup({
        label: 'uniform bind group',
        layout: layout,
        entries: [{
            binding: 0,
            resource: {
                buffer: staticUniformBuffer
            }
        }, {
            binding: 1,
            resource: {
                buffer: changingUniformBuffer
            }
        }]
    });
}