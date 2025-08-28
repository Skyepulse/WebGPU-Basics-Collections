// TRIVIA:
// maximum size of a Uniform buffer is 16kiB (read only)
// maximum size of a Storage buffer is 128MiB (read/write)

//================================//
import vertWGSL from './shader_vert.wgsl?raw';
import fragWGSL from './shader_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice } from '@src/helpers/WebGPUutils';
import { rand } from '@src/helpers/MathUtils';
import { createCircleVertices } from '@src/helpers/GeometryUtils';

//================================//
const kColorOffset = 0;
const kOffsetOffset = 4;

const knumObjects = 50;

//================================//
interface InstanceInfo {
    scale: number;
}

//================================//
// Pass WebGPU canvas as argument
export async function startup_4(canvas: HTMLCanvasElement)
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

    // Storage buffer sizes
    const staticStorageBufferSize = staticUniformBufferSize * knumObjects;
    const scaleStorageBufferSize = scaleUniformBufferSize * knumObjects;

    const CircleVertices = createCircleVertices({ radius: 1, innerRadius: 0.5 });
    const vertexStorageBufferSize = CircleVertices.byteLength;
    const numVertices = CircleVertices.length / 2;

    const staticStorageBuffer: GPUBuffer = createBasicStorageBuffer(device, staticStorageBufferSize);
    const scaleStorageBuffer: GPUBuffer = createBasicStorageBuffer(device, scaleStorageBufferSize);
    const vertexStorageBuffer: GPUBuffer = createBasicStorageBuffer(device, vertexStorageBufferSize);

    device.queue.writeBuffer(vertexStorageBuffer, 0, CircleVertices as BufferSource);
    const objectInfos: InstanceInfo[] = [];
    {
        const staticStorageValues = new Float32Array(staticStorageBufferSize / 4);
        for( let i = 0; i < knumObjects; i++)
        {
            const dataOffset = i * (staticUniformBufferSize / 4);

            staticStorageValues.set([rand(0.1), rand(0.1), rand(0.1), 1], dataOffset + kColorOffset);
            staticStorageValues.set([rand(-0.9, 0.9), rand(-0.9, 0.9)], dataOffset + kOffsetOffset);

            const info: InstanceInfo = {
                scale: rand(0.1, 0.4)
            };
            objectInfos.push(info);
        }
        device.queue.writeBuffer(staticStorageBuffer, 0, staticStorageValues as BufferSource);
    }

    const storageValues = new Float32Array(scaleStorageBufferSize / 4);
    const bindGroup: GPUBindGroup = createVertexStorageBindGroup(device, basicPipeline.getBindGroupLayout(0), staticStorageBuffer, scaleStorageBuffer, vertexStorageBuffer);

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

        render(device, canvas, webGPUcontext, basicPipeline, basicRenderPassDescriptor as GPURenderPassDescriptor, objectInfos, bindGroup, storageValues, scaleStorageBuffer, numVertices);
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
function render(device: GPUDevice, canvas: HTMLCanvasElement, context: GPUCanvasContext, pipeline: GPURenderPipeline, renderPassDescriptor: GPURenderPassDescriptor, objectInfos: InstanceInfo[], bindGroup: GPUBindGroup, storageValues: Float32Array, scaleStorageBuffer: GPUBuffer, numVertices: number) 
{
    // Set the texture we want to render to
    (renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = context.getCurrentTexture().createView();

    // Create a command buffer
    const encoder: GPUCommandEncoder = device.createCommandEncoder({ label: 'pass encoder' });

    // Begin a render pass
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);

    const aspectRatio = canvas.width / canvas.height;

    objectInfos.forEach((info, index) => {
        const offset = 2 * index;
        storageValues.set([info.scale / aspectRatio, info.scale], offset);
    });
    device.queue.writeBuffer(scaleStorageBuffer, 0, storageValues as BufferSource);

    pass.setBindGroup(0, bindGroup);
    pass.draw(numVertices, knumObjects); // 3 vertices, knumObjects instances
    pass.end();

    const commandBuffer: GPUCommandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
}

//================================//
function createBasicStorageBuffer(device: GPUDevice, size: number) {
    const buffer = device.createBuffer({
        label: 'storage buffer',
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    return buffer;
}

//================================//
function createVertexStorageBindGroup(device: GPUDevice, layout: GPUBindGroupLayout, staticStorageBuffer: GPUBuffer, changingStorageBuffer: GPUBuffer, vertexStorageBuffer: GPUBuffer) {
    return device.createBindGroup({
        label: 'storage bind group',
        layout: layout,
        entries: [{
            binding: 0,
            resource: {
                buffer: staticStorageBuffer
            }
        }, {
            binding: 1,
            resource: {
                buffer: changingStorageBuffer
            }
        }, {
            binding: 2,
            resource: {
                buffer: vertexStorageBuffer
            }
        }]
    });
}