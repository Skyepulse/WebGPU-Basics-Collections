// TRIVIA
// 8 bit optimization per channel ( 4 bytes total ) for color vs 4 x 32 bit floats ( 16 bytes total ).

//================================//
import vertWGSL from './shader_vert.wgsl?raw';
import fragWGSL from './shader_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice } from '@src/helpers/WebGPUutils';
import { rand } from '@src/helpers/MathUtils';
import { createCircleVerticesWithColor, type TopologyInformation } from '@src/helpers/GeometryUtils';

//================================//
const kColorOffset = 0;
const kOffsetOffset = 1; // 1 because color is now 4 bytes

const knumObjects = 50;

//================================//
interface InstanceInfo {
    scale: number;
}

//================================//
// Pass WebGPU canvas as argument
export async function startup_5(canvas: HTMLCanvasElement)
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
    const basicPipeline: GPURenderPipeline = createVertexBufferPipeline(device, vsModule, fsModule, presentationFormat);

    // Buffer size
    const staticBufferSize =
        1 * 4 + // color is 1 byte per channel, 4 channels total now
        2 * 4; // offset vec2f

    const scaleBufferSize = 
        2 * 4; // 2 32bit floats

    // Vertex buffer sizes
    const staticVertexBufferSize = staticBufferSize * knumObjects;
    const scaleVertexBufferSize = scaleBufferSize * knumObjects;

    const CircleTopologyInformation: TopologyInformation = createCircleVerticesWithColor({ radius: 1, innerRadius: 0.5 });
    const vertexBufferSize = CircleTopologyInformation.vertexData.byteLength;
    const numVertices = CircleTopologyInformation.numVertices; // 2 position floats (8 bytes), 1 color (4 bytes)

    const staticVertexBuffer: GPUBuffer = createVertexBuffer(device, staticVertexBufferSize);
    const scaleVertexBuffer: GPUBuffer = createVertexBuffer(device, scaleVertexBufferSize);
    const vertexBuffer: GPUBuffer = createVertexBuffer(device, vertexBufferSize);
    const indexBuffer: GPUBuffer = createIndexBuffer(device, CircleTopologyInformation.indexData.byteLength);

    device.queue.writeBuffer(vertexBuffer, 0, CircleTopologyInformation.vertexData as BufferSource);
    device.queue.writeBuffer(indexBuffer, 0, CircleTopologyInformation.indexData as BufferSource);

    const objectInfos: InstanceInfo[] = [];
    {
        const staticVertexValuesU8 = new Uint8Array(staticVertexBufferSize);
        const staticVertexValuesF32 = new Float32Array(staticVertexValuesU8.buffer); // Float 32 view of the same buffer

        for( let i = 0; i < knumObjects; i++)
        {
            const dataOffsetU8 = i * (staticBufferSize);
            const dataOffsetF32 = i * (staticBufferSize / 4);

            staticVertexValuesU8.set([Math.round(rand(0.1) * 255), Math.round(rand(0.1) * 255), Math.round(rand(0.1) * 255), 255], dataOffsetU8 + kColorOffset);
            staticVertexValuesF32.set([rand(-0.9, 0.9), rand(-0.9, 0.9)], dataOffsetF32 + kOffsetOffset);

            const info: InstanceInfo = {
                scale: rand(0.1, 0.4)
            };
            objectInfos.push(info);
        }
        device.queue.writeBuffer(staticVertexBuffer, 0, staticVertexValuesF32 as BufferSource);
    }

    const scaleValues = new Float32Array(scaleVertexBufferSize / 4);

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

        render(device, canvas, webGPUcontext, basicPipeline, basicRenderPassDescriptor as GPURenderPassDescriptor, objectInfos, staticVertexBuffer, scaleValues, scaleVertexBuffer, numVertices, vertexBuffer, indexBuffer);
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
function createVertexBufferPipeline(device: GPUDevice, Vmodule: GPUShaderModule, Fmodule: GPUShaderModule, format: GPUTextureFormat) {
    return device.createRenderPipeline({
        label: 'vertex buffer pipeline',
        layout: 'auto',
        vertex: {
            module: Vmodule,
            entryPoint: 'vs',
            buffers: [
                {
                    arrayStride: 2 * 4 + 1 * 4, // Position 2 floats, x and y, per vertex color 4 bytes
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 4, offset: 2 * 4, format: 'unorm8x4' },
                    ],
                },
                {
                    arrayStride: 2 * 4 + 1 * 4, // Color 4 bytes, r, g, b, a, Offset 2 floats, x and y
                    stepMode: 'instance', // This means this attribute will only advance to next value once per instance.
                    attributes: [
                        { shaderLocation: 1, offset: 0, format: 'unorm8x4' },
                        { shaderLocation: 2, offset: 1 * 4, format: 'float32x2' },
                    ],
                },
                {
                    arrayStride: 2 * 4, // Scale 2 floats, x and y
                    stepMode: 'instance', 
                    attributes: [
                        { shaderLocation: 3, offset: 0, format: 'float32x2' },
                    ],
                }
            ],
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
function render(device: GPUDevice, canvas: HTMLCanvasElement, context: GPUCanvasContext, pipeline: GPURenderPipeline, renderPassDescriptor: GPURenderPassDescriptor, objectInfos: InstanceInfo[], staticVertexBuffer: GPUBuffer, scaleValues: Float32Array, scaleVertexBuffer: GPUBuffer, numVertices: number, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer) 
{
    // Set the texture we want to render to
    (renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = context.getCurrentTexture().createView();

    // Create a command buffer
    const encoder: GPUCommandEncoder = device.createCommandEncoder({ label: 'pass encoder' });

    // Begin a render pass
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vertexBuffer); // linked to the 0 on the pipeline buffer definition
    pass.setVertexBuffer(1, staticVertexBuffer);
    pass.setVertexBuffer(2, scaleVertexBuffer);
    pass.setIndexBuffer(indexBuffer, "uint16");

    const aspectRatio = canvas.width / canvas.height;

    objectInfos.forEach((info, index) => {
        const offset = 2 * index;
        scaleValues.set([info.scale / aspectRatio, info.scale], offset);
    });
    device.queue.writeBuffer(scaleVertexBuffer, 0, scaleValues as BufferSource);

    pass.drawIndexed(numVertices, knumObjects); // 3 vertices, knumObjects instances
    pass.end();

    const commandBuffer: GPUCommandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
}

//================================//
function createVertexBuffer(device: GPUDevice, size: number) {
    const buffer = device.createBuffer({
        label: 'vertex buffer',
        size,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    return buffer;
}

//================================//
function createIndexBuffer(device: GPUDevice, size: number) {
    const buffer = device.createBuffer({
        label: 'index buffer',
        size,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    return buffer;
}