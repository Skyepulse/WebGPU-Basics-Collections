/*
 * GameRenderer.ts
 *
 * Responsible for rendering the main game view.
 *
 */

//================================//
import cubeVertWGSL from '../shader/cubeShader_vert.wgsl?raw';
import cubeFragWGSL from '../shader/cubeShader_frag.wgsl?raw';

//================================//
import type GameManager from "./GameManager";
import { RequestWebGPUDevice } from "@src/helpers/WebGPUutils";
import type { ShaderModule, TimestampQuerySet } from "@src/helpers/WebGPUutils";
import { CreateShaderModule, CreateTimestampQuerySet, ResolveTimestampQuery } from '@src/helpers/WebGPUutils';
import { createQuadVertices } from '@src/helpers/GeometryUtils';

const positionSize = 2 * 4; // 2 floats, 4 bytes each
const scaleSize = 2 * 4;    // 2 floats, 4 bytes each
const colorSize = 1 * 4;    // 4 bytes (1 byte per channel RGBA)
const vertexSize = 2 * 4; // position
const indicesPerInstance = 6;  // 2 triangles per quad

const initialInstanceSize = 256;
const screenUniformSize = 16; // Uniform buffers should be 16-byte aligned. We store 2 floats + 2 pad floats.
const xWorldSize = 100;
const yWorldSize = 50;

//================================//
class GameRenderer
{
    private gameManager: GameManager | null = null;

    private canvas: HTMLCanvasElement | null = null;
    private device: GPUDevice | null = null;
    private context: GPUCanvasContext | null = null;
    private presentationFormat: GPUTextureFormat | null = null;
    private observer: ResizeObserver | null = null;

    // Rendering pipeline
    private CubesShaderModule: ShaderModule | null = null;
    private CubesPipeline: GPURenderPipeline | null = null;

    // Storage buffers
    private vertexBuffer: GPUBuffer | null = null;
    private indexBuffer: GPUBuffer | null = null;
    private staticBuffer: GPUBuffer | null = null;
    private changingBuffer: GPUBuffer | null = null;
    private timestampQuerySet: TimestampQuerySet | null = null;
    private screenUniformBuffer: GPUBuffer | null = null;
    private screenBindGroup: GPUBindGroup | null = null;

    private changingCpuArray: Float32Array = new Float32Array(initialInstanceSize * (positionSize + scaleSize) / 4);

    // Members
    private numInstances: number = 0;
    private maxInstances: number = initialInstanceSize;
    private nextId: number = 1;
    private idToIndexMap: Map<number, number> = new Map();
    private indexToId: number[] = [];

    //=============== PUBLIC =================//
    constructor(canvas: HTMLCanvasElement, gameManager: GameManager)
    {
        this.canvas = canvas;
        this.gameManager = gameManager;
    }

    //================================//
    public async initialize()
    {
        if (!this.canvas)
        {
            this.gameManager?.logWarn("No canvas provided to GameRenderer.");
            return;
        }

        this.device = await RequestWebGPUDevice(['timestamp-query']);
        if (this.device === null || this.device === undefined) 
        {
            this.gameManager?.logWarn("Was not able to acquire a WebGPU device.");
            return;
        }

        this.context = this.canvas.getContext('webgpu');
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        if (!this.context) {
            this.gameManager?.logWarn("WebGPU context is not available.");
            return;
        }

        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied'
        });

        this.observer = new ResizeObserver(entries => {
            for (const entry of entries) {

                const width = entry.contentBoxSize[0].inlineSize;
                const height = entry.contentBoxSize[0].blockSize;

                if (this.canvas && this.device) {
                    this.canvas.width = Math.max(1, Math.min(width, this.device.limits.maxTextureDimension2D));
                    this.canvas.height = Math.max(1, Math.min(height, this.device.limits.maxTextureDimension2D));
                }
            }
        });
        this.observer.observe(this.canvas);

        this.buildBuffers();
        this.initializePipeline();
    }

    //================================//
    public addInstance(position: Float32Array, scale: Float32Array, color: Uint8Array): number | null
    {
        if (!this.device || !this.staticBuffer || !this.changingBuffer) return null;

        // Check if there are free slots
        let instanceIndex: number;

        if (this.numInstances >= this.maxInstances)
            this.extendBuffers();

        instanceIndex = this.numInstances++;

        this.device.queue.writeBuffer(this.staticBuffer, instanceIndex * colorSize, color as BufferSource);

        const id = this.nextId++;
        this.indexToId[instanceIndex] = id;
        this.idToIndexMap.set(id, instanceIndex);

        this.updateInstancePosition(id, position);
        this.updateInstanceScale(id, scale);

        return id;
    }

    //================================//
    public removeInstance(id: number): void
    {
        if (!this.device || !this.staticBuffer || !this.changingBuffer) return;

        const instanceIndex = this.idToIndexMap.get(id);
        if (instanceIndex === undefined) return;

        const lastIndex = this.numInstances - 1;

        if (instanceIndex !== lastIndex) // Need to swap with last
        {
            const commandEncoder = this.device.createCommandEncoder({ label: 'Remove instance encoder' });

            commandEncoder.copyBufferToBuffer(this.staticBuffer, lastIndex * colorSize, this.staticBuffer, instanceIndex * colorSize, colorSize);
            this.device.queue.submit([commandEncoder.finish()]);

            const a = this.changingCpuArray;
            const dstBase = instanceIndex * (positionSize + scaleSize) / 4;
            const srcBase = lastIndex * (positionSize + scaleSize) / 4;
            a[dstBase + 0] = a[srcBase + 0]; // pos.x
            a[dstBase + 1] = a[srcBase + 1]; // pos.y
            a[dstBase + 2] = a[srcBase + 2]; // scale.x
            a[dstBase + 3] = a[srcBase + 3]; // scale.y

            const movedId = this.indexToId[lastIndex];
            this.indexToId[instanceIndex] = movedId;
            this.idToIndexMap.set(movedId, instanceIndex);
        }

        // In any case, pop the last
        this.idToIndexMap.delete(id);
        this.indexToId.pop();
        this.numInstances--;
    }

    //================================//
    public updateInstanceScale(id: number, scale: Float32Array): void
    {
        const instanceIndex = this.idToIndexMap.get(id);
        if (instanceIndex === undefined) return;

        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 2] = scale[0];
        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 3] = scale[1];
    }

    //================================//
    public updateInstancePosition(id: number, position: Float32Array): void
    {
        const instanceIndex = this.idToIndexMap.get(id);
        if (instanceIndex === undefined) return;

        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 0] = position[0];
        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 1] = position[1];
    }

    //================================//
    public render()
    {
        if (!this.device || !this.context || !this.presentationFormat || !this.CubesPipeline || !this.changingBuffer) return;

        const textureView = this.context.getCurrentTexture().createView();
        const renderPassDescriptor: GPURenderPassDescriptor = {
            label: 'basic canvas renderPass',
            colorAttachments: [{
                view: textureView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1 }
            }],
            ... (this.timestampQuerySet != null && {
                timestampWrites: {
                    querySet: this.timestampQuerySet.querySet,
                    beginningOfPassWriteIndex: 0,
                    endOfPassWriteIndex: 1,
                }
            }),
        };

        const byteLen = this.numInstances * (positionSize + scaleSize);

        const encoder = this.device.createCommandEncoder({ label: 'canvas render encoder' });
        this.device.queue.writeBuffer(this.changingBuffer, 0, this.changingCpuArray.buffer, 0, byteLen);

        const pass = encoder.beginRenderPass(renderPassDescriptor);
        pass.setPipeline(this.CubesPipeline);
        pass.setVertexBuffer(0, this.vertexBuffer as GPUBuffer);
        pass.setVertexBuffer(1, this.staticBuffer as GPUBuffer);
        pass.setVertexBuffer(2, this.changingBuffer as GPUBuffer);
        pass.setIndexBuffer(this.indexBuffer as GPUBuffer, 'uint16');
        pass.setBindGroup(0, this.screenBindGroup as GPUBindGroup); 
        
        pass.drawIndexed(indicesPerInstance, this.numInstances, 0, 0, 0);
        pass.end();

        if (this.timestampQuerySet != null)
        {
            const res = ResolveTimestampQuery(this.timestampQuerySet, encoder);
            if (!res)
            {
                this.gameManager?.logWarn("Failed to resolve timestamp query.");
                return;
            }
        }

        this.device.queue.submit([encoder.finish()]);
    }

    //=============== PRIVATE =================//
    private buildBuffers()
    {
        if (!this.device) return;

        const staticBufferSize = this.maxInstances * (colorSize);
        const changingBufferSize = this.maxInstances * (positionSize + scaleSize);

        const quadTopology = createQuadVertices();
        const vertexBufferSize = quadTopology.vertexData.byteLength;
        const indexBufferSize = quadTopology.indexData.byteLength;

        this.vertexBuffer = this.device.createBuffer({
            label: 'Quad vertex buffer',
            size: vertexBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.vertexBuffer, 0, quadTopology.vertexData as BufferSource);

        this.indexBuffer = this.device.createBuffer({
            label: 'Quad index buffer',
            size: indexBufferSize,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.indexBuffer, 0, quadTopology.indexData as BufferSource);

        this.staticBuffer = this.device.createBuffer({
            label: 'Quad static instance buffer',
            size: staticBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.changingBuffer = this.device.createBuffer({    
            label: 'Quad changing instance buffer',
            size: changingBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.timestampQuerySet = CreateTimestampQuerySet(this.device, 2);

        this.screenUniformBuffer = this.device.createBuffer({
            label: 'Screen uniform buffer',
            size: screenUniformSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Write world size to uniform buffer (won't change)
        const screenData = new Float32Array([xWorldSize, yWorldSize, 0, 0]);
        this.device.queue.writeBuffer(this.screenUniformBuffer, 0, screenData.buffer, screenData.byteOffset, screenData.byteLength);
    }

    //================================//
    private extendBuffers()
    {
        if (!this.device || !this.staticBuffer || !this.changingBuffer || !this.indexBuffer) return;

        this.maxInstances *= 2;

        const newStaticBufferSize = this.maxInstances * (colorSize);
        const newChangingBufferSize = this.maxInstances * (positionSize + scaleSize);

        const newStaticBuffer = this.device.createBuffer({
            label: 'Extended static instance buffer',
            size: newStaticBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        const newChangingBuffer = this.device.createBuffer({
            label: 'Extended changing instance buffer',
            size: newChangingBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        // Copy old data to new buffers (and cpu array)
        const commandEncoder = this.device.createCommandEncoder({ label: 'Extend buffer encoder' });
        commandEncoder.copyBufferToBuffer(this.staticBuffer, 0, newStaticBuffer, 0, this.staticBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);

        const oldChangingArray = this.changingCpuArray;
        this.changingCpuArray = new Float32Array(this.maxInstances * (positionSize + scaleSize) / 4);
        this.changingCpuArray.set(oldChangingArray);

        this.staticBuffer.destroy();
        this.changingBuffer.destroy();

        this.staticBuffer = newStaticBuffer;
        this.changingBuffer = newChangingBuffer;
    }

    //================================//
    private initializePipeline()
    {
        if (!this.device || !this.presentationFormat) return;

        this.CubesShaderModule = CreateShaderModule(this.device, cubeVertWGSL, cubeFragWGSL, "Cubes Shader");
        if (!this.CubesShaderModule)
        {
            this.gameManager?.logWarn("Failed to create shader modules.");
            return;
        }

        this.CubesPipeline = this.device.createRenderPipeline({
            label: 'Cubes Render Pipeline',
            layout: 'auto',
            vertex: {
                module: this.CubesShaderModule.vertex,
                entryPoint: 'vs',
                buffers: [
                    {
                        arrayStride: vertexSize,
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: 'float32x2' } // Vertex position
                        ]
                    },
                    {
                        arrayStride: colorSize,
                        stepMode: 'instance',
                        attributes: [
                            { shaderLocation: 1, offset: 0, format: 'unorm8x4' } // Instance color
                        ]
                    },
                    {
                        arrayStride: positionSize + scaleSize,
                        stepMode: 'instance',
                        attributes: [
                            { shaderLocation: 2, offset: 0, format: 'float32x2' }, // Instance position
                            { shaderLocation: 3, offset: positionSize, format: 'float32x2' }  // Instance scale
                        ]
                    }
                ]
            },
            fragment: {
                module: this.CubesShaderModule.fragment,
                entryPoint: 'fs',
                targets: [
                    {
                        format: this.presentationFormat
                    }
                ]
            }
        });

        const bgl0 = this.CubesPipeline.getBindGroupLayout(0);
        if (!this.device || !this.screenUniformBuffer) return;

        this.screenBindGroup = this.device.createBindGroup({
            label: 'Screen uniform bind group',
            layout: bgl0,
            entries: [
                { binding: 0, resource: { buffer: this.screenUniformBuffer } }
            ]
        });
    }
}

//================================//
export default GameRenderer;