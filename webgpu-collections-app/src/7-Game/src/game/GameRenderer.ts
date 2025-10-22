/*
 * GameRenderer.ts
 *
 * Responsible for rendering the main game view.
 *
 */

//================================//
import cubeVertWGSL from '../shader/cubeShader_vert.wgsl?raw';
import cubeFragWGSL from '../shader/cubeShader_frag.wgsl?raw';
import contactVertWGSL from '../shader/contact_vert.wgsl?raw';
import contactFragWGSL from '../shader/contact_frag.wgsl?raw';

//================================//
import type GameManager from "./GameManager";
import type { ContactRender } from "./Manifold";
import RigidBox from "./RigidBox";
import { RequestWebGPUDevice } from "@src/helpers/WebGPUutils";
import type { ShaderModule, TimestampQuerySet } from "@src/helpers/WebGPUutils";
import { CreateShaderModule, CreateTimestampQuerySet, ResolveTimestampQuery } from '@src/helpers/WebGPUutils';
import { createQuadVertices, createCircleVerticesTopology } from '@src/helpers/GeometryUtils';
import * as glm from 'gl-matrix';

const positionSize = 3 * 4; // 2 floats for posx posy and rotation, 4 bytes each
const scaleSize = 2 * 4;    // 2 floats, 4 bytes each
const colorSize = 1 * 4;    // 4 bytes (1 byte per channel RGBA)
const vertexSize = 2 * 4; // position
const indicesPerInstance = 6;  // 2 triangles per quad

const initialInstanceSize = 256;
const screenUniformSize = 16; // Uniform buffers should be 16-byte aligned. We store 2 floats + 2 pad floats.

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

    // Rendering pipeline for contacts
    private ContactShaderModule: ShaderModule | null = null;
    private ContactPipeline: GPURenderPipeline | null = null;
    private cubePipelineLayout: GPUPipelineLayout | null = null;

    // Storage buffers
    private vertexBuffer: GPUBuffer | null = null;
    private indexBuffer: GPUBuffer | null = null;
    private staticBuffer: GPUBuffer | null = null;
    private changingBuffer: GPUBuffer | null = null;

    // contact buffers
    private contactVertexBuffer: GPUBuffer | null = null;
    private contactIndexBuffer: GPUBuffer | null = null;
    private contactPositionBuffer: GPUBuffer | null = null;

    // Timestamp query
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

    // Contact points
    private contactPositions: Float32Array = new Float32Array(0);
    private numContacts: number = 0;
    private maxContacts: number = 128;
    private contactIndicesPerInstance: number = 0;

    // Static world size (matches physics world)
    static xWorldSize: number = 100;
    static yWorldSize: number = 60;

    // Texture to render with MSAA
    private msaaTexture: GPUTexture | null = null;
    private msaaView: GPUTextureView | null = null;
    private sampleCount: number = 4;

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

                this.createMSAATexture(); // Recreate MSAA texture on resize
            }
        });
        this.observer.observe(this.canvas);

        this.createMSAATexture();
        this.buildBuffers();
        this.initializePipeline();
        this.initializeContactPipeline();
    }

    //================================//
    public addInstanceBox(RigidBox: RigidBox): number
    {
        return this.addInstance(RigidBox.getPosition(), RigidBox.getScale(), RigidBox.getColor());
    }

    //================================//
    public addInstance(position: glm.vec3, scale: glm.vec2, color: Uint8Array): number
    {
        if (!this.device || !this.staticBuffer || !this.changingBuffer) return -1;

        // Check if there are free slots
        let instanceIndex: number;

        if (this.numInstances >= this.maxInstances)
            this.extendBuffers();

        instanceIndex = this.numInstances++;

        this.device.queue.writeBuffer(this.staticBuffer, instanceIndex * colorSize, color as BufferSource);

        const id = this.nextId++;
        this.indexToId[instanceIndex] = id;
        this.idToIndexMap.set(id, instanceIndex);

        this.updateInstancePosition(id, position as Float32Array);
        this.updateInstanceScale(id, scale as Float32Array);

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

        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 3] = scale[0];
        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 4] = scale[1];
    }

    //================================//
    public updateInstancePosition(id: number, position: Float32Array): void
    {
        const instanceIndex = this.idToIndexMap.get(id);
        if (instanceIndex === undefined) return;

        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 0] = position[0];
        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 1] = position[1];
        this.changingCpuArray[instanceIndex * (positionSize + scaleSize) / 4 + 2] = position[2];
    }

    //================================//
    public updateContacts(contacts: ContactRender[]) {
        this.numContacts = Math.min(contacts.length, this.maxContacts);
        if (this.numContacts === 0) return;

        if (this.contactPositions.length < this.numContacts * 2)
            this.contactPositions = new Float32Array(this.maxContacts * 2);

        for (let i = 0; i < this.numContacts; ++i) {
            this.contactPositions[i * 2 + 0] = contacts[i].pos[0];
            this.contactPositions[i * 2 + 1] = contacts[i].pos[1];
        }

        if (this.device && this.contactPositionBuffer)
            this.device.queue.writeBuffer(this.contactPositionBuffer, 0, this.contactPositions as BufferSource);
    }

    //================================//
    public render()
    {
        if (!this.device || !this.context || !this.presentationFormat) return;

        const textureView = this.context.getCurrentTexture().createView();
        const renderPassDescriptor: GPURenderPassDescriptor = {
            label: 'basic canvas renderPass',
            colorAttachments: [{
                view: this.msaaView as GPUTextureView,
                resolveTarget: textureView, // ← resolves the 4x MSAA buffer into the canvas
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

        const encoder = this.device.createCommandEncoder({ label: 'canvas render encoder' });
        const pass = encoder.beginRenderPass(renderPassDescriptor);

        if (this.CubesPipeline && this.changingBuffer)
        {
            const byteLen = this.numInstances * (positionSize + scaleSize);
            this.device.queue.writeBuffer(this.changingBuffer, 0, this.changingCpuArray.buffer, 0, byteLen);

            pass.setPipeline(this.CubesPipeline);
            pass.setVertexBuffer(0, this.vertexBuffer as GPUBuffer);
            pass.setVertexBuffer(1, this.staticBuffer as GPUBuffer);
            pass.setVertexBuffer(2, this.changingBuffer as GPUBuffer);
            pass.setIndexBuffer(this.indexBuffer as GPUBuffer, 'uint16');
            pass.setBindGroup(0, this.screenBindGroup as GPUBindGroup);
            pass.drawIndexed(indicesPerInstance, this.numInstances, 0, 0, 0);
        } else
            this.gameManager?.logWarn("CubesPipeline or changingBuffer not initialized.");

        if (this.ContactPipeline && this.contactVertexBuffer && this.contactIndexBuffer && this.contactPositionBuffer)
        {
            pass.setPipeline(this.ContactPipeline);
            pass.setVertexBuffer(0, this.contactVertexBuffer);
            pass.setVertexBuffer(1, this.contactPositionBuffer);
            pass.setIndexBuffer(this.contactIndexBuffer, "uint16");
            pass.setBindGroup(0, this.screenBindGroup!);
            pass.drawIndexed(this.contactIndicesPerInstance, this.numContacts, 0, 0, 0);
        } else
            this.gameManager?.logWarn("ContactPipeline or contact buffers not initialized.");
         
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

    private createMSAATexture() 
    {
        if (!this.device || !this.presentationFormat || !this.canvas) return;

        this.msaaTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            sampleCount: this.sampleCount,
            format: this.presentationFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.msaaView = this.msaaTexture.createView();
    }
    //================================//
    private buildBuffers()
    {
        if (!this.device) return;

        const staticBufferSize = this.maxInstances * (colorSize);
        const changingBufferSize = this.maxInstances * (positionSize + scaleSize);

        // Boxes buffers
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

        // Contact buffers
        const circleTopology = createCircleVerticesTopology({ radius: 1, innerRadius: 0.01 });
        this.contactIndicesPerInstance = circleTopology.numVertices;

        this.contactVertexBuffer = this.device.createBuffer({
            label: 'Contact vertex buffer',
            size: circleTopology.vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.contactVertexBuffer, 0, circleTopology.vertexData as BufferSource);

        this.contactIndexBuffer = this.device.createBuffer({
            label: 'Contact index buffer',
            size: circleTopology.indexData.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.contactIndexBuffer, 0, circleTopology.indexData as BufferSource);

        this.contactPositionBuffer = this.device.createBuffer({
            label: 'Contact position buffer',
            size: this.maxContacts * 2 * 4, // 2 floats (x, y) per contact
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        // Instance data buffers
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
        const screenData = new Float32Array([GameRenderer.xWorldSize, GameRenderer.yWorldSize, 0, 0]);
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

        const bgl0 = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" }
                }
            ]
        });
        this.cubePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bgl0] });


        this.CubesPipeline = this.device.createRenderPipeline({
            label: 'Cubes Render Pipeline',
            layout: this.cubePipelineLayout,
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
                            { shaderLocation: 2, offset: 0, format: 'float32x3' }, // Instance position (x,y,rotation)
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
            },
            multisample: { count: 4 },
        });

        if (!this.device || !this.screenUniformBuffer) return;

        this.screenBindGroup = this.device.createBindGroup({
            label: 'Screen uniform bind group',
            layout: bgl0,
            entries: [
                { binding: 0, resource: { buffer: this.screenUniformBuffer } }
            ]
        });
    }

    //================================//
    private initializeContactPipeline()
    {
        if (!this.device || !this.presentationFormat || !this.cubePipelineLayout) return;

        this.ContactShaderModule = CreateShaderModule(this.device, contactVertWGSL, contactFragWGSL, "Contact Shader");
        if (!this.ContactShaderModule)
        {
            this.gameManager?.logWarn("Failed to create contact shader modules.");
            return;
        }

        this.ContactPipeline = this.device.createRenderPipeline({
            label: "Contacts Render Pipeline",
            layout: this.cubePipelineLayout,
            vertex: {
                module: this.ContactShaderModule.vertex,
                entryPoint: "vs",
                buffers: [
                    {
                        arrayStride: 8, // 2 floats (x,y)
                        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
                    },
                    {
                        arrayStride: 8, // 2 floats per instance position (x,y)
                        stepMode: "instance",
                        attributes: [{ shaderLocation: 1, offset: 0, format: "float32x2" }],
                    },
                ],
            },
            fragment: {
                module: this.ContactShaderModule.fragment,
                entryPoint: "fs",
                targets: [{ format: this.presentationFormat }],
            },
            primitive: { topology: "triangle-list" },
            multisample: { count: 4 },
        });
    }
}

//================================//
export default GameRenderer;