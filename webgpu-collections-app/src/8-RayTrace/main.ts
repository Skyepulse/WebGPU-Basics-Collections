//================================//
import rayTraceVertWGSL from './shader_vert.wgsl?raw';
import rayTraceFragWGSL from './shader_frag.wgsl?raw';

import normalVertWgsl from './normal_vert.wgsl?raw';
import normalFragWgsl from './normal_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice, CreateShaderModule } from '@src/helpers/WebGPUutils';
import type { PipelineResources, TimestampQuerySet } from '@src/helpers/WebGPUutils';
import { getInfoElement } from '@src/helpers/Others';
import { createCamera, moveCameraLocal, rotateCameraByMouse, setCameraPosition, setCameraNearFar, setCameraAspect } from '@src/helpers/CameraHelpers';
import { createCornellBox, type TopologyInformation } from '@src/helpers/GeometryUtils';

//================================//
export async function startup_8(canvas: HTMLCanvasElement)
{
    const renderer = new RayTracer();
    await renderer.initialize(canvas);
    
    return renderer;
}

//================================//
const normalUniformDataSize = (16 * 4 * 4) + (4 * 2); // = 224 bytes
interface normalObjects extends PipelineResources
{
    uniformBuffer: GPUBuffer;
    
    positionBuffer: GPUBuffer;
    normalBuffer: GPUBuffer;
    uvBuffer: GPUBuffer;
    colorBuffer: GPUBuffer;
    indexBuffer: GPUBuffer;
    numIndices: number;

    depthTexture: GPUTexture;
};

interface rayTracerObjects extends PipelineResources
{

};

interface lightObject
{
    position: Float32Array;
    color: Float32Array;
};

//================================//
class RayTracer
{
    //================================//
    private device: GPUDevice | null;
    private canvas: HTMLCanvasElement | null = null;
    private context: GPUCanvasContext | null = null;
    private presentationFormat: GPUTextureFormat | null = null;
    private timestampQuerySet: TimestampQuerySet | null = null;

    //================================//
    private animationFrameId: number | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private infoElement: HTMLElement | null = getInfoElement();

    //================================//
    private keysPressed: Set<string> = new Set();
    private isMouseDown: boolean = false;
    private lastMouseX: number = 0;
    private lastMouseY: number = 0;

    //================================//
    private camera = createCamera(1.0);
    private light: lightObject;
    private normalObjects: normalObjects;
    private rayTracerObjects: rayTracerObjects;

    //================================//
    constructor () 
    {
        setCameraPosition(this.camera, 278, 273, -800);
        setCameraNearFar(this.camera, 0.1, 2000); // Increase far plane to see entire Cornell box
        this.camera.moveSpeed = 20.0;
        this.camera.rotateSpeed = 0.05;
        this.device = null;
        this.normalObjects = {} as normalObjects;
        this.rayTracerObjects = {} as rayTracerObjects;
        this.light = {
            position: new Float32Array([200.0, 200.0, 200.5]),
            color: new Float32Array([1.0, 1.0, 1.0])
        };
    }

    //================================//
    async initialize(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.device = await RequestWebGPUDevice(['timestamp-query']);
        if (this.device === null || this.device === undefined) 
        {
            console.log("Was not able to acquire a WebGPU device.");
            return;
        }

        this.context = canvas.getContext('webgpu');
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        if (!this.context) {
            console.error("WebGPU context is not available.");
            return;
        }

        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied'
        });

        this.initializeShaderModules();        
        this.initializePipelines();
        this.initializeBuffers();
        this.initializeInputHandlers();

        await this.startRendering();
    }

    //================================//
    initializeShaderModules()
    {
        if (this.device === null) return;
        this.rayTracerObjects.shaderModule = CreateShaderModule(this.device, rayTraceVertWGSL, rayTraceFragWGSL, 'Ray Trace Shader Module');
        this.normalObjects.shaderModule = CreateShaderModule(this.device, normalVertWgsl, normalFragWgsl, 'Normal Shader Module');
    }

    //================================//
    initializePipelines()
    {
        if (this.device === null || this.presentationFormat === null) return;

        // RAY TRACE PIPELINE
        if (this.rayTracerObjects.shaderModule !== null) {
            this.rayTracerObjects.pipeline = this.device.createRenderPipeline({
                label: 'Ray Trace Pipeline',
                layout: 'auto',
                vertex: {
                    module: this.rayTracerObjects.shaderModule.vertex,
                    entryPoint: 'vs',
                },
                fragment: {
                    module: this.rayTracerObjects.shaderModule.fragment,
                    entryPoint: 'fs',
                    targets: [
                        {
                            format: this.presentationFormat
                        }
                    ],
                }
            });
        }

        // NORMAL PIPELINE
        this.normalObjects.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Normal Bind Group Layout',
            entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform" },
        }]});

        this.normalObjects.pipelineLayout = this.device.createPipelineLayout({
            label: 'Normal Pipeline Layout',
            bindGroupLayouts: [this.normalObjects.bindGroupLayout],
        });

        this.normalObjects.depthTexture = this.device.createTexture({
            size: [this.canvas!.width, this.canvas!.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        if (this.normalObjects.shaderModule !== null) {
            this.normalObjects.pipeline = this.device.createRenderPipeline({
                label: 'Normal Pipeline',
                layout: this.normalObjects.pipelineLayout,
                vertex: {
                    module: this.normalObjects.shaderModule.vertex,
                    entryPoint: 'vs',
                    buffers: [{
                                arrayStride: 3 * 4,
                                attributes: [
                                { shaderLocation: 0, offset: 0, format: "float32x3" } // position
                                ]
                            },
                            {
                                arrayStride: 3 * 4,
                                attributes: [
                                { shaderLocation: 1, offset: 0, format: "float32x3" } // normal
                                ]
                            },
                            {
                                arrayStride: 2 * 4,
                                attributes: [
                                { shaderLocation: 2, offset: 0, format: "float32x2" } // uv
                                ]
                            },
                            {
                                arrayStride: 3 * 4,
                                attributes: [
                                { shaderLocation: 3, offset: 0, format: "float32x3" } // color
                                ]
                            }]
                },
                fragment: {
                    module: this.normalObjects.shaderModule.fragment,
                    entryPoint: 'fs',
                    targets: [
                        {
                            format: this.presentationFormat
                        }
                    ],
                },
                primitive: {
                    topology: "triangle-list",
                    cullMode: "back", // Cull back faces
                },
                    depthStencil: {
                    format: "depth24plus",   // Must match depth texture
                    depthWriteEnabled: true,
                    depthCompare: "less",    // Standard depth (Z) test
                },
            });
        }

        const timeStampQueryCount = 2;
        if (this.device.features.has('timestamp-query')) {

            const querySet = this.device.createQuerySet({
                label: 'timestamp query set',
                type: 'timestamp',
                count: timeStampQueryCount
            });

            const resolveBuffer = this.device.createBuffer({
                label: 'timestamp resolve buffer',
                size: timeStampQueryCount * 8,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
            });

            const resultBuffer = this.device.createBuffer({
                label: 'timestamp result buffer',
                size: timeStampQueryCount * 8,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            this.timestampQuerySet = { querySet, resolveBuffer, resultBuffer };
        }
    }

    //================================//
    initializeBuffers()
    {
        if (this.device === null) return;

        // Normal Objects Buffers
        const info: TopologyInformation = createCornellBox();
        this.normalObjects.positionBuffer = this.device.createBuffer({
            label: 'Normal Position Buffer',
            size: info.vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.normalObjects.positionBuffer, 0, info.vertexData as BufferSource);

        this.normalObjects.indexBuffer = this.device.createBuffer({
            label: 'Normal Index Buffer',
            size: info.indexData.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.normalObjects.indexBuffer, 0, info.indexData as BufferSource);
        this.normalObjects.numIndices = info.indexData.length;

        this.normalObjects.normalBuffer = this.device.createBuffer({
            label: 'Normal Normal Buffer',
            size: info.normalData!.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.normalObjects.normalBuffer, 0, info.normalData as BufferSource);

        this.normalObjects.uvBuffer = this.device.createBuffer({
            label: 'Normal UV Buffer',
            size: info.uvData!.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.normalObjects.uvBuffer, 0, info.uvData as BufferSource);

        this.normalObjects.colorBuffer = this.device.createBuffer({
            label: 'Normal Color Buffer',
            size: info.colorData!.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.normalObjects.colorBuffer, 0, info.colorData as BufferSource);

        this.normalObjects.uniformBuffer = this.device.createBuffer({
            label: 'Normal Uniform Buffer',
            size: normalUniformDataSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.normalObjects.bindGroup = this.device.createBindGroup({
            label: 'Normal Bind Group',
            layout: this.normalObjects.bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.normalObjects.uniformBuffer },
            }],
        });

        // Ray Tracer Objects Buffers
    }

    //================================//
    initializeInputHandlers()
    {
        if (!this.canvas) return;

        window.addEventListener('keydown', this.onKeyDown);
        window.addEventListener('keyup', this.onKeyUp);

        this.canvas.addEventListener('mousedown', this.onMouseDown);
        window.addEventListener('mouseup', this.onMouseUp);
        window.addEventListener('mousemove', this.onMouseMove);

        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    //================================//
    private onKeyDown = (e: KeyboardEvent) => {
        this.keysPressed.add(e.key.toLowerCase());
    };

    //================================//
    private onKeyUp = (e: KeyboardEvent) => {
        this.keysPressed.delete(e.key.toLowerCase());
    };

    //================================//
    private onMouseDown = (e: MouseEvent) => {
        this.isMouseDown = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
    };

    //================================//
    private onMouseUp = () => {
        this.isMouseDown = false;
    };

    //================================//
    private onMouseMove = (e: MouseEvent) => {
        if (!this.isMouseDown) return;

        const deltaX = e.clientX - this.lastMouseX;
        const deltaY = e.clientY - this.lastMouseY;

        rotateCameraByMouse(this.camera, -deltaX, -deltaY);

        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
    };

    //================================//
    handleInput()
    {
        let dx = 0, dy = 0, dz = 0;

        if (this.keysPressed.has('z') || this.keysPressed.has('w')) dz -= this.camera.moveSpeed; // Forward
        if (this.keysPressed.has('s')) dz += this.camera.moveSpeed; // Backward
        if (this.keysPressed.has('q') || this.keysPressed.has('a')) dx -= this.camera.moveSpeed; // Left
        if (this.keysPressed.has('d')) dx += this.camera.moveSpeed; // Right
        if (this.keysPressed.has(' ')) dy += this.camera.moveSpeed; // Up (space)
        if (this.keysPressed.has('shift')) dy -= this.camera.moveSpeed; // Down (shift)

        if (dx !== 0 || dy !== 0 || dz !== 0) 
        {
            moveCameraLocal(this.camera, -dz, dx, dy);
        }

        // rotation with arrow keys
        if (this.keysPressed.has('arrowleft')) rotateCameraByMouse(this.camera, -1, 0); // Yaw left
        if (this.keysPressed.has('arrowright')) rotateCameraByMouse(this.camera, 1, 0);
        if (this.keysPressed.has('arrowup')) rotateCameraByMouse(this.camera, 0, 1); // Pitch up
        if (this.keysPressed.has('arrowdown')) rotateCameraByMouse(this.camera, 0, -1);
    }

    //================================//
    async startRendering()
    {
        await this.smallCleanup();
        this.mainLoop();
    }

    //================================//
    updateUniforms()
    {
        if (this.device === null) return;

        const data = new Float32Array(16 * 3 + 4 + 4);
        let offset = 0;
        data.set(this.camera.modelMatrix, offset); offset += 16;
        data.set(this.camera.viewMatrix, offset); offset += 16;
        data.set(this.camera.projectionMatrix, offset); offset += 16;
        data.set(this.camera.position, offset); offset += 4;
        data.set(this.light.color, offset); offset += 4;
        this.device.queue.writeBuffer(this.normalObjects.uniformBuffer, 0, data as BufferSource);
    }

    //================================//
    mainLoop()
    {
        if (this.device === null || this.canvas === null) return;

        let then = 0;
        let totalTime = 0;
        let gpuTime = 0;

        // RENDER LOOP
        const render = (now: number) =>
        {
            if (this.canvas === null || this.device === null || this.context === null) return;

            const dt = now - then;
            totalTime += dt;
            then = now;
            const startTime = performance.now();

            this.handleInput();
            this.updateUniforms();

            const textureView = this.context.getCurrentTexture().createView();
            const renderPassDescriptor: GPURenderPassDescriptor = {
                label: 'basic canvas renderPass',
                colorAttachments: [{
                    view: textureView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1 }
                }],
                depthStencilAttachment: {
                    view: this.normalObjects.depthTexture.createView(),
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                    depthClearValue: 1.0,
                },
                ... (this.timestampQuerySet != null && {
                    timestampWrites: {
                        querySet: this.timestampQuerySet.querySet,
                        beginningOfPassWriteIndex: 0,
                        endOfPassWriteIndex: 1,
                    }
                }),
            };

            const encoder = this.device.createCommandEncoder({label: 'Render Quad Encoder'});
            const pass = encoder.beginRenderPass(renderPassDescriptor);
            pass.setPipeline(this.normalObjects.pipeline);
            pass.setBindGroup(0, this.normalObjects.bindGroup);
            pass.setVertexBuffer(0, this.normalObjects.positionBuffer);
            pass.setVertexBuffer(1, this.normalObjects.normalBuffer);
            pass.setVertexBuffer(2, this.normalObjects.uvBuffer);
            pass.setVertexBuffer(3, this.normalObjects.colorBuffer);
            pass.setIndexBuffer(this.normalObjects.indexBuffer, 'uint16');
            pass.drawIndexed(this.normalObjects.numIndices);
            // Draw two triangles to m  ake a fullscreen quad
            pass.end();

            if (this.timestampQuerySet != null)
            {
                encoder.resolveQuerySet(
                    this.timestampQuerySet.querySet,
                    0, this.timestampQuerySet.querySet.count,
                    this.timestampQuerySet.resolveBuffer, 0
                );

                if (this.timestampQuerySet.resultBuffer.mapState === 'unmapped')
                    encoder.copyBufferToBuffer(this.timestampQuerySet.resolveBuffer, 0, this.timestampQuerySet.resultBuffer, 0, this.timestampQuerySet.resultBuffer.size);
            }

            const commandBuffer = encoder.finish();
            this.device.queue.submit([commandBuffer]);

            if (this.timestampQuerySet != null && this.timestampQuerySet.resultBuffer.mapState === 'unmapped')
            {
                this.timestampQuerySet.resultBuffer.mapAsync(GPUMapMode.READ).then(() =>
                {
                    const times = new BigUint64Array(this.timestampQuerySet!.resultBuffer.getMappedRange());
                    gpuTime = Number(times[1] - times[0]); // nanoseconds
                    this.timestampQuerySet!.resultBuffer.unmap(); // When finished reading the data.
                });
            }

            const jsTime = performance.now() - startTime;
            if ( this.infoElement && this.device )
            {
                const content = 
                `\
                FPS: ${(1000/dt).toFixed(1)}
                JS Time: ${jsTime.toFixed(1)} ms
                GPU Time: ${(gpuTime/1e6).toFixed(2)} ms
                `
                this.infoElement.textContent = content;
            }

            this.animationFrameId = requestAnimationFrame(render);
        }
        this.animationFrameId = requestAnimationFrame(render);

        this.resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {

                const width = entry.contentBoxSize[0].inlineSize;
                const height = entry.contentBoxSize[0].blockSize;

                if (this.canvas && this.device) {
                    this.canvas.width = Math.max(1, Math.min(width, this.device.limits.maxTextureDimension2D));
                    this.canvas.height = Math.max(1, Math.min(height, this.device.limits.maxTextureDimension2D));
                    
                    // Update camera aspect ratio to prevent distortion
                    setCameraAspect(this.camera, this.canvas.width / this.canvas.height);
                    
                    // Recreate depth texture with new size
                    if (this.normalObjects.depthTexture) {
                        this.normalObjects.depthTexture.destroy();
                        this.normalObjects.depthTexture = this.device.createTexture({
                            size: [this.canvas.width, this.canvas.height],
                            format: "depth24plus",
                            usage: GPUTextureUsage.RENDER_ATTACHMENT,
                        });
                    }
                }
            }
        });
        this.resizeObserver.observe(this.canvas);
    }

    //================================//
    async cleanup() {

        await this.smallCleanup();

        if (this.infoElement)
        {
            while(this.infoElement.firstChild) 
            {
                this.infoElement.removeChild(this.infoElement.firstChild);
            }
        }
    }

    //================================//
    async smallCleanup()
    {
        if (this.animationFrameId !== null) 
        {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.resizeObserver && this.canvas) 
        {
            this.resizeObserver.unobserve(this.canvas);
            this.resizeObserver = null;
        }
    }

    //================================//
    removeInputHandlers()
    {
        window.removeEventListener('keydown', this.onKeyDown);
        window.removeEventListener('keyup', this.onKeyUp);
        window.removeEventListener('mouseup', this.onMouseUp);
        window.removeEventListener('mousemove', this.onMouseMove);
        if (this.canvas) {
            this.canvas.removeEventListener('mousedown', this.onMouseDown);
        }
    }
}