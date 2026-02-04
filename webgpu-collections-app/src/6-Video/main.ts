//================================//
import simpleTextureVertWGSL from './simpleTexture_vert.wgsl?raw';
import simpleTextureFragWGSL from './simpleTexture_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice, CreateShaderModule } from '@src/helpers/WebGPUutils';
import type { ShaderModule, TimestampQuerySet } from '@src/helpers/WebGPUutils';
import { cleanUtilElement, getInfoElement, getUtilElement } from '@src/helpers/Others';
import { createQuadVertices, type TopologyInformation } from '@src/helpers/GeometryUtils';
import {mat4} from 'wgpu-matrix';
import { rand } from '@src/helpers/MathUtils';

//================================//
export async function startup_6(canvas: HTMLCanvasElement)
{
    const renderer = new TextureExampleRenderer();
    await renderer.initialize(canvas);
    
    return renderer; // Return renderer for proper cleanup
}

//================================//
interface InstanceInfo 
{
    scale: number;
}

//================================//
// Definition of a class to help with the 
// rendering, etc...
class TextureExampleRenderer
{
    //================================//
    private device: GPUDevice | null;
    private canvas: HTMLCanvasElement | null = null;
    private context: GPUCanvasContext | null = null;
    private presentationFormat: GPUTextureFormat | null = null;

    private simpleTextureModule: ShaderModule | null = null;
    private simpleTexturePipeline: GPURenderPipeline | null = null;
    private timestampQuerySet: TimestampQuerySet | null = null;

    //================================//
    private video: HTMLVideoElement | null = null;
    private animationFrameId: number | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private infoElement: HTMLElement | null = getInfoElement();

    //================================//
    private vertexBuffer: GPUBuffer | null = null;
    private indexBuffer: GPUBuffer | null = null;
    private staticBuffer: GPUBuffer | null = null;
    private changingBuffer: GPUBuffer | null = null;
    private storageBuffer: GPUBuffer | null = null;
    private perInstanceOffsets: Float32Array | null = null;

    //================================//
    private static maxObjects = 100;
    private static minObjects = 1;
    private numberOfObjects = 10;
    private newNumberOfObjects = this.numberOfObjects;
    private slider: HTMLInputElement | null = null;

    //================================//
    constructor () {
        this.device = null;
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

        // Initialize all Shader Modules
        this.initializeShaderModules();

        // Initialize Pipelines
        this.initializePipelines();

        this.addNumberOfObjectsSlider();

        await this.startRendering();
    }

    //================================//
    initializeShaderModules()
    {
        if (this.device === null) return;

        this.simpleTextureModule = CreateShaderModule(this.device, simpleTextureVertWGSL, simpleTextureFragWGSL, "simple texture");
    }

    //================================//
    initializePipelines()
    {
        if (this.device === null || this.presentationFormat === null) return;

        // Simple texture video pipeline
        if (this.simpleTextureModule !== null) {
            this.simpleTexturePipeline = this.device.createRenderPipeline({
                label: 'Simple Texture Video Pipeline',
                layout: 'auto',
                vertex: {
                    module: this.simpleTextureModule.vertex,
                    entryPoint: 'vs',
                    buffers: [
                        {
                            arrayStride: 2 * 4, // position (vec2f)
                            attributes: [
                                { shaderLocation: 0, offset: 0, format: 'float32x2' }
                            ]
                        },
                        {
                            arrayStride: 2 * 4, // offset (vec2f)
                            stepMode: 'instance', // Per instance
                            attributes: [
                                { shaderLocation: 1, offset: 0, format: 'float32x2' }
                            ]
                        },
                        {
                            arrayStride: 2 * 4, // scale (vec2f)
                            stepMode: 'instance',
                            attributes: [
                                { shaderLocation: 2, offset: 0, format: 'float32x2' }
                            ]
                        }
                    ],
                },
                fragment: {
                    module: this.simpleTextureModule.fragment,
                    entryPoint: 'fs',
                    targets: [
                        {
                            format: this.presentationFormat
                        }
                    ],
                }
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

    // SPECIAL METHODS FOR THE TEXTURE AND VIDEO RENDERING
    //================================//
    async startRendering()
    {
        await this.smallCleanup();
        await this.initializeVideo();
        this.simpleTextureContentInit();
    }

    //================================//
    async initializeVideo()
    {
        this.video = document.createElement('video');
        this.video.crossOrigin = 'anonymous'; // Pass CORS
        this.video.muted = true;
        this.video.playsInline = true;
        this.video.loop = true;
        this.video.preload = 'auto';
        this.video.src = encodeURI('https://githubpagesvideos.s3.eu-north-1.amazonaws.com/GlassOverflowDemo.mp4');

        await this.startAndWaitVideo(this.video);
    }

    //================================//
    startAndWaitVideo(video: HTMLVideoElement)
    {
        if (video === null) return;

        return new Promise<void>((resolve, reject) => 
        {
            video.addEventListener('error', reject); // reject on video error

            if ('requestVideoFrameCallback' in video) 
            {
                video.requestVideoFrameCallback(
                    (_now, _metadata) => {
                        resolve();
                    }
                );
            } 
            else 
            {
                const timeWatcher = (video: HTMLVideoElement) => 
                {
                    if (video.currentTime > 0) 
                    {
                        resolve();
                    } 
                    else 
                    {
                        requestAnimationFrame(() => timeWatcher(video));
                    }
                };
                timeWatcher(video);
            }
            video.play().catch(reject); // reject on play error
        });
    }

    //================================//
    simpleTextureContentInit()
    {
        if (this.device === null || this.video === null || this.canvas === null) return;

        const sampler = this.device.createSampler({
            addressModeU: 'repeat',
            addressModeV: 'repeat',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        const staticBufferSize =
            2 * 4; // offset vec2f
        const changingBufferSize =
            2 * 4;
        const storageBufferSize =
            16 * 4; // MVP mat4x4f

        const staticVertexBufferSize = staticBufferSize * this.numberOfObjects;
        const changingVertexBufferSize = changingBufferSize * this.numberOfObjects;
        const mvpStorageBufferSize = storageBufferSize * this.numberOfObjects;

        const quadTopologyInformation: TopologyInformation = createQuadVertices();
        const vertexBufferSize = quadTopologyInformation.vertexData.byteLength;
        const numVertices = quadTopologyInformation.numVertices;

        this.vertexBuffer = this.device.createBuffer({
            label: 'Quad vertex buffer',
            size: vertexBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.vertexBuffer, 0, quadTopologyInformation.vertexData as BufferSource);
        this.indexBuffer = this.device.createBuffer({
            label: 'Quad index buffer',
            size: quadTopologyInformation.indexData.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.indexBuffer, 0, quadTopologyInformation.indexData as BufferSource);

        this.staticBuffer = this.device.createBuffer({
            label: 'Static vertex buffer',
            size: staticVertexBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.changingBuffer = this.device.createBuffer({
            label: 'Changing vertex buffer',
            size: changingVertexBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.storageBuffer = this.device.createBuffer({
            label: 'MVP storage buffer',
            size: mvpStorageBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const objectInfos: InstanceInfo[] = [];
        {
            const staticVertexValuesF32 = new Float32Array(staticVertexBufferSize / 4);

            for (let i = 0; i < this.numberOfObjects; i++)
            {
                const dataOffsetF32 = i * (staticBufferSize / 4);
                staticVertexValuesF32.set([rand(-0.9, 0.9), rand(-0.9, 0.9)], dataOffsetF32); // offset
                const info: InstanceInfo = {
                    scale: rand(0.2, 0.6)
                };
                objectInfos.push(info);
            }
            this.perInstanceOffsets = new Float32Array(staticVertexValuesF32);
            this.device.queue.writeBuffer(this.staticBuffer, 0, staticVertexValuesF32 as BufferSource);
        }

        const changingValues = new Float32Array(changingVertexBufferSize / 4);
        const mvpValues = new Float32Array(mvpStorageBufferSize / 4);

        let then = 0;
        let totalTime = 0;
        let gpuTime = 0;
        const rotationTime = 10000;

        // RENDER LOOP
        const render = (now: number) =>
        {
            if (this.canvas === null || this.device === null || this.context === null) return;

            const dt = now - then;
            totalTime += dt;
            then = now;

            const startTime = performance.now();

            const fov = 60 * Math.PI / 180; // rads
            const aspect = this.canvas.width / this.canvas.height;
            const zNear = 0.1;
            const zFar = 2000;
            const projectionMatrix = mat4.perspective(fov, aspect, zNear, zFar);

            const cameraPos = [0, 0, 2];
            const up = [0, 1, 0];
            const target = [0, 0, 0];
            const view = mat4.lookAt(cameraPos, target, up);
            const VPM = mat4.multiply(projectionMatrix, view);

            const vp = VPM; // Projection + View matrix
            const angle = (totalTime / rotationTime) * 2 * Math.PI;

            const aspectRatio = (this.canvas.width / this.canvas.height) * 0.5;

            objectInfos.forEach((info, index) => {
                const CVoffset = index * (changingBufferSize / 4);
                const MVPOffset = index * (storageBufferSize / 4);

                changingValues.set([info.scale, info.scale], CVoffset); // scale

                const oX = this.perInstanceOffsets![2 * index + 0];
                const oY = this.perInstanceOffsets![2 * index + 1];

                const M = mat4.create();
                mat4.copy(vp, M);
                mat4.translate(M, [oX, oY, 0], M);
                mat4.rotateX(M, angle, M);
                mat4.rotateY(M, 0.2 * Math.sin(angle), M);
                mat4.scale(M, [2 * aspectRatio, 1 * aspectRatio, 1], M);

                mvpValues.set(M, MVPOffset);
            });

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

            const encoder = this.device.createCommandEncoder({label: 'Render Quad Encoder'});
            const pass = encoder.beginRenderPass(renderPassDescriptor);
            pass.setPipeline(this.simpleTexturePipeline!);
            pass.setVertexBuffer(0, this.vertexBuffer);
            pass.setVertexBuffer(1, this.staticBuffer);
            pass.setVertexBuffer(2, this.changingBuffer);
            pass.setIndexBuffer(this.indexBuffer!, 'uint16');

            const texture = this.device.importExternalTexture({source: this.video!});

            const bindGroup = this.device.createBindGroup({
                layout: this.simpleTexturePipeline!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: sampler},
                    { binding: 1, resource: texture},
                    { binding: 2, resource: { buffer: this.storageBuffer! } }
                ]
            });

            this.device.queue.writeBuffer(this.changingBuffer!, 0, changingValues as BufferSource);
            this.device.queue.writeBuffer(this.storageBuffer!, 0, mvpValues as BufferSource);

            pass.setBindGroup(0, bindGroup);
            pass.drawIndexed(numVertices, this.numberOfObjects);
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
                }
            }
        });
        this.resizeObserver.observe(this.canvas);
    }

    //================================//
    async cleanup() {

        await this.smallCleanup();

        // Other cleanup tasks
        if(this.slider)
        {
            this.slider = null;
        }

        cleanUtilElement();
    }

    //================================//
    async smallCleanup()
    {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.resizeObserver && this.canvas) {
            this.resizeObserver.unobserve(this.canvas);
            this.resizeObserver = null;
        }
        if (this.video) {
            this.video.pause();
            this.video.src = '';
            this.video.load();
            this.video = null;
        }
    }

    //================================//
    addNumberOfObjectsSlider()
    {
        const utilElement = getUtilElement();
        if (utilElement === null) return;

        const label = document.createElement('label');
        label.textContent = `Number of Objects: ${this.numberOfObjects}`;
        label.htmlFor = 'numObjectsSlider';
        utilElement.appendChild(label);

        this.slider = document.createElement('input');
        this.slider.type = 'range';
        this.slider.id = 'numObjectsSlider';
        this.slider.min = TextureExampleRenderer.minObjects.toString();
        this.slider.max = TextureExampleRenderer.maxObjects.toString();
        this.slider.value = this.numberOfObjects.toString();
        this.slider.step = '1';
        this.slider.style.width = '150px';
        utilElement.appendChild(this.slider);

        this.slider.addEventListener('input', (_event) => {
            if (!this.slider) return;
            this.newNumberOfObjects = parseInt(this.slider.value, 10);
            label.textContent = `Number of Objects: ${this.newNumberOfObjects}`;
        });

        let restarting = false;
        const commit = async () => {
            if (restarting) return;     // guard against double-fires (change + pointerup)
            restarting = true;
            try {
                this.numberOfObjects = this.newNumberOfObjects;
                await this.startRendering();
            } finally {
                restarting = false;
            }
        };

        // Most browsers: fires when the user releases the slider
        this.slider.addEventListener('change', commit);

        // Extra safety on some platforms: release events
        this.slider.addEventListener('pointerup', commit);
        this.slider.addEventListener('mouseup', commit);
        this.slider.addEventListener('touchend', commit);
    }
}