//================================//
import simpleTextureVertWGSL from './simpleTexture_vert.wgsl?raw';
import simpleTextureFragWGSL from './simpleTexture_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice, CreateShaderModule } from '@src/helpers/WebGPUutils';
import type { ShaderModule } from '@src/helpers/WebGPUutils';
import {mat4} from 'wgpu-matrix';

//================================//
export async function startup_6(canvas: HTMLCanvasElement)
{
    const renderer = new TextureExampleRenderer();
    await renderer.initialize(canvas);
    
    return renderer; // Return renderer for proper cleanup
}

//================================//
const kMatrixOffset = 0;

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

    //================================//
    private video: HTMLVideoElement | null = null;
    private animationFrameId: number | null = null;
    private resizeObserver: ResizeObserver | null = null;

    //================================//
    constructor () {
        this.device = null;
    }

    //================================//
    async initialize(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.device = await RequestWebGPUDevice();
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

        // Load and play the source data video in the background
        await this.initializeVideo();

        // By default the simple texture video pipeline is used
        this.simpleTextureContentInit();
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
                    entryPoint: 'vs'
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
    }

    // SPECIAL METHODS FOR THE TEXTURE AND VIDEO RENDERING

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
                    (now, metadata) => {
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

        const UniformMatrixBufferSize = 16 * 4; // 4 bytes per float
        const uniformMatrixBuffer = this.device.createBuffer({
            label: 'matrix uniform buffer',
            size: UniformMatrixBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const uniformValues = new Float32Array(UniformMatrixBufferSize / 4);
        const matrix = uniformValues.subarray(kMatrixOffset, 16);

        let lastTime = performance.now();
        let totalTime = 0;

        const rotationTime = 10000;
        const upDownTime = 5000;

        // Simple texture video render function
        const render = (now: number) =>
        {
            if (this.canvas === null || this.device === null || this.context === null) return;

            const dt = lastTime - now;
            lastTime = now;
            totalTime += dt;

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

            const textureView = this.context.getCurrentTexture().createView();
            const renderPassDescriptor: GPURenderPassDescriptor = {
                label: 'basic canvas renderPass',
                colorAttachments: [{
                    view: textureView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1 }
                }]
            };

            mat4.copy(VPM, matrix);
            mat4.rotateX(matrix, totalTime / rotationTime * 2 * Math.PI, matrix);
            mat4.translate(matrix, [-1.0, 0.5, 0], matrix);
            // mat4.translate(matrix, [0, Math.sin((totalTime + upDownTime / 2) / upDownTime * 2 * Math.PI) * 0.5, 0], matrix);
            mat4.scale(matrix, [2, -1, 1], matrix);

            const encoder = this.device.createCommandEncoder({label: 'Render Quad Encoder'});
            const pass = encoder.beginRenderPass(renderPassDescriptor);
            pass.setPipeline(this.simpleTexturePipeline!);

            const texture = this.device.importExternalTexture({source: this.video!});

            const bindGroup = this.device.createBindGroup({
                layout: this.simpleTexturePipeline!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: sampler},
                    { binding: 1, resource: texture},
                    { binding: 2, resource: uniformMatrixBuffer}
                ]
            });

            this.device.queue.writeBuffer(uniformMatrixBuffer, 0, uniformValues);

            pass.setBindGroup(0, bindGroup);
            pass.draw(6);
            pass.end();

            const commandBuffer = encoder.finish();
            this.device.queue.submit([commandBuffer]);

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
        console.log("Cleaning up resources");

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
        // Optionally: destroy GPU resources if needed
    }
}