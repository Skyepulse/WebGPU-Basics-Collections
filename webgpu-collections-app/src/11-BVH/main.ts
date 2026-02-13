
import * as glm from 'gl-matrix';

//================================//
import rayTraceVertWGSL from './shader_vert.wgsl?raw';
import rayTraceFragWGSL from './shader_frag.wgsl?raw';

import normalVertWgsl from './normal_vert.wgsl?raw';
import normalFragWgsl from './normal_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice, CreateShaderModule, CreateTimestampQuerySet } from '@src/helpers/WebGPUutils';
import type { PipelineResources, TimestampQuerySet } from '@src/helpers/WebGPUutils';
import { addButton, addCheckbox, addSlider, cleanUtilElement, createLightContextMenu, createMaterialContextMenu, getInfoElement, getUtilElement, type SpotLight } from '@src/helpers/Others';
import { createCamera, moveCameraLocal, rotateCameraByMouse, setCameraPosition, setCameraNearFar, setCameraAspect, computePixelToRayMatrix, rotateCameraBy, cameraPointToRay, rayIntersectsAABB } from '@src/helpers/CameraHelpers';
import { createCornellBox3, type SceneInformation, type Transform } from '@src/helpers/GeometryUtils';
import { createPlaceholderImage, createPlaceholderTexture, createTextureFromImage, loadImageFromUrl, resizeImage, TextureType } from '@src/helpers/ImageHelpers';
import { flattenMaterial, flattenMaterialArray, MATERIAL_SIZE, type Material } from '@src/helpers/MaterialUtils';

//================================//
export async function startup_11(canvas: HTMLCanvasElement)
{
    const renderer = new RayTracer();
    await renderer.initialize(canvas);
    
    return renderer;
}

//================================//
const normalUniformDataSize = (16 * 3) * 4 + 2 * 16 * 4 + (48 * 3);
const rayTracerUniformDataSize = 224 + 16*4;

//================================//
interface normalObjects extends PipelineResources
{
    uniformBuffer: GPUBuffer;

    sceneInformation: SceneInformation;
    materials: Material[];

    materialUniforms: GPUBuffer[];  
    materialBindGroups: GPUBindGroup[];
    materialUniformBindGroupLayout: GPUBindGroupLayout;

    positionBuffers: GPUBuffer[];
    normalBuffers: GPUBuffer[];
    uvBuffers: GPUBuffer[];
    indexBuffers: GPUBuffer[];

    depthTexture: GPUTexture;
    sampler: GPUSampler;

    bvhLineGeometryBuffer: GPUBuffer;
    bvhLineCount: number;

    bvhDrawPipelineLayout: GPUPipelineLayout;
    bvhDrawPipeline: GPURenderPipeline;
};

interface rayTracerObjects extends PipelineResources
{
    uniformBuffer: GPUBuffer;

    materialBuffer: GPUBuffer;
    positionStorageBuffer: GPUBuffer;
    normalStorageBuffer: GPUBuffer;
    uvStorageBuffer: GPUBuffer;
    indexStorageBuffer: GPUBuffer;
    materialIndexStorageBuffer: GPUBuffer;

    materialBindGroupLayout: GPUBindGroupLayout;
    materialBindGroup: GPUBindGroup;

    sampler: GPUSampler;
    textureArray: GPUTexture;
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
    private lights: SpotLight[] = [];
    private a_c: number = 1.0;
    private a_l: number = 0.09;
    private a_q: number = 0.0032;
    private normalObjects: normalObjects;
    private rayTracerObjects: rayTracerObjects;

    //================================//
    private useRaytracing: boolean = false;
    private meshesInfo: any;

    //================================//
    private activeContextMenu: HTMLDivElement | null = null;

    //================================//
    private showBVH: boolean = false;
    private bvhDepth: number = Infinity;

    //================================//
    constructor () 
    {
        setCameraPosition(this.camera, 278, 500, -700);
        rotateCameraBy(this.camera, 0, -0.3);
        setCameraNearFar(this.camera, 0.1, 2000);
        this.camera.moveSpeed = 20.0;
        this.camera.rotateSpeed = 0.05;
        this.device = null;
        this.normalObjects = {} as normalObjects;
        this.rayTracerObjects = {} as rayTracerObjects;

        const light1 = {
            position: glm.vec3.fromValues(500, 500.0, 0),
            intensity: 1000.0,
            direction: glm.vec3.fromValues(-0.5, -0.9, 1),
            coneAngle: Math.PI / 6,
            color: glm.vec3.fromValues(0.85, 0.1, 0.1),
            enabled: true
        };
        this.lights.push(light1);

        const light2 = {
            position: glm.vec3.fromValues(50, 500.0, 0), 
            intensity: 1000.0,
            direction: glm.vec3.fromValues(0.5, -0.9, 1),
            coneAngle: Math.PI / 6,
            color: glm.vec3.fromValues(0.1, 0.85, 0.1),
            enabled: true
        };
        this.lights.push(light2);

        const light3 = {
            position: glm.vec3.fromValues(275, 255, 0),
            intensity: 1500.0,
            direction: glm.vec3.fromValues(0, 0, 1),
            coneAngle: Math.PI / 6,
            color: glm.vec3.fromValues(0.9, 0.9, 0.9),
            enabled: true
        };
        this.lights.push(light3);
    }

    //================================//
    initializeUtils()
    {
        const utilElement = getUtilElement();
        if (!utilElement) return;

        addCheckbox('Use Ray Tracing', this.useRaytracing, utilElement, (value) => { this.useRaytracing = value; });
        utilElement.appendChild(document.createElement('br'));

        this.lights.forEach((_, index) =>
        {
            const callback = (e: MouseEvent) => 
            {
                e.preventDefault();
                if (this.activeContextMenu) {
                    this.activeContextMenu.remove();
                    this.activeContextMenu = null;
                }
                const middleOfCanvas = {
                    x: (this.canvas!.offsetLeft + this.canvas!.width - 300) ,
                    y: (this.canvas!.offsetTop + this.canvas!.height / 2 - 150)
                };
                this.activeContextMenu = createLightContextMenu(middleOfCanvas, this.lights[index], `Edit Light ${index + 1}`, 
                    (newLight) => { this.lights[index] = newLight; },
                    () => { this.activeContextMenu?.remove(); this.activeContextMenu = null; }
                );
                document.body.appendChild(this.activeContextMenu);
            };
            utilElement.appendChild(document.createElement('br'));
            addButton(`Edit Light ${index + 1}`, utilElement, callback);
        });
        utilElement.appendChild(document.createElement('br'));
        addCheckbox('Show BVH', this.showBVH, utilElement, (value) => { this.showBVH = value; });
        utilElement.appendChild(document.createElement('br'));
        addSlider('BVH Depth', this.bvhDepth === Infinity ? 32 : this.bvhDepth, 1, 32, 1, utilElement, (value) => { this.bvhDepth = value === 32 ? Infinity : value; this.rebuildBVHBuffer(); });
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
        this.rayTracerObjects.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Ray Trace Bind Group Layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1, // positions
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2, // normals
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 3, // UVs
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 4, // indices
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 5, // material indices
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" },
                }
            ],
        });
        this.rayTracerObjects.materialBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Ray Trace Material Bind Group Layout',
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: { type: "filtering" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "float", viewDimension: "2d-array", multisampled: false },
            }]
        });

        this.rayTracerObjects.pipelineLayout = this.device.createPipelineLayout({
            label: 'Ray Trace Pipeline Layout',
            bindGroupLayouts: [this.rayTracerObjects.bindGroupLayout, this.rayTracerObjects.materialBindGroupLayout],
        });

        if (this.rayTracerObjects.shaderModule !== null) {
            this.rayTracerObjects.pipeline = this.device.createRenderPipeline({
                label: 'Ray Trace Pipeline',
                layout: this.rayTracerObjects.pipelineLayout,
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
            }]
        });
        this.normalObjects.materialUniformBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Material Uniform Bind Group Layout',
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: { type: "filtering" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "float", viewDimension: "2d" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "float", viewDimension: "2d" },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "float", viewDimension: "2d" },
            },
            {
                binding: 5,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "float", viewDimension: "2d" },
            }]
        });

        this.normalObjects.pipelineLayout = this.device.createPipelineLayout({
            label: 'Normal Pipeline Layout',
            bindGroupLayouts: [this.normalObjects.bindGroupLayout, this.normalObjects.materialUniformBindGroupLayout],
        });

        this.normalObjects.bvhDrawPipelineLayout = this.device.createPipelineLayout({
            label: 'BVH Draw Pipeline Layout',
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

            this.normalObjects.bvhDrawPipeline = this.device.createRenderPipeline({
                label: 'BVH Draw Pipeline',
                layout: this.normalObjects.bvhDrawPipelineLayout,
                vertex: {
                    module: this.normalObjects.shaderModule.vertex,
                    entryPoint: 'vsBVH',
                    buffers: [
                        {
                            arrayStride: 3*4,
                            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }]
                        }
                    ]
                },
                fragment: {
                    module: this.normalObjects.shaderModule.fragment,
                    entryPoint: 'fsBVH',
                    targets: [
                        {
                            format: this.presentationFormat
                        }
                    ],
                },
                primitive: {
                    topology: "line-list"
                },
                    depthStencil: {
                    format: "depth24plus",   // Must match depth texture
                    depthWriteEnabled: false,
                    depthCompare: "less",
                },
            });
        }

        this.timestampQuerySet = CreateTimestampQuerySet(this.device, 2);

        // Samplers
        this.normalObjects.sampler = this.device.createSampler({
            label: 'Normal Objects Sampler',
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
            addressModeU: "repeat",
            addressModeV: "repeat",
        });

        this.rayTracerObjects.sampler = this.device.createSampler({
            label: 'Ray Tracer Sampler',
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
            addressModeU: "repeat",
            addressModeV: "repeat",
        });
    }

    //================================//
    async initializeBuffers()
    {
        if (this.device === null) return;

        // Normal Objects Buffers
        const placeholderTexture = createPlaceholderTexture(this.device);

        const meshMaterials = this.meshesInfo?.meshMaterials || [];
        const info: SceneInformation = await createCornellBox3(meshMaterials);
        this.normalObjects.sceneInformation = info;
        this.meshesInfo = info.additionalInfo;

        const numMaterials = info.meshes.length;
        for (let mesh of info.meshes)
        {
            mesh.ComputeBVH();
        }

        this.normalObjects.materialUniforms = [];
        this.normalObjects.materialBindGroups = [];
        this.normalObjects.positionBuffers = [];
        this.normalObjects.normalBuffers = [];
        this.normalObjects.uvBuffers = [];
        this.normalObjects.indexBuffers = [];

        for (let matNum = 0; matNum < numMaterials; matNum++)
        {
            this.normalObjects.materialUniforms.push(this.device.createBuffer({
                label: 'Material Uniform Buffer ' + matNum,
                size: MATERIAL_SIZE * 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            }));

            const materialData = info.meshes[matNum].GetFlattenedMaterial();
            this.device.queue.writeBuffer(this.normalObjects.materialUniforms[matNum], 0, materialData as BufferSource);

            this.normalObjects.materialBindGroups.push(this.device.createBindGroup({
                label: 'Material Bind Group ' + matNum,
                layout: this.normalObjects.materialUniformBindGroupLayout,
                entries: [{
                    binding: 0,
                    resource: { buffer: this.normalObjects.materialUniforms[matNum] },
                },
                {
                    binding: 1,
                    resource: this.normalObjects.sampler,
                },
                {
                    binding: 2,
                    resource: info.meshes[matNum].Material.albedoGPUTexture ? info.meshes[matNum].Material.albedoGPUTexture!.createView() : placeholderTexture.createView(),
                },
                {
                    binding: 3,
                    resource: info.meshes[matNum].Material.metalnessGPUTexture ? info.meshes[matNum].Material.metalnessGPUTexture!.createView() : placeholderTexture.createView(),
                },
                {
                    binding: 4,
                    resource: info.meshes[matNum].Material.roughnessGPUTexture ? info.meshes[matNum].Material.roughnessGPUTexture!.createView() : placeholderTexture.createView(),
                },
                {
                    binding: 5,
                    resource: info.meshes[matNum].Material.normalGPUTexture ? info.meshes[matNum].Material.normalGPUTexture!.createView() : placeholderTexture.createView(),
                }],
            }));

            const vertData = info.meshes[matNum].getVertexData();
            this.normalObjects.positionBuffers.push(this.device.createBuffer({
                label: 'Normal Position Buffer ' + matNum,
                size: vertData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.positionBuffers[matNum], 0, vertData as BufferSource);

            const indexData = info.meshes[matNum].getIndexData16();
            this.normalObjects.indexBuffers.push(this.device.createBuffer({
                label: 'Normal Index Buffer ' + matNum,
                size: indexData.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.indexBuffers[matNum], 0, indexData as BufferSource);

            const normalData = info.meshes[matNum].getNormalData();
            this.normalObjects.normalBuffers.push(this.device.createBuffer({
                label: 'Normal Normal Buffer ' + matNum,
                size: normalData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.normalBuffers[matNum], 0, normalData as BufferSource);

            const uvData = info.meshes[matNum].getUVData();
            this.normalObjects.uvBuffers.push(this.device.createBuffer({
                label: 'Normal UV Buffer ' + matNum,
                size: uvData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.uvBuffers[matNum], 0, uvData as BufferSource);
        }

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

        const lineData = this.getBVHGeometry(Infinity); // This way create buffer at max capacity
        this.normalObjects.bvhLineGeometryBuffer = this.device.createBuffer({
            label: 'BVH Line Geometry Buffer',
            size: lineData.length * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.normalObjects.bvhLineGeometryBuffer, 0, lineData as BufferSource);

        // Ray Tracer Objects Buffers
        const flattenedPositions: number[] = [];
        const flattenedNormals: number[] = [];
        const flattenedUVs: number[] = [];
        const flattenedIndices: number[] = [];
        const flattenedMaterialIndices: number[] = [];
        let indexOffset = 0;
        for (let matNum = 0; matNum < numMaterials; matNum++)
        {
            let mesh = info.meshes[matNum];
            flattenedPositions.push(...mesh.getVertexData());
            flattenedNormals.push(...mesh.getNormalData());
            flattenedUVs.push(...mesh.getUVData());

            for (let index of mesh.getIndexData32())
            {
                flattenedIndices.push(index + indexOffset);
            }
            indexOffset += mesh.getNumVertices();
            
            for (let i = 0; i < mesh.getNumTriangles(); i++)
            {
                flattenedMaterialIndices.push(matNum); // One material index per triangle
            }
        }
        const positionData = new Float32Array(flattenedPositions);
        const normalData = new Float32Array(flattenedNormals);
        const uvData = new Float32Array(flattenedUVs);
        const indexData = new Uint32Array(flattenedIndices); // Mandatory to be u32 for storage buffer
        const materialIndexData = new Uint32Array(flattenedMaterialIndices);

        this.rayTracerObjects.uniformBuffer = this.device.createBuffer({
            label: 'Ray Tracer Uniform Buffer',
            size: rayTracerUniformDataSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.rayTracerObjects.positionStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Position Storage Buffer',
            size: positionData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.positionStorageBuffer, 0, positionData as BufferSource);

        this.rayTracerObjects.normalStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Normal Storage Buffer',
            size: normalData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.normalStorageBuffer, 0, normalData as BufferSource);

        this.rayTracerObjects.uvStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer UV Storage Buffer',
            size: uvData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.uvStorageBuffer, 0, uvData as BufferSource);

        this.rayTracerObjects.indexStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Index Storage Buffer',
            size: indexData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.indexStorageBuffer, 0, indexData as BufferSource);

        this.rayTracerObjects.materialIndexStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Material Index Storage Buffer',
            size: materialIndexData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.materialIndexStorageBuffer, 0, materialIndexData as BufferSource);

        this.rayTracerObjects.bindGroup = this.device.createBindGroup({
            label: 'Ray Tracer Bind Group',
            layout: this.rayTracerObjects.bindGroupLayout,
            entries: [{
                    binding: 0,
                    resource: { buffer: this.rayTracerObjects.uniformBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: this.rayTracerObjects.positionStorageBuffer },
                },
                {
                    binding: 2,
                    resource: { buffer: this.rayTracerObjects.normalStorageBuffer },
                },
                {
                    binding: 3,
                    resource: { buffer: this.rayTracerObjects.uvStorageBuffer },
                },
                {
                    binding: 4,
                    resource: { buffer: this.rayTracerObjects.indexStorageBuffer },
                },
                {
                    binding: 5,
                    resource: { buffer: this.rayTracerObjects.materialIndexStorageBuffer },
                }
            ],
        });

        // material buffer for ray tracer
        const materials = info.meshes.map(mesh => mesh.Material);
        const materialData = flattenMaterialArray(materials);
        this.rayTracerObjects.materialBuffer = this.device.createBuffer({
            label: 'Ray Tracer Material Storage Buffer',
            size: materialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.materialBuffer, 0, materialData as BufferSource);

        // Texture array creation
        const numTexturesPerMaterial = 4;
        var numTexturedMaterials = this.meshesInfo?.meshMaterials.filter((mat: Material) => mat.albedoTexture || mat.metalnessTexture || mat.roughnessTexture || mat.normalTexture).length || 0;
        if (numTexturedMaterials === 0) numTexturedMaterials = 1;

        const commonW = 256;
        const commonH = 256;

        this.rayTracerObjects.textureArray = this.device.createTexture({
            label: 'Ray Tracer Material Texture Array',
            size: [commonW, commonH, numTexturesPerMaterial * numTexturedMaterials],
            format: 'rgba8unorm',
            mipLevelCount: 1,
            sampleCount: 1,
            dimension: '2d',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT, 
        });

        const placeHolderImage = createPlaceholderImage(256, 32);
        for (let matNum = 0; matNum < numTexturedMaterials; matNum++)
        {
            const albedoImage = this.meshesInfo?.meshMaterials[matNum].albedoImage ? this.meshesInfo.meshMaterials[matNum].albedoImage : placeHolderImage;
            const metalnessImage = this.meshesInfo?.meshMaterials[matNum].metalnessImage ? this.meshesInfo.meshMaterials[matNum].metalnessImage : placeHolderImage;
            const roughnessImage = this.meshesInfo?.meshMaterials[matNum].roughnessImage ? this.meshesInfo.meshMaterials[matNum].roughnessImage : placeHolderImage;
            const normalImage = this.meshesInfo?.meshMaterials[matNum].normalImage ? this.meshesInfo.meshMaterials[matNum].normalImage : placeHolderImage;

            this.device.queue.copyExternalImageToTexture(
                { source: albedoImage },
                { texture: this.rayTracerObjects.textureArray, origin: [0, 0, matNum * numTexturesPerMaterial] },
                [commonW, commonH]
            );
            this.device.queue.copyExternalImageToTexture(
                { source: metalnessImage },
                { texture: this.rayTracerObjects.textureArray, origin: [0, 0, matNum * numTexturesPerMaterial + 1] },
                [commonW, commonH]
            );
            this.device.queue.copyExternalImageToTexture(
                { source: roughnessImage },
                { texture: this.rayTracerObjects.textureArray, origin: [0, 0, matNum * numTexturesPerMaterial + 2] },
                [commonW, commonH]
            );
            this.device.queue.copyExternalImageToTexture(
                { source: normalImage },
                { texture: this.rayTracerObjects.textureArray, origin: [0, 0, matNum * numTexturesPerMaterial + 3] },
                [commonW, commonH]
            );
        }

        this.rayTracerObjects.materialBindGroup = this.device.createBindGroup({
            label: 'Ray Tracer Material Bind Group',
            layout: this.rayTracerObjects.materialBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.rayTracerObjects.materialBuffer },
            },
            {
                binding: 1,
                resource: this.rayTracerObjects.sampler,
            },
            {
                binding: 2,
                resource: this.rayTracerObjects.textureArray.createView(),
            }],
        });
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
    private onMouseUp = (e: MouseEvent) => {
        this.isMouseDown = false;

        if (e.target !== this.canvas) return;

        // Check if we clicked on the context menu first
        if (this.activeContextMenu !== null)
        {
            const rect = this.activeContextMenu.getBoundingClientRect();
            if (e.clientX >= rect.left && e.clientX <= rect.right &&
                e.clientY >= rect.top && e.clientY <= rect.bottom)
            return;
        }

        let meshIndex = this.rayCastOnMeshes(e.clientX, e.clientY);
        if (meshIndex !== -1)
        {
            this.spawnMaterialContextMenu(meshIndex, e.clientX, e.clientY);
        }
    };

    //================================//
    private onMouseMove = (e: MouseEvent) => {
        if (!this.isMouseDown) return;

        const deltaX = e.clientX - this.lastMouseX;
        const deltaY = e.clientY - this.lastMouseY;

        const sensitivity = 0.05;

        rotateCameraByMouse(this.camera, deltaX * sensitivity, -deltaY * sensitivity);

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
        if (this.keysPressed.has('alt')) dy -= this.camera.moveSpeed; // Down (alt)

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

        await this.initializeBuffers();
        this.initializeUtils();
        this.initializeInputHandlers();

        this.mainLoop();
    }

    //================================//
    updateUniforms()
    {
        if (this.device === null) return;

        if (this.useRaytracing)
        {
            const data = new ArrayBuffer(rayTracerUniformDataSize);
            const floatView = new Float32Array(data);
            const uintView = new Uint32Array(data);

            floatView.set(computePixelToRayMatrix(this.camera), 0);
            floatView.set(this.camera.position, 16);
            uintView[19] =  0;
            floatView[20] = this.a_c;
            floatView[21] = this.a_l;
            floatView[22] = this.a_q;
            floatView[23] = 0.0;

            // All lights
            for (let i = 0; i < 3; i++)
            {
                if (i >= this.lights.length)
                    break;

                const light = this.lights[i];
                const baseIndex = 24 + i * 12; // Each light is in total 48 bytes, 12 floats

                floatView.set(light.position, baseIndex);
                floatView[baseIndex + 3] = light.intensity;

                floatView.set(light.direction, baseIndex + 4);
                floatView[baseIndex + 7] = light.coneAngle;

                floatView.set(light.color, baseIndex + 8);
                floatView[baseIndex + 11] = light.enabled ? 1.0 : 0.0;
                // pad
            }
            this.device.queue.writeBuffer(this.rayTracerObjects.uniformBuffer, 0, data);
        }
        else
        {
            const data = new ArrayBuffer(normalUniformDataSize);
            const floatView = new Float32Array(data);
            floatView.set(this.camera.modelMatrix, 0);
            floatView.set(this.camera.viewMatrix, 16);
            floatView.set(this.camera.projectionMatrix, 32);
            floatView.set(this.camera.position, 48); // vec3 + pad
            floatView[52] = this.a_c;
            floatView[53] = this.a_l;
            floatView[54] = this.a_q;
            floatView[55] = 0.0; // pad

            for (let i = 0; i < 3; i++)
            {
                if (i >= this.lights.length)
                    break;

                const light = this.lights[i];
                const baseIndex = 56 + i * 12;

                floatView.set(light.position, baseIndex);
                floatView[baseIndex + 3] = light.intensity;

                floatView.set(light.direction, baseIndex + 4);
                floatView[baseIndex + 7] = light.coneAngle;

                floatView.set(light.color, baseIndex + 8);
                floatView[baseIndex + 11] = light.enabled ? 1.0 : 0.0;
            }
            this.device.queue.writeBuffer(this.normalObjects.uniformBuffer, 0, data as BufferSource);
        }
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
            const depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined = !this.useRaytracing ? {
                view: this.normalObjects.depthTexture.createView(),
                depthLoadOp: 'clear' as const,
                depthStoreOp: 'store' as const,
                depthClearValue: 1.0,
            } : undefined;
            const renderPassDescriptor: GPURenderPassDescriptor = {
                label: 'basic canvas renderPass',
                colorAttachments: [{
                    view: textureView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 }
                }],
                depthStencilAttachment: depthStencilAttachment,
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

            if (this.useRaytracing)
            {
                pass.setPipeline(this.rayTracerObjects.pipeline);
                pass.setBindGroup(0, this.rayTracerObjects.bindGroup);
                pass.setBindGroup(1, this.rayTracerObjects.materialBindGroup);
                pass.draw(6); // Fullscreen quad
            }
            else
            {
                pass.setPipeline(this.normalObjects.pipeline);
                pass.setBindGroup(0, this.normalObjects.bindGroup);
                for (let matNum = 0; matNum < this.normalObjects.sceneInformation.meshes.length; matNum++)
                {
                    pass.setBindGroup(1, this.normalObjects.materialBindGroups[matNum]);
                    pass.setVertexBuffer(0, this.normalObjects.positionBuffers[matNum]);
                    pass.setVertexBuffer(1, this.normalObjects.normalBuffers[matNum]);
                    pass.setVertexBuffer(2, this.normalObjects.uvBuffers[matNum]);
                    pass.setIndexBuffer(this.normalObjects.indexBuffers[matNum], 'uint16');

                    pass.drawIndexed(this.normalObjects.indexBuffers[matNum].size / 2, 1, 0, 0, 0);
                }

                // BVH Rendering
                if (this.showBVH)
                {
                    pass.setPipeline(this.normalObjects.bvhDrawPipeline);
                    pass.setBindGroup(0, this.normalObjects.bindGroup);
                    pass.setVertexBuffer(0, this.normalObjects.bvhLineGeometryBuffer);
                    pass.draw(this.normalObjects.bvhLineCount);   
                }
            }
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
                    gpuTime = Number(times[1] - times[0]);
                    this.timestampQuerySet!.resultBuffer.unmap();
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
    async cleanup() 
    {

        await this.smallCleanup();
        this.removeContextMenu();

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
        // Clean rest of handlers
        this.removeInputHandlers();

        cleanUtilElement();

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
    changeMeshMaterial(meshIndex: number, newMaterial: Material)
    {
        if (meshIndex < 0 || meshIndex >= (this.meshesInfo?.meshMaterialIndices.length || 0)) return;

        const matName: string = newMaterial.name;
        const totalMaterialIndex = this.normalObjects.sceneInformation.meshes.findIndex(mesh => mesh.Material.name === matName) || -1;
        if (totalMaterialIndex === -1) return;

        this.meshesInfo!.meshMaterials[meshIndex] = newMaterial;
        this.normalObjects.sceneInformation.meshes[totalMaterialIndex].Material = newMaterial;

        const materialBufferIndex = this.meshesInfo!.meshMaterialIndices[meshIndex];
        const materialData = flattenMaterial(newMaterial);
        
        // NORMAL
        let buffer = this.normalObjects.materialUniforms[materialBufferIndex];
        this.device!.queue.writeBuffer(buffer, 0, materialData as BufferSource);

        // RAY TRACING
        const offset = materialBufferIndex * MATERIAL_SIZE * 4;
        this.device!.queue.writeBuffer(this.rayTracerObjects.materialBuffer, offset, materialData as BufferSource);
    }

    //================================//
    recreateBindGroup(material: Material)
    {
        const matName: string = material.name;
        const totalMaterialIndex = this.normalObjects.sceneInformation.meshes.findIndex(mesh => mesh.Material.name === matName) || -1;
        if (totalMaterialIndex === -1) return;

        const newBindGroup = this.device!.createBindGroup({
            label: 'Material Bind Group ' + totalMaterialIndex,
            layout: this.normalObjects.materialUniformBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.normalObjects.materialUniforms[totalMaterialIndex] },
            },
            {
                binding: 1,
                resource: this.normalObjects.sampler,
            },
            {
                binding: 2,
                resource: material.albedoGPUTexture ? material.albedoGPUTexture.createView() : createPlaceholderTexture(this.device!).createView(),
            },
            {
                binding: 3,
                resource: material.metalnessGPUTexture ? material.metalnessGPUTexture.createView() : createPlaceholderTexture(this.device!).createView(),
            },
            {
                binding: 4,
                resource: material.roughnessGPUTexture ? material.roughnessGPUTexture.createView() : createPlaceholderTexture(this.device!).createView(),
            },
            {
                binding: 5,
                resource: material.normalGPUTexture ? material.normalGPUTexture.createView() : createPlaceholderTexture(this.device!).createView(),
            }],
        });

        this.normalObjects.materialBindGroups[totalMaterialIndex] = newBindGroup;

        // For the raytrace pipeline, write the texture into the array at the correct index
        var index = material.textureIndex;
        for (let typeIndex = 0; typeIndex < 4; typeIndex++)
        {
            const image = (() => {
                switch (typeIndex)
                {
                    case 0: return material.albedoTexture;
                    case 1: return material.metalnessTexture;
                    case 2: return material.roughnessTexture;
                    case 3: return material.normalTexture;
                }
            })() || createPlaceholderImage();

            this.device!.queue.copyExternalImageToTexture(
                { source: image },
                { texture: this.rayTracerObjects.textureArray, origin: [0, 0, index * 4 + typeIndex] },
                [256, 256]
            );
        }
    }

    //================================//
    getBVHGeometry(desiredDepth: number): Float32Array
    {
        if (this.normalObjects.sceneInformation.meshes.length === 0) return new Float32Array();

        this.normalObjects.bvhLineCount = 0;
        const chunks: Float32Array[] = [];
        let totalLength = 0;
        for (let matNum = 0; matNum < this.normalObjects.sceneInformation.meshes.length; matNum++)
        {
            const { vertexData, count } = this.normalObjects.sceneInformation.meshes[matNum].BVH.generateWireframeGeometry(desiredDepth);
            chunks.push(vertexData);
            totalLength += vertexData.length;
            this.normalObjects.bvhLineCount += count;
        }
        const result = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of chunks)
        {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        return result;
    }

    //================================//
    rebuildBVHBuffer()
    {
        if (this.device === null) return;

        const lineData = this.getBVHGeometry(this.bvhDepth);
        this.normalObjects.bvhLineGeometryBuffer = this.device.createBuffer({
            label: 'BVH Line Geometry Buffer',
            size: lineData.length * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.normalObjects.bvhLineGeometryBuffer, 0, lineData as BufferSource);
    }

    //================================//
    rayCastOnMeshes(screenX: number, screenY: number): number
    {
        if (this.canvas === null || this.camera === null || this.meshesInfo === null) return -1;

        const transforms: Transform[] = this.meshesInfo.meshTransforms!;

        // Convert viewport coordinates to canvas-relative coordinates
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = screenX - rect.left;
        const canvasY = screenY - rect.top;

        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const ndcX = (2 * canvasX * scaleX) / this.canvas.width - 1;
        const ndcY = 1 - (2 * canvasY * scaleY) / this.canvas.height;

        const ray: Float32Array = cameraPointToRay(this.camera, ndcX, ndcY);

        // Now check if the ray intersects any of the meshes,
        // we know their world position and radius
        let currentClosestMeshIndex = -1;
        let currentClosestDistance = Number.POSITIVE_INFINITY;

        for (let i = 0; i < transforms.length; i++)
        {
            const transform = transforms[i];
            const center = transform.translation;
            const scale = transform.scale;

            const minBounds = [center[0] - scale[0], center[1] - scale[1], center[2] - scale[2]];
            const maxBounds = [center[0] + scale[0], center[1] + scale[1], center[2] + scale[2]];

            const distance = rayIntersectsAABB(this.camera.position, ray, minBounds, maxBounds);
            if (distance <= 0) continue;

            if (distance < currentClosestDistance)
            {
                currentClosestDistance = distance;
                currentClosestMeshIndex = i;
            }
        }

        return currentClosestMeshIndex;
    }

    //================================//
    spawnMaterialContextMenu(meshIndex: number, screenX: number, screenY: number)
    {
        if (this.canvas === null) return;

        this.removeContextMenu();
        const currentMaterial: Material = this.meshesInfo?.meshMaterials?.[meshIndex];
        if (!currentMaterial) return;

        this.activeContextMenu = createMaterialContextMenu(
            {x: screenX, y: screenY},
            currentMaterial,
            (newMaterial: Material) => {
                this.changeMeshMaterial(meshIndex, newMaterial);
            },
            () => {
                this.removeContextMenu();
            }
        );
        document.body.appendChild(this.activeContextMenu);

        const closeOnClickOutside = (e: MouseEvent) => {
            if (this.activeContextMenu && !this.activeContextMenu.contains(e.target as Node)) {
                this.removeContextMenu();
                document.removeEventListener('mousedown', closeOnClickOutside);
            }
        };

        // This delay is just to avoid immediate closing when clicking to open
        setTimeout(() => {
            document.addEventListener('mousedown', closeOnClickOutside);
        }, 0);
    }

    //================================//
    removeContextMenu()
    {
        if (this.activeContextMenu) 
        {
            this.activeContextMenu.remove();
            this.activeContextMenu = null;
        }
    }

    //================================//
    fetchTextureForMaterial(material: Material, type: TextureType, url: string): void
    {
        if (!material) return;  

        const promise: Promise<HTMLImageElement> = loadImageFromUrl(url);
        promise.then(image => {

            // Resize
            const resizedImage: HTMLImageElement = resizeImage(image, 256, 256);
            const gpuTexture = createTextureFromImage(this.device!, resizedImage);

            switch (type)
            {
                case TextureType.Albedo:
                    material.albedoTexture = resizedImage;
                    material.albedoGPUTexture = gpuTexture;
                    break;
                case TextureType.Metalness:
                    material.metalnessTexture = resizedImage;
                    material.metalnessGPUTexture = gpuTexture;
                    break;
                case TextureType.Roughness:
                    material.roughnessTexture = resizedImage;
                    material.roughnessGPUTexture = gpuTexture;
                    break;
                case TextureType.Normal:
                    material.normalTexture = resizedImage;
                    material.normalGPUTexture = gpuTexture;
                    break;
            }
            this.recreateBindGroup(material);
        }).catch(error => {
            console.error('Error loading texture with name:', url, 'and type:', TextureType[type], error);
        });
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