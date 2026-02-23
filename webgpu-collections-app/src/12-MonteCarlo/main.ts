
import * as glm from 'gl-matrix';

//================================//
import rayTraceVertWGSL from './shader_vert.wgsl?raw';
import rayTraceFragWGSL from './shader_frag.wgsl?raw';

import normalVertWgsl from './normal_vert.wgsl?raw';
import normalFragWgsl from './normal_frag.wgsl?raw';

import bvhVertWGSL from './bvh_vert.wgsl?raw';
import bvhFragWGSL from './bvh_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice, CreateShaderModule, CreateTimestampQuerySet } from '@src/helpers/WebGPUutils';
import type { PipelineResources, ShaderModule, TimestampQuerySet } from '@src/helpers/WebGPUutils';
import { addButton, addCheckbox, addSlider, cleanUtilElement, createLightContextMenu, createMaterialContextMenu, getInfoElement, getUtilElement, type SpotLight } from '@src/helpers/Others';
import { createCamera, moveCameraLocal, rotateCameraByMouse, setCameraPosition, setCameraNearFar, setCameraAspect, computePixelToRayMatrix, rotateCameraBy, cameraPointToRay } from '@src/helpers/CameraHelpers';
import { createCornellBox4, type Ray, type SceneInformation } from '@src/helpers/GeometryUtils';
import { createPlaceholderImage, createPlaceholderTexture, createTextureFromImage, loadImageFromUrl, resizeImage, TextureType } from '@src/helpers/ImageHelpers';
import { flattenMaterial, flattenMaterialArray, MATERIAL_SIZE, type Material } from '@src/helpers/MaterialUtils';

//================================//
export async function startup_12(canvas: HTMLCanvasElement)
{
    const renderer = new RayTracer();
    await renderer.initialize(canvas);
    
    return renderer;
}

//================================//
const normalUniformDataSize = (16 * 2) * 4 + (2 * 4) * 4 + (48 * 3);
const rayTracerUniformDataSize = 224 + 16*4 + 16 + 16;
const meshInstanceSize = 20 * 4; // 16 byte matrix, + 4 floats

//================================//
enum RayTracerMode
{
    pathTrace = 0,
    BVHVisualization = 1,
    normal = 2,
    distance = 3,
    rayDirections = 4
}

//================================//
interface normalObjects extends PipelineResources
{
    uniformBuffer: GPUBuffer;

    meshesModelMatrixBuffers: GPUBuffer[];
    meshesNormalMatrixBuffers: GPUBuffer[];

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

    bvhLineGeometryBuffers: GPUBuffer[];
    bvhLineCounts: number[];

    bvhDrawPipelineLayout: GPUPipelineLayout;
    bvhDrawPipeline: GPURenderPipeline;
    bvhShaderModule: ShaderModule | null; 
};

interface rayTracerObjects extends PipelineResources
{
    uniformBuffer: GPUBuffer;

    materialBuffer: GPUBuffer;
    positionStorageBuffer: GPUBuffer;
    normalStorageBuffer: GPUBuffer;
    uvStorageBuffer: GPUBuffer;
    indexStorageBuffer: GPUBuffer;
    meshInstancesStorageBuffer: GPUBuffer;
    bvhNodesStorageBuffer: GPUBuffer;

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
    private usePathTracing: boolean = true;
    private meshesInfo: any;

    //================================//
    private activeContextMenu: HTMLDivElement | null = null;

    //================================//
    private showBVH: boolean = false;
    private bvhDepth: number = Infinity;
    private rayTracerMode: RayTracerMode = RayTracerMode.pathTrace;
    private ptDepth: number = 3;
    private ptSamples: number = 1;
    private randSeed: number = Math.floor(Math.random() * 0xFFFFFFFF);
    private russianRoulette: boolean = true;

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
            intensity: 5000.0,
            direction: glm.vec3.fromValues(-0.5, -0.9, 1),
            coneAngle: Math.PI / 6,
            color: glm.vec3.fromValues(0.85, 0.1, 0.1),
            enabled: false
        };
        this.lights.push(light1);

        const light2 = {
            position: glm.vec3.fromValues(50, 500.0, 0), 
            intensity: 5000.0,
            direction: glm.vec3.fromValues(0.5, -0.9, 1),
            coneAngle: Math.PI / 6,
            color: glm.vec3.fromValues(0.1, 0.85, 0.1),
            enabled: false
        };
        this.lights.push(light2);

        const light3 = {
            // Cube center
            position: glm.vec3.fromValues(278, 500, 279),
            intensity: 10000.0,
            direction: glm.vec3.fromValues(0, -1, 0),
            coneAngle: Math.PI / 3,
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

        addCheckbox('Use Path Tracing', this.usePathTracing, utilElement, (value) => { this.usePathTracing = value; });
        utilElement.appendChild(document.createElement('br'));
        
        addSlider('Depth of path tracing', this.ptDepth, 0, 20, 1, utilElement, (value) => { this.ptDepth = value; });
        utilElement.appendChild(document.createElement('br'));
        addSlider('Path tracing samples', this.ptSamples, 1, 100, 1, utilElement, (value) => { this.ptSamples = value; });

        addCheckbox('Russian Roulette', this.russianRoulette, utilElement, (value) => { this.russianRoulette = value; });

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
        addCheckbox('Show BVH', this.showBVH, utilElement, (value) => { this.showBVH = value; this.rayTracerMode = value ? RayTracerMode.BVHVisualization : RayTracerMode.pathTrace; });
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
        this.normalObjects.bvhShaderModule = CreateShaderModule(this.device, bvhVertWGSL, bvhFragWGSL, 'BVH Draw Shader Module');
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
                    binding: 5, // bvhNodes
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 6, // meshInstances
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
            },
            {
                binding: 6, // Model matrix
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 7, // Normal matrix
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "read-only-storage" },
            }],
        });

        this.normalObjects.pipelineLayout = this.device.createPipelineLayout({
            label: 'Normal Pipeline Layout',
            bindGroupLayouts: [this.normalObjects.bindGroupLayout, this.normalObjects.materialUniformBindGroupLayout],
        });

        this.normalObjects.bvhDrawPipelineLayout = this.device.createPipelineLayout({
            label: 'BVH Draw Pipeline Layout',
            bindGroupLayouts: [this.normalObjects.bindGroupLayout, this.normalObjects.materialUniformBindGroupLayout],
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
                    module: this.normalObjects.bvhShaderModule!.vertex,
                    entryPoint: 'vsBVH',
                    buffers: [
                        {
                            arrayStride: 3*4,
                            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }]
                        }
                    ]
                },
                fragment: {
                    module: this.normalObjects.bvhShaderModule!.fragment,
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
        const placeholderTexture = createPlaceholderTexture(this.device, 1024, 32);

        const meshMaterials = this.meshesInfo?.meshMaterials || [];
        const info: SceneInformation = await createCornellBox4(meshMaterials);
        this.normalObjects.sceneInformation = info;
        this.meshesInfo = info.additionalInfo;

        const numMaterials = info.meshes.length;
        this.normalObjects.materialUniforms = [];
        this.normalObjects.materialBindGroups = [];
        this.normalObjects.positionBuffers = [];
        this.normalObjects.normalBuffers = [];
        this.normalObjects.uvBuffers = [];
        this.normalObjects.indexBuffers = [];

        this.normalObjects.meshesModelMatrixBuffers = [];
        this.normalObjects.meshesNormalMatrixBuffers = [];

        for (let matNum = 0; matNum < numMaterials; matNum++)
        {
            this.normalObjects.meshesModelMatrixBuffers.push(this.device.createBuffer({
                label: 'Mesh Model Matrix Buffer ' + matNum,
                size: 16 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.meshesModelMatrixBuffers[matNum], 0, info.meshes[matNum].GetFlatWorldMatrix() as BufferSource);

            this.normalObjects.meshesNormalMatrixBuffers.push(this.device.createBuffer({
                label: 'Mesh Normal Matrix Buffer ' + matNum,
                size: 16 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.meshesNormalMatrixBuffers[matNum], 0, info.meshes[matNum].GetFlatNormalMatrix() as BufferSource);

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
                },
                {
                    binding: 6,
                    resource: { buffer: this.normalObjects.meshesModelMatrixBuffers[matNum] },
                },
                {
                        binding: 7,
                        resource: { buffer: this.normalObjects.meshesNormalMatrixBuffers[matNum] },
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

        const lineData: Float32Array[] = this.getBVHGeometry(Infinity); // This way create buffer at max capacity
        this.normalObjects.bvhLineGeometryBuffers = [];
        for (let i = 0; i < lineData.length; i++)
        {
            this.normalObjects.bvhLineGeometryBuffers[i] = this.device.createBuffer({
                label: `BVH Line Geometry Buffer ${i}`,
                size: lineData[i].byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(this.normalObjects.bvhLineGeometryBuffers[i], 0, lineData[i] as BufferSource);
        }

        // Ray Tracer Objects Buffers
        const flattenedPositions: number[] = [];
        const flattenedNormals: number[] = [];
        const flattenedUVs: number[] = [];
        const flattenedIndices: number[] = [];
        const flattenedMeshInstances: number[] = [];
        const bvhChunks: ArrayBuffer[] = [];
        let totalBVHBytes = 0; 

        let indexOffset = 0;
        let triangleOffset = 0;
        let vertexOffset = 0;
        let bvhRootIndex = 0;

        for (let matNum = 0; matNum < numMaterials; matNum++)
        {
            let mesh = info.meshes[matNum];
            flattenedPositions.push(...mesh.getVertexData());
            flattenedNormals.push(...mesh.getNormalData());
            flattenedUVs.push(...mesh.getUVData());

            // IMPORTANT CHANGE: here we need
            // to get the reordered indices to account
            // for BVH traversal after partitioning swapping
            const reorderedIndices = mesh.getReorderedIndexData32();
            for (let index of reorderedIndices)
            {
                flattenedIndices.push(index + indexOffset);
            }

            const { data: bvhData, numNodes: bvhNodeCount } = mesh.getFlattenedBVHData(bvhRootIndex);
            bvhChunks.push(bvhData);
            totalBVHBytes += bvhData.byteLength;

            const meshInstanceData = new ArrayBuffer(meshInstanceSize);
            const float32View = new Float32Array(meshInstanceData);
            const uint32View = new Uint32Array(meshInstanceData);
            float32View.set(mesh.GetFlatInverseWorldMatrix(), 0);
            uint32View[16] = bvhRootIndex;
            uint32View[17] = triangleOffset;
            uint32View[18] = vertexOffset;
            uint32View[19] = matNum;
            flattenedMeshInstances.push(...float32View);

            indexOffset += mesh.getNumVertices();
            triangleOffset += mesh.getNumTriangles();
            vertexOffset += mesh.getNumVertices();
            bvhRootIndex += bvhNodeCount;
        }
        const positionData = new Float32Array(flattenedPositions);
        const normalData = new Float32Array(flattenedNormals);
        const uvData = new Float32Array(flattenedUVs);
        const indexData = new Uint32Array(flattenedIndices); // Mandatory to be u32 for storage buffer
        const meshInstancesData = new Float32Array(flattenedMeshInstances);

        const bvhNodesData = new Uint8Array(totalBVHBytes);
        let bvhDataOffset = 0;
        for (let chunk of bvhChunks)
        {
            bvhNodesData.set(new Uint8Array(chunk), bvhDataOffset);
            bvhDataOffset += chunk.byteLength;
        }

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

        this.rayTracerObjects.bvhNodesStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer BVH Nodes Storage Buffer',
            size: bvhNodesData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.bvhNodesStorageBuffer, 0, bvhNodesData as BufferSource);

        this.rayTracerObjects.meshInstancesStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Mesh Instances Storage Buffer',
            size: meshInstancesData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.meshInstancesStorageBuffer, 0, meshInstancesData as BufferSource);

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
                    resource: { buffer: this.rayTracerObjects.bvhNodesStorageBuffer },
                },
                {
                    binding: 6,
                    resource: { buffer: this.rayTracerObjects.meshInstancesStorageBuffer },
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

        const commonW = 1024;
        const commonH = 1024;

        this.rayTracerObjects.textureArray = this.device.createTexture({
            label: 'Ray Tracer Material Texture Array',
            size: [commonW, commonH, numTexturesPerMaterial * numTexturedMaterials],
            format: 'rgba8unorm',
            mipLevelCount: 1,
            sampleCount: 1,
            dimension: '2d',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT, 
        });

        const placeHolderImage = createPlaceholderImage(1024, 32);
        for (let matNum = 0; matNum < numTexturedMaterials; matNum++)
        {
            const albedoImage = this.meshesInfo?.meshMaterials[matNum]?.albedoImage ? this.meshesInfo.meshMaterials[matNum].albedoImage : placeHolderImage;
            const metalnessImage = this.meshesInfo?.meshMaterials[matNum]?.metalnessImage ? this.meshesInfo.meshMaterials[matNum].metalnessImage : placeHolderImage;
            const roughnessImage = this.meshesInfo?.meshMaterials[matNum]?.roughnessImage ? this.meshesInfo.meshMaterials[matNum].roughnessImage : placeHolderImage;
            const normalImage = this.meshesInfo?.meshMaterials[matNum]?.normalImage ? this.meshesInfo.meshMaterials[matNum].normalImage : placeHolderImage;

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

        if (this.usePathTracing)
        {
            const data = new ArrayBuffer(rayTracerUniformDataSize);
            const floatView = new Float32Array(data);
            const uintView = new Uint32Array(data);

            floatView.set(computePixelToRayMatrix(this.camera), 0);
            floatView.set(this.camera.position, 16);
            uintView[19] =  this.rayTracerMode;
            floatView[20] = this.a_c;
            floatView[21] = this.a_l;
            floatView[22] = this.a_q;
            floatView[23] = this.bvhDepth;

            uintView[24] = this.ptDepth;
            uintView[25] = this.randSeed;
            uintView[26] = this.ptSamples;
            uintView[27] = this.russianRoulette ? 1 : 0;

            const canvasDimensions = new Float32Array([this.canvas!.width, this.canvas!.height]);
            floatView.set(canvasDimensions, 28);
            floatView[30] = 0.0; // pad
            floatView[31] = 0.0; // pad

            // All lights
            for (let i = 0; i < 3; i++)
            {
                if (i >= this.lights.length)
                    break;

                const light = this.lights[i];
                const baseIndex = 32 + i * 12; // Each light is in total 48 bytes, 12 floats

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
            floatView.set(this.camera.viewMatrix, 0);
            floatView.set(this.camera.projectionMatrix, 16);
            floatView.set(this.camera.position, 32); // vec3 + pad
            floatView[36] = this.a_c;
            floatView[37] = this.a_l;
            floatView[38] = this.a_q;
            floatView[39] = 0.0; // pad

            for (let i = 0; i < 3; i++)
            {
                if (i >= this.lights.length)
                    break;

                const light = this.lights[i];
                const baseIndex = 40 + i * 12;

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
    animate()
    {

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
            this.animate();

            const textureView = this.context.getCurrentTexture().createView();
            const depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined = !this.usePathTracing ? {
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

            if (this.usePathTracing)
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
                    for (let i = 0; i < this.normalObjects.bvhLineGeometryBuffers.length; i++)
                    {
                        pass.setBindGroup(1, this.normalObjects.materialBindGroups[i]);
                        pass.setVertexBuffer(0, this.normalObjects.bvhLineGeometryBuffers[i]);
                        pass.draw(this.normalObjects.bvhLineCounts[i]);   
                    }
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
        if (meshIndex < 0 || meshIndex >= (this.meshesInfo?.meshIndices.length || 0)) return;

        const matName: string = newMaterial.name;
        const totalMaterialIndex = this.normalObjects.sceneInformation.meshes.findIndex(mesh => mesh.Material.name === matName) || -1;
        if (totalMaterialIndex === -1) return;

        this.meshesInfo!.meshMaterials[meshIndex] = newMaterial;
        this.normalObjects.sceneInformation.meshes[totalMaterialIndex].Material = newMaterial;

        const materialBufferIndex = this.meshesInfo!.meshIndices[meshIndex];
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
            },
            {
                    binding: 6,
                    resource: { buffer: this.normalObjects.meshesModelMatrixBuffers[totalMaterialIndex] },
            },
            {
                    binding: 7,
                    resource: { buffer: this.normalObjects.meshesNormalMatrixBuffers[totalMaterialIndex] },
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
            })() || createPlaceholderImage(1024, 32);

            this.device!.queue.copyExternalImageToTexture(
                { source: image },
                { texture: this.rayTracerObjects.textureArray, origin: [0, 0, index * 4 + typeIndex] },
                [1024, 1024]
            );
        }
    }

    //================================//
    getBVHGeometry(desiredDepth: number): Float32Array[]
    {
        if (this.normalObjects.sceneInformation.meshes.length === 0) return [];

        this.normalObjects.bvhLineCounts = [];
        const chunks: Float32Array[] = [];
        for (let matNum = 0; matNum < this.normalObjects.sceneInformation.meshes.length; matNum++)
        {
            const { vertexData, count } = this.normalObjects.sceneInformation.meshes[matNum].GetBVHGeometry(desiredDepth);
            chunks.push(vertexData);
            this.normalObjects.bvhLineCounts.push(count);
        }
        return chunks;
    }

    //================================//
    rebuildBVHBuffer()
    {
        if (this.device === null) return;

        const lineData: Float32Array[] = this.getBVHGeometry(this.bvhDepth);
        
        for (let i = 0; i < lineData.length; i++)
        {
            this.normalObjects.bvhLineGeometryBuffers[i] = this.device.createBuffer({
                label: `BVH Line Geometry Buffer ${i}`,
                size: lineData[i].byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(this.normalObjects.bvhLineGeometryBuffers[i], 0, lineData[i] as BufferSource);
        }
    }

    //================================//
    rayCastOnMeshes(screenX: number, screenY: number): number
    {
        if (this.canvas === null || this.camera === null || this.meshesInfo === null) return -1;

        const potentialMeshesIndices: number[] = this.meshesInfo!.meshIndices;
        const potentialMeshes = potentialMeshesIndices.map(index => this.normalObjects.sceneInformation.meshes[index]);

        // Convert viewport coordinates to canvas-relative coordinates
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = screenX - rect.left;
        const canvasY = screenY - rect.top;

        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const ndcX = (2 * canvasX * scaleX) / this.canvas.width - 1;
        const ndcY = 1 - (2 * canvasY * scaleY) / this.canvas.height;

        const ray: Ray = cameraPointToRay(this.camera, ndcX, ndcY);

        // Now check if the ray intersects any of the meshes,
        // we know their world position and radius
        let currentClosestMeshIndex = -1;
        let currentClosestDistance = Number.POSITIVE_INFINITY;

        for (let i = 0; i < potentialMeshes.length; i++)
        {
            const mesh = potentialMeshes[i];
            const dist = mesh.intersectMeshWithRay(ray, this.bvhDepth);

            if (dist < 0) continue;

            if (dist < currentClosestDistance)
            {
                currentClosestDistance = dist;
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
        if (!material)
        {
            console.error('Material is undefined when trying to fetch texture with name:', url, 'and type:', TextureType[type]);
            return;
        };  

        const promise: Promise<HTMLImageElement> = loadImageFromUrl(url);
        promise.then(image => {

            // Resize (returns HTMLCanvasElement which WebGPU can use directly)
            const resizedImage = resizeImage(image, 1024, 1024);
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