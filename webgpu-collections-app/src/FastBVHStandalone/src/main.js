import { RequestWebGPUDevice, CreateShaderModule, CreateTimestampQuerySet } from './helpers/WebGPUutils.js';
import { addButton, addCheckbox, addNumberInput, addProfilerFrameTime, addSlider, cleanUtilElement, createLightContextMenu, getInfoElement, getUtilElement } from './helpers/Others.js';
import { createCamera, moveCameraLocal, rotateCameraByMouse, setCameraPosition, setCameraNearFar, setCameraAspect, computePixelToRayMatrix, rotateCameraBy } from './helpers/CameraHelpers.js';
import { fastBVHExampleScene } from './helpers/GeometryUtils.js';
import { createPlaceholderImage, createPlaceholderTexture } from './helpers/ImageHelpers.js';
import { flattenMaterial, flattenMaterialArray, MATERIAL_SIZE } from './helpers/MaterialUtils.js';
import { FastParallelBVH } from './FastParallelBVH.js';

//================================//
const normalUniformDataSize = (16 * 2) * 4 + (2 * 4) * 4 + (48 * 3);
const rayTracerUniformDataSize = 224 + 16 * 4 + 16;

//================================//
const RayTracerMode = {
    raytrace: 0,
    BVHVisualization: 1,
    normal: 2,
    distance: 3,
    rayDirections: 4
};

//================================//
async function loadShaders()
{
    const load = (path) => fetch(path).then(r => { if (!r.ok) throw new Error(`Failed to load shader: ${path}`); return r.text(); });
    return {
        rayTraceVert: await load('./MainArchitectureShaders/raytrace_vert.wgsl'),
        rayTraceFrag: await load('./MainArchitectureShaders/raytrace_frag.wgsl'),
        rasterVert:   await load('./MainArchitectureShaders/raster_vert.wgsl'),
        rasterFrag:   await load('./MainArchitectureShaders/raster_frag.wgsl'),
        bvhVert:      await load('./MainArchitectureShaders/bvh_vert.wgsl'),
        bvhFrag:      await load('./MainArchitectureShaders/bvh_frag.wgsl'),
    };
}

//================================//
export async function startup(canvas)
{
    const renderer = new RayTracer();
    await renderer.initialize(canvas);
    return renderer;
}

//================================//
class RayTracer
{
    constructor()
    {
        this.device = null;
        this.canvas = null;
        this.context = null;
        this.presentationFormat = null;
        this.timestampQuerySet = null;

        this.animationFrameId = null;
        this.resizeObserver = null;
        this.infoElement = getInfoElement();

        this.keysPressed = new Set();
        this.isMouseDown = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.camera = createCamera(1.0);
        this.lights = [];
        this.a_c = 1.0;
        this.a_l = 0.09;
        this.a_q = 0.0032;
        this.NO = {};
        this.RO = {};

        this.useRaytracing = true;
        this.rayTracerMode = RayTracerMode.raytrace;
        this.numBounces = 3;
        this.numSpheres = 5;
        this.materials = [];
        this.perMeshData = [];
        this.sphereCenters = [];
        this.activeContextMenu = null;
        this.seed = 0;
        this.withPlane = true;

        this.showBVH = false;
        this.animateFlag = true;
        this.bvhDepth = Infinity;
        this.minMaxBoundsText = '';

        this.fastBVH = null;
        this._shaders = null;

        setCameraPosition(this.camera, 0, 100, -200);
        rotateCameraBy(this.camera, 0, -0.5);
        setCameraNearFar(this.camera, 0.1, 2000);
        this.camera.moveSpeed = 5.0;
        this.camera.rotateSpeed = 0.02;

        this.lights.push({
            position:  new Float32Array([0, 100, 0]),
            intensity: 200.0,
            direction: new Float32Array([0, -1, 0]),
            coneAngle: Math.PI / 2,
            color:     new Float32Array([0.1, 0.1, 0.85]),
            enabled:   true
        });
        this.lights.push({
            position:  new Float32Array([100, 100, 0]),
            intensity: 1000.0,
            direction: new Float32Array([-1, -3, 0]),
            coneAngle: Math.PI / 5,
            color:     new Float32Array([0.1, 0.85, 0.1]),
            enabled:   true
        });
        this.lights.push({
            position:  new Float32Array([-100, 100, 0]),
            intensity: 1000.0,
            direction: new Float32Array([1, -3, 0]),
            coneAngle: Math.PI / 5,
            color:     new Float32Array([0.85, 0.1, 0.1]),
            enabled:   true
        });
    }

    //================================//
    async initialize(canvas)
    {
        this.canvas = canvas;

        this._shaders = await loadShaders();
        this.fastBVH = await FastParallelBVH.create();

        this.device = await RequestWebGPUDevice(['timestamp-query']);
        if (!this.device) {
            console.error('Was not able to acquire a WebGPU device.');
            return;
        }

        this.context = canvas.getContext('webgpu');
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        if (!this.context) {
            console.error('WebGPU context is not available.');
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
        if (!this.device || !this._shaders) return;

        this.RO.shaderModule = CreateShaderModule(this.device, this._shaders.rayTraceVert, this._shaders.rayTraceFrag, 'Ray Trace Shader Module');
        this.NO.shaderModule = CreateShaderModule(this.device, this._shaders.rasterVert, this._shaders.rasterFrag, 'Normal Shader Module');
        this.NO.bvhShaderModule = CreateShaderModule(this.device, this._shaders.bvhVert, this._shaders.bvhFrag, 'BVH Draw Shader Module');
    }

    //================================//
    initializePipelines()
    {
        if (!this.device || !this.presentationFormat) return;

        this.RO.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Ray Trace Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // world vertices
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // world normals
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // UVs
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // indices
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // bvhNodes
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // materialsPerTriangle
            ],
        });

        this.RO.materialBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Ray Trace Material Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array', multisampled: false } },
            ]
        });

        this.RO.pipelineLayout = this.device.createPipelineLayout({
            label: 'Ray Trace Pipeline Layout',
            bindGroupLayouts: [this.RO.bindGroupLayout, this.RO.materialBindGroupLayout],
        });

        if (this.RO.shaderModule !== null) {
            this.RO.pipeline = this.device.createRenderPipeline({
                label: 'Ray Trace Pipeline',
                layout: this.RO.pipelineLayout,
                vertex: {
                    module: this.RO.shaderModule.vertex,
                    entryPoint: 'vs',
                },
                fragment: {
                    module: this.RO.shaderModule.fragment,
                    entryPoint: 'fs',
                    targets: [{ format: this.presentationFormat }],
                }
            });
        }

        this.NO.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Normal Bind Group Layout',
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }]
        });

        this.NO.materialUniformBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Material Uniform Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
                { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // Model matrix
                { binding: 7, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // Normal matrix
            ],
        });

        this.NO.pipelineLayout = this.device.createPipelineLayout({
            label: 'Normal Pipeline Layout',
            bindGroupLayouts: [this.NO.bindGroupLayout, this.NO.materialUniformBindGroupLayout],
        });

        this.NO.bvhDrawPipelineLayout = this.device.createPipelineLayout({
            label: 'BVH Draw Pipeline Layout',
            bindGroupLayouts: [this.NO.bindGroupLayout, this.NO.materialUniformBindGroupLayout],
        });

        this.NO.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        if (this.NO.shaderModule !== null) {
            this.NO.pipeline = this.device.createRenderPipeline({
                label: 'Normal Pipeline',
                layout: this.NO.pipelineLayout,
                vertex: {
                    module: this.NO.shaderModule.vertex,
                    entryPoint: 'vs',
                    buffers: [
                        { arrayStride: 3 * 4, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
                        { arrayStride: 3 * 4, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
                        { arrayStride: 2 * 4, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x2' }] },
                    ]
                },
                fragment: {
                    module: this.NO.shaderModule.fragment,
                    entryPoint: 'fs',
                    targets: [{ format: this.presentationFormat }],
                },
                primitive: { topology: 'triangle-list', cullMode: 'back' },
                depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
            });

            this.NO.bvhDrawPipeline = this.device.createRenderPipeline({
                label: 'BVH Draw Pipeline',
                layout: this.NO.bvhDrawPipelineLayout,
                vertex: {
                    module: this.NO.bvhShaderModule.vertex,
                    entryPoint: 'vsBVH',
                    buffers: [{ arrayStride: 3 * 4, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }]
                },
                fragment: {
                    module: this.NO.bvhShaderModule.fragment,
                    entryPoint: 'fsBVH',
                    targets: [{ format: this.presentationFormat }],
                },
                primitive: { topology: 'line-list' },
                depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
            });
        }

        this.timestampQuerySet = CreateTimestampQuerySet(this.device, 10);

        this.NO.sampler = this.device.createSampler({
            label: 'Normal Objects Sampler',
            magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
            addressModeU: 'repeat', addressModeV: 'repeat',
        });

        this.RO.sampler = this.device.createSampler({
            label: 'Ray Tracer Sampler',
            magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
            addressModeU: 'repeat', addressModeV: 'repeat',
        });
    }

    //================================//
    async initializeBuffers()
    {
        if (!this.device) return;

        const placeholderTexture = createPlaceholderTexture(this.device, 1024, 32);

        const meshMaterials = this.materials.length > 0 ? this.materials : [];

        const sceneData = await fastBVHExampleScene(meshMaterials, this.seed, this.numSpheres, this.withPlane);
        const {
            worldPositionData,
            worldNormalData,
            worldUVData,
            worldIndexData,
            perTriangleMaterialIndices,
            perMeshWorldPositionOffsets,
            materials,
            perMeshData,
        } = sceneData;

        this.materials = materials;
        this.perMeshData = perMeshData;
        this.sphereCenters = sceneData.sphereCenters;
        this.sphereMeshOffset = perMeshData.length - sceneData.sphereCenters.length;
        this.RO.perMeshWorldPositionOffsets = perMeshWorldPositionOffsets;

        const numMeshes = perMeshData.length;

        this.NO.materialUniforms = [];
        this.NO.materialBindGroups = [];
        this.NO.positionBuffers = [];
        this.NO.normalBuffers = [];
        this.NO.uvBuffers = [];
        this.NO.indexBuffers = [];
        this.NO.meshesModelMatrixBuffers = [];
        this.NO.meshesNormalMatrixBuffers = [];

        const identityMat4 = new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]);
        const identityNormalMat = new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
        ]);

        for (let i = 0; i < numMeshes; i++) {
            const mesh = perMeshData[i];
            const mat = materials[i];

            const modelMatBuf = this.device.createBuffer({
                label: `Mesh Model Matrix Buffer ${i}`,
                size: 16 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(modelMatBuf, 0, identityMat4);
            this.NO.meshesModelMatrixBuffers.push(modelMatBuf);

            const normMatBuf = this.device.createBuffer({
                label: `Mesh Normal Matrix Buffer ${i}`,
                size: 12 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(normMatBuf, 0, identityNormalMat);
            this.NO.meshesNormalMatrixBuffers.push(normMatBuf);

            const matUniform = this.device.createBuffer({
                label: `Material Uniform Buffer ${i}`,
                size: MATERIAL_SIZE * 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(matUniform, 0, flattenMaterial(mat));
            this.NO.materialUniforms.push(matUniform);

            this.NO.materialBindGroups.push(this.device.createBindGroup({
                label: `Material Bind Group ${i}`,
                layout: this.NO.materialUniformBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: matUniform } },
                    { binding: 1, resource: this.NO.sampler },
                    { binding: 2, resource: mat.albedoGPUTexture    ? mat.albedoGPUTexture.createView()    : placeholderTexture.createView() },
                    { binding: 3, resource: mat.metalnessGPUTexture ? mat.metalnessGPUTexture.createView() : placeholderTexture.createView() },
                    { binding: 4, resource: mat.roughnessGPUTexture ? mat.roughnessGPUTexture.createView() : placeholderTexture.createView() },
                    { binding: 5, resource: mat.normalGPUTexture    ? mat.normalGPUTexture.createView()    : placeholderTexture.createView() },
                    { binding: 6, resource: { buffer: modelMatBuf } },
                    { binding: 7, resource: { buffer: normMatBuf } },
                ],
            }));

            const posBuf = this.device.createBuffer({
                label: `Position Buffer ${i}`,
                size: mesh.positions.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(posBuf, 0, mesh.positions);
            this.NO.positionBuffers.push(posBuf);

            const normBuf = this.device.createBuffer({
                label: `Normal Buffer ${i}`,
                size: mesh.normals.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(normBuf, 0, mesh.normals);
            this.NO.normalBuffers.push(normBuf);

            const uvBuf = this.device.createBuffer({
                label: `UV Buffer ${i}`,
                size: mesh.uvs.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(uvBuf, 0, mesh.uvs);
            this.NO.uvBuffers.push(uvBuf);

            const idxBuf = this.device.createBuffer({
                label: `Index Buffer ${i}`,
                size: mesh.indices.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(idxBuf, 0, mesh.indices);
            this.NO.indexBuffers.push(idxBuf);
        }

        this.NO.uniformBuffer = this.device.createBuffer({
            label: 'Normal Uniform Buffer',
            size: normalUniformDataSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.NO.bindGroup = this.device.createBindGroup({
            label: 'Normal Bind Group',
            layout: this.NO.bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.NO.uniformBuffer } }],
        });

        this.NO.bvhDebugMaterialBuffer = this.device.createBuffer({
            label: 'BVH Debug Material Buffer',
            size: MATERIAL_SIZE * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.NO.bvhDebugMaterialBuffer, 0, new Float32Array(MATERIAL_SIZE));

        this.NO.bvhDebugModelMatrixBuffer = this.device.createBuffer({
            label: 'BVH Debug Model Matrix Buffer',
            size: 16 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.NO.bvhDebugModelMatrixBuffer, 0, identityMat4);

        this.NO.bvhDebugNormalMatrixBuffer = this.device.createBuffer({
            label: 'BVH Debug Normal Matrix Buffer',
            size: 12 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.NO.bvhDebugNormalMatrixBuffer, 0, identityNormalMat);

        this.NO.bvhDebugBindGroup = this.device.createBindGroup({
            label: 'BVH Debug Bind Group',
            layout: this.NO.materialUniformBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.NO.bvhDebugMaterialBuffer } },
                { binding: 1, resource: this.NO.sampler },
                { binding: 2, resource: placeholderTexture.createView() },
                { binding: 3, resource: placeholderTexture.createView() },
                { binding: 4, resource: placeholderTexture.createView() },
                { binding: 5, resource: placeholderTexture.createView() },
                { binding: 6, resource: { buffer: this.NO.bvhDebugModelMatrixBuffer } },
                { binding: 7, resource: { buffer: this.NO.bvhDebugNormalMatrixBuffer } },
            ],
        });

        this.NO.bvhLineGeometryBuffers = [this.device.createBuffer({
            label: 'BVH Debug Line Geometry Buffer',
            size: 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        })];
        this.NO.bvhLineCounts = [0];

        this.RO.uniformBuffer = this.device.createBuffer({
            label: 'Ray Tracer Uniform Buffer',
            size: rayTracerUniformDataSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.RO.worldPositionStorageBuffer = this.device.createBuffer({
            label: 'World Position Storage Buffer',
            size: worldPositionData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.worldPositionStorageBuffer, 0, worldPositionData);

        this.RO.worldNormalStorageBuffer = this.device.createBuffer({
            label: 'World Normal Storage Buffer',
            size: worldNormalData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.worldNormalStorageBuffer, 0, worldNormalData);

        this.RO.uvStorageBuffer = this.device.createBuffer({
            label: 'UV Storage Buffer',
            size: worldUVData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.uvStorageBuffer, 0, worldUVData);

        this.RO.indexStorageBuffer = this.device.createBuffer({
            label: 'Index Storage Buffer',
            size: worldIndexData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.indexStorageBuffer, 0, worldIndexData);

        this.RO.perTriangleMaterialIndicesStorageBuffer = this.device.createBuffer({
            label: 'Per-Triangle Material Indices Storage Buffer',
            size: perTriangleMaterialIndices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.perTriangleMaterialIndicesStorageBuffer, 0, perTriangleMaterialIndices);

        const numVertices = worldPositionData.length / 3;
        const numTriangles = worldIndexData.length / 3;

        this.fastBVH.initializeMinMaxPipeline(this.device, this.RO.worldPositionStorageBuffer, numVertices);
        this.fastBVH.initializeMortonPipeline(this.device, this.RO.worldPositionStorageBuffer, this.RO.indexStorageBuffer, numTriangles);
        this.fastBVH.initializeRadixSortPipelines(this.device);
        this.fastBVH.initializePatriciaTreePipeline(this.device);
        this.fastBVH.initializeAABBUpPassPipeline(this.device, this.RO.worldPositionStorageBuffer, this.RO.indexStorageBuffer);
        this.fastBVH.initializeDFSFlatteningPipeline(this.device);
        this.fastBVH.initializeBVHWireframePipeline(this.device);

        this.RO.bindGroup = this.device.createBindGroup({
            label: 'Ray Tracer Bind Group',
            layout: this.RO.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.RO.uniformBuffer } },
                { binding: 1, resource: { buffer: this.RO.worldPositionStorageBuffer } },
                { binding: 2, resource: { buffer: this.RO.worldNormalStorageBuffer } },
                { binding: 3, resource: { buffer: this.RO.uvStorageBuffer } },
                { binding: 4, resource: { buffer: this.RO.indexStorageBuffer } },
                { binding: 5, resource: { buffer: this.fastBVH.getFinalFlattenedBVHBuffer() } },
                { binding: 6, resource: { buffer: this.RO.perTriangleMaterialIndicesStorageBuffer } },
            ],
        });

        const materialData = flattenMaterialArray(materials);
        this.RO.materialBuffer = this.device.createBuffer({
            label: 'Ray Tracer Material Storage Buffer',
            size: materialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.materialBuffer, 0, materialData);

        const numTexturesPerMaterial = 4;
        const commonW = 1024;
        const commonH = 1024;

        const numTexturedMaterials = Math.max(1, materials.filter(m => m.textureIndex >= 0).length);
        this.RO.textureArray = this.device.createTexture({
            label: 'Ray Tracer Material Texture Array',
            size: [commonW, commonH, numTexturesPerMaterial * numTexturedMaterials],
            format: 'rgba8unorm',
            mipLevelCount: 1,
            sampleCount: 1,
            dimension: '2d',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const placeHolderImage = createPlaceholderImage(1024, 32);
        for (let i = 0; i < numTexturedMaterials; i++) {
            for (let typeIndex = 0; typeIndex < numTexturesPerMaterial; typeIndex++) {
                this.device.queue.copyExternalImageToTexture(
                    { source: placeHolderImage },
                    { texture: this.RO.textureArray, origin: [0, 0, i * numTexturesPerMaterial + typeIndex] },
                    [commonW, commonH]
                );
            }
        }

        this.RO.materialBindGroup = this.device.createBindGroup({
            label: 'Ray Tracer Material Bind Group',
            layout: this.RO.materialBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.RO.materialBuffer } },
                { binding: 1, resource: this.RO.sampler },
                { binding: 2, resource: this.RO.textureArray.createView({ dimension: '2d-array' }) },
            ],
        });
    }

    //================================//
    initializeUtils()
    {
        const utilElement = getUtilElement();
        if (!utilElement) return;

        addCheckbox('Debug', this.fastBVH.debug, utilElement, (value) => { this.fastBVH.debug = value; });
        utilElement.appendChild(document.createElement('br'));

        addCheckbox('Use Ray Tracing', this.useRaytracing, utilElement, (value) => { this.useRaytracing = value; });
        utilElement.appendChild(document.createElement('br'));

        addSlider('Number of Bounces', this.numBounces, 0, 20, 1, utilElement, (value) => { this.numBounces = value; });
        utilElement.appendChild(document.createElement('br'));

        addCheckbox('Animate', this.animateFlag, utilElement, (value) => { this.animateFlag = value; });
        utilElement.appendChild(document.createElement('br'));

        this.lights.forEach((_, index) => {
            const callback = (e) => {
                e.preventDefault();
                if (this.activeContextMenu) {
                    this.activeContextMenu.remove();
                    this.activeContextMenu = null;
                }
                const middleOfCanvas = {
                    x: this.canvas.offsetLeft + this.canvas.width - 300,
                    y: this.canvas.offsetTop + this.canvas.height / 2 - 150
                };
                this.activeContextMenu = createLightContextMenu(
                    middleOfCanvas,
                    this.lights[index],
                    `Edit Light ${index + 1}`,
                    (newLight) => { this.lights[index] = newLight; },
                    () => { if (this.activeContextMenu) { this.activeContextMenu.remove(); this.activeContextMenu = null; } }
                );
                document.body.appendChild(this.activeContextMenu);
            };
            utilElement.appendChild(document.createElement('br'));
            addButton(`Edit Light ${index + 1}`, utilElement, callback);
        });

        utilElement.appendChild(document.createElement('br'));
        addCheckbox('Show BVH', this.showBVH, utilElement, (value) => {
            this.showBVH = value;
            this.rayTracerMode = value ? RayTracerMode.BVHVisualization : RayTracerMode.raytrace;
        });
        utilElement.appendChild(document.createElement('br'));
        addSlider('BVH Depth', this.bvhDepth === Infinity ? 32 : this.bvhDepth, 1, 32, 1, utilElement, (value) => {
            this.bvhDepth = value === 32 ? Infinity : value;
        });
        utilElement.appendChild(document.createElement('br'));
        addNumberInput('Random Seed', this.seed, 0, 10 << 20, 1, utilElement, (value) => {
            this.seed = value;
            this.initializeBuffers();
        });
        utilElement.appendChild(document.createElement('br'));
        addSlider('Number of Spheres', this.numSpheres, 1, 99, 1, utilElement, (value) => {
            this.numSpheres = value;
            this.initializeBuffers();
        });
        utilElement.appendChild(document.createElement('br'));
        addCheckbox('Include Plane', this.withPlane, utilElement, (value) => {
            this.withPlane = value;
            this.initializeBuffers();
        });
    }

    //================================//
    initializeInputHandlers()
    {
        if (!this.canvas) return;

        this.onKeyDown = (e) => { this.keysPressed.add(e.key.toLowerCase()); };
        this.onKeyUp = (e) => { this.keysPressed.delete(e.key.toLowerCase()); };
        this.onMouseDown = (e) => {
            this.isMouseDown = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        };
        this.onMouseUp = () => { this.isMouseDown = false; };
        this.onMouseMove = (e) => {
            if (!this.isMouseDown) return;
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            rotateCameraByMouse(this.camera, deltaX * 0.05, -deltaY * 0.05);
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        };

        window.addEventListener('keydown', this.onKeyDown);
        window.addEventListener('keyup', this.onKeyUp);
        this.canvas.addEventListener('mousedown', this.onMouseDown);
        window.addEventListener('mouseup', this.onMouseUp);
        window.addEventListener('mousemove', this.onMouseMove);
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    //================================//
    removeInputHandlers()
    {
        if (this.onKeyDown)   window.removeEventListener('keydown', this.onKeyDown);
        if (this.onKeyUp)     window.removeEventListener('keyup', this.onKeyUp);
        if (this.onMouseUp)   window.removeEventListener('mouseup', this.onMouseUp);
        if (this.onMouseMove) window.removeEventListener('mousemove', this.onMouseMove);
        if (this.canvas && this.onMouseDown) this.canvas.removeEventListener('mousedown', this.onMouseDown);
    }

    //================================//
    handleInput()
    {
        let dx = 0, dy = 0, dz = 0;

        if (this.keysPressed.has('z') || this.keysPressed.has('w')) dz -= this.camera.moveSpeed;
        if (this.keysPressed.has('s'))                               dz += this.camera.moveSpeed;
        if (this.keysPressed.has('q') || this.keysPressed.has('a')) dx -= this.camera.moveSpeed;
        if (this.keysPressed.has('d'))                               dx += this.camera.moveSpeed;
        if (this.keysPressed.has('shift'))                           dy += this.camera.moveSpeed;
        if (this.keysPressed.has('alt'))                             dy -= this.camera.moveSpeed;

        if (dx !== 0 || dy !== 0 || dz !== 0)
            moveCameraLocal(this.camera, -dz, dx, dy);

        if (this.keysPressed.has('arrowleft'))  rotateCameraByMouse(this.camera, -1, 0);
        if (this.keysPressed.has('arrowright')) rotateCameraByMouse(this.camera, 1, 0);
        if (this.keysPressed.has('arrowup'))    rotateCameraByMouse(this.camera, 0, 1);
        if (this.keysPressed.has('arrowdown'))  rotateCameraByMouse(this.camera, 0, -1);
    }

    //================================//
    updateUniforms()
    {
        if (!this.device) return;

        if (this.useRaytracing) 
        {
            const data = new ArrayBuffer(rayTracerUniformDataSize);
            const floatView = new Float32Array(data);
            const uintView = new Uint32Array(data);

            floatView.set(computePixelToRayMatrix(this.camera), 0);
            floatView.set(this.camera.position, 16);
            uintView[19] = this.rayTracerMode;
            floatView[20] = this.a_c;
            floatView[21] = this.a_l;
            floatView[22] = this.a_q;
            floatView[23] = this.bvhDepth === Infinity ? 9999 : this.bvhDepth;

            uintView[24] = this.numBounces;
            uintView[25] = this.fastBVH.getFinalFlattenedBVHNodeCount();
            uintView[26] = 0;
            uintView[27] = 0;

            for (let i = 0; i < 3; i++) 
            {
                if (i >= this.lights.length) break;
                const light = this.lights[i];
                const baseIndex = 28 + i * 12;
                floatView.set(light.position, baseIndex);
                floatView[baseIndex + 3] = light.intensity;
                floatView.set(light.direction, baseIndex + 4);
                floatView[baseIndex + 7] = light.coneAngle;
                floatView.set(light.color, baseIndex + 8);
                floatView[baseIndex + 11] = light.enabled ? 1.0 : 0.0;
            }
            this.device.queue.writeBuffer(this.RO.uniformBuffer, 0, data);
        } 
        else 
        {
            const data = new ArrayBuffer(normalUniformDataSize);
            const floatView = new Float32Array(data);
            floatView.set(this.camera.viewMatrix, 0);
            floatView.set(this.camera.projectionMatrix, 16);
            floatView.set(this.camera.position, 32);
            floatView[36] = this.a_c;
            floatView[37] = this.a_l;
            floatView[38] = this.a_q;
            floatView[39] = 0.0;

            for (let i = 0; i < 3; i++) {
                if (i >= this.lights.length) break;
                const light = this.lights[i];
                const baseIndex = 40 + i * 12;
                floatView.set(light.position, baseIndex);
                floatView[baseIndex + 3] = light.intensity;
                floatView.set(light.direction, baseIndex + 4);
                floatView[baseIndex + 7] = light.coneAngle;
                floatView.set(light.color, baseIndex + 8);
                floatView[baseIndex + 11] = light.enabled ? 1.0 : 0.0;
            }
            this.device.queue.writeBuffer(this.NO.uniformBuffer, 0, data);
        }
    }

    //================================//
    animate()
    {
        if (!this.device || !this.sphereCenters || !this.perMeshData) return;

        const time = performance.now() * 0.001;
        const maxAnimated = Math.min(10, this.sphereCenters.length);

        let s = (this.seed + 777) | 0;
        const random = () => {
            s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        const randomRange = (min, max) => random() * (max - min) + min;

        for (let i = 0; i < maxAnimated; i++)
        {
            const { cx: baseCX, cy: baseCY, cz: baseCZ } = this.sphereCenters[i];
            const orbitRadius = randomRange(20, 80);
            const speed       = randomRange(0.3, 1.5);
            const phaseOffset = randomRange(0, Math.PI * 2);
            const pattern     = random();

            const t = time * speed + phaseOffset;
            let nx, ny, nz;

            if (pattern < 0.25) {
                nx = baseCX + orbitRadius * Math.sin(t);
                nz = baseCZ + orbitRadius * Math.sin(t) * Math.cos(t);
                ny = baseCY;
            } else if (pattern < 0.5) {
                const ecc = randomRange(0.3, 0.8);
                nx = baseCX + orbitRadius * Math.cos(t);
                nz = baseCZ + orbitRadius * ecc * Math.sin(t);
                ny = baseCY + Math.abs(Math.sin(t * 2.0)) * 15.0;
            } else if (pattern < 0.75) {
                const axis = randomRange(0, Math.PI * 2);
                const dist = Math.sin(t) * orbitRadius;
                nx = baseCX + Math.cos(axis) * dist;
                nz = baseCZ + Math.sin(axis) * dist;
                ny = baseCY + Math.abs(Math.sin(t * 1.5)) * 8.0;
            } else {
                const spiralR = orbitRadius * (0.5 + 0.5 * Math.sin(t * 0.3));
                nx = baseCX + spiralR * Math.cos(t);
                nz = baseCZ + spiralR * Math.sin(t);
                ny = baseCY;
            }

            const dx = nx - baseCX;
            const dy = ny - baseCY;
            const dz = nz - baseCZ;

            const meshIndex = i + this.sphereMeshOffset;
            const basePositions = this.perMeshData[meshIndex].positions;
            const numVerts = basePositions.length / 3;
            const translated = new Float32Array(basePositions.length);
            for (let v = 0; v < numVerts; v++) {
                translated[v * 3 + 0] = basePositions[v * 3 + 0] + dx;
                translated[v * 3 + 1] = basePositions[v * 3 + 1] + dy;
                translated[v * 3 + 2] = basePositions[v * 3 + 2] + dz;
            }

            const byteOffset = this.RO.perMeshWorldPositionOffsets[meshIndex];
            this.device.queue.writeBuffer(this.RO.worldPositionStorageBuffer, byteOffset, translated);
            this.device.queue.writeBuffer(this.NO.positionBuffers[meshIndex], 0, translated);
        }
    }

    //================================//
    mainLoop()
    {
        if (!this.device || !this.canvas) return;

        let then = 0;
        let gpuTime = 0;
        let gpuComputeTime = 0;

        const render = async (now) => {
            if (!this.canvas || !this.device || !this.context) return;

            const dt = now - then;
            then = now;
            const startTime = performance.now();

            this.handleInput();

            if (this.animateFlag)
                this.animate();

            this.updateUniforms();

            // For some reason, if we did not dispatch the BVH build in isolation,
            // it would hang the GPU and break the application.
            // This is the workaround I foung to make it work
            const bvhEncoder = this.device.createCommandEncoder({ label: 'BVH Build Encoder' });

            this.fastBVH.clearAtomicCounters(bvhEncoder);
            if (this.showBVH && !this.useRaytracing)
                this.fastBVH.clearBVHWireframe(bvhEncoder);

            const computePass = bvhEncoder.beginComputePass({
                label: 'FastBVH Compute Pass',
                ...(this.timestampQuerySet != null && {
                    timestampWrites: {
                        querySet: this.timestampQuerySet.querySet,
                        beginningOfPassWriteIndex: 2,
                        endOfPassWriteIndex: 3,
                    }
                })
            });
            this.fastBVH.dispatch(computePass);

            if (this.showBVH && !this.useRaytracing)
                this.fastBVH.dispatchBVHWireframePass(computePass, this.bvhDepth);
            computePass.end();

            if (this.fastBVH.minMaxReadbackBuffer?.mapState === 'unmapped' && this.fastBVH.debug)
                this.fastBVH.copyResultForReadback(bvhEncoder);

            this.device.queue.submit([bvhEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            const textureView = this.context.getCurrentTexture().createView();
            const depthStencilAttachment = !this.useRaytracing ? {
                view: this.NO.depthTexture.createView(),
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1.0,
            } : undefined;

            const renderPassDescriptor = {
                label: 'Main Render Pass',
                colorAttachments: [{
                    view: textureView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 }
                }],
                depthStencilAttachment,
                ...(this.timestampQuerySet != null && {
                    timestampWrites: {
                        querySet: this.timestampQuerySet.querySet,
                        beginningOfPassWriteIndex: 0,
                        endOfPassWriteIndex: 1,
                    }
                }),
            };

            const encoder = this.device.createCommandEncoder({ label: 'Main Encoder' });

            const pass = encoder.beginRenderPass(renderPassDescriptor);

            if (this.useRaytracing) {
                pass.setPipeline(this.RO.pipeline);
                pass.setBindGroup(0, this.RO.bindGroup);
                pass.setBindGroup(1, this.RO.materialBindGroup);
                pass.draw(6);
            } else {
                pass.setPipeline(this.NO.pipeline);
                pass.setBindGroup(0, this.NO.bindGroup);
                for (let i = 0; i < this.perMeshData.length; i++) {
                    pass.setBindGroup(1, this.NO.materialBindGroups[i]);
                    pass.setVertexBuffer(0, this.NO.positionBuffers[i]);
                    pass.setVertexBuffer(1, this.NO.normalBuffers[i]);
                    pass.setVertexBuffer(2, this.NO.uvBuffers[i]);
                    pass.setIndexBuffer(this.NO.indexBuffers[i], 'uint16');
                    pass.drawIndexed(this.NO.indexBuffers[i].size / 2, 1, 0, 0, 0);
                }

                if (this.showBVH) {
                    pass.setPipeline(this.NO.bvhDrawPipeline);
                    pass.setBindGroup(0, this.NO.bindGroup);
                    pass.setBindGroup(1, this.NO.bvhDebugBindGroup);
                    pass.setVertexBuffer(0, this.fastBVH.bvhWireframeVertexBuffer);
                    pass.drawIndirect(this.fastBVH.bvhWireframeArgsBuffer, 0);
                }
            }
            pass.end();

            if (this.timestampQuerySet != null) {
                encoder.resolveQuerySet(
                    this.timestampQuerySet.querySet,
                    0, this.timestampQuerySet.querySet.count,
                    this.timestampQuerySet.resolveBuffer, 0
                );
                if (this.timestampQuerySet.resultBuffer.mapState === 'unmapped')
                    encoder.copyBufferToBuffer(
                        this.timestampQuerySet.resolveBuffer, 0,
                        this.timestampQuerySet.resultBuffer, 0,
                        this.timestampQuerySet.resultBuffer.size
                    );
            }

            this.device.queue.submit([encoder.finish()]);

            if (this.fastBVH.minMaxReadbackBuffer?.mapState === 'unmapped' && this.fastBVH.debug) {
                this.fastBVH.minMaxReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
                    const d = new Float32Array(this.fastBVH.minMaxReadbackBuffer.getMappedRange());
                    this.minMaxBoundsText = `Min: (${d[0].toFixed(1)}, ${d[1].toFixed(1)}, ${d[2].toFixed(1)}) Max: (${d[3].toFixed(1)}, ${d[4].toFixed(1)}, ${d[5].toFixed(1)})`;
                    this.fastBVH.minMaxReadbackBuffer.unmap();
                });
            }

            if (this.timestampQuerySet != null && this.timestampQuerySet.resultBuffer.mapState === 'unmapped') {
                this.timestampQuerySet.resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
                    const times = new BigUint64Array(this.timestampQuerySet.resultBuffer.getMappedRange());
                    gpuTime = Number(times[1] - times[0]);
                    gpuComputeTime = Number(times[3] - times[2]);
                    this.timestampQuerySet.resultBuffer.unmap();
                });
            }

            const jsTime = performance.now() - startTime;
            if (this.infoElement && this.device) {
                this.infoElement.textContent =
                    `FPS: ${(1000 / dt).toFixed(1)}\n` +
                    `RGPU: ${(gpuTime / 1e6).toFixed(2)} ms\n` +
                    `CGPU: ${(gpuComputeTime / 1e6).toFixed(2)} ms\n` +
                    `Triangles: ${this.fastBVH.numTriangles}\n` +
                    (this.fastBVH.debug ? this.minMaxBoundsText : '');
            }
            addProfilerFrameTime(1000 / dt);

            this.animationFrameId = requestAnimationFrame(render);
        };

        this.animationFrameId = requestAnimationFrame(render);

        this.resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const width = entry.contentBoxSize[0].inlineSize;
                const height = entry.contentBoxSize[0].blockSize;

                if (this.canvas && this.device) {
                    this.canvas.width = Math.max(1, Math.min(width, this.device.limits.maxTextureDimension2D));
                    this.canvas.height = Math.max(1, Math.min(height, this.device.limits.maxTextureDimension2D));
                    setCameraAspect(this.camera, this.canvas.width / this.canvas.height);

                    if (this.NO.depthTexture) {
                        this.NO.depthTexture.destroy();
                        this.NO.depthTexture = this.device.createTexture({
                            size: [this.canvas.width, this.canvas.height],
                            format: 'depth24plus',
                            usage: GPUTextureUsage.RENDER_ATTACHMENT,
                        });
                    }
                }
            }
        });
        this.resizeObserver.observe(this.canvas);
    }

    //================================//
    async smallCleanup()
    {
        this.removeInputHandlers();
        cleanUtilElement();

        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.resizeObserver && this.canvas) {
            this.resizeObserver.unobserve(this.canvas);
            this.resizeObserver = null;
        }
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
}
