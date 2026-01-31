
import * as glm from 'gl-matrix';

//================================//
import rayTraceVertWGSL from './shader_vert.wgsl?raw';
import rayTraceFragWGSL from './shader_frag.wgsl?raw';

import normalVertWgsl from './normal_vert.wgsl?raw';
import normalFragWgsl from './normal_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice, CreateShaderModule } from '@src/helpers/WebGPUutils';
import type { PipelineResources, TimestampQuerySet } from '@src/helpers/WebGPUutils';
import { createLightContextMenu, createMaterialContextMenu, getInfoElement, getUtilElement, type SpotLight } from '@src/helpers/Others';
import { createCamera, moveCameraLocal, rotateCameraByMouse, setCameraPosition, setCameraNearFar, setCameraAspect, computePixelToRayMatrix, rotateCameraBy, cameraPointToRay, rayIntersectsSphere } from '@src/helpers/CameraHelpers';
import { createCornellBox2, type Material, type PerMaterialTopologyInformation, type Transform } from '@src/helpers/GeometryUtils';

//================================//
export async function startup_10(canvas: HTMLCanvasElement)
{
    const renderer = new RayTracer();
    await renderer.initialize(canvas);
    
    return renderer;
}

//================================//
const normalUniformDataSize = (16 * 3) * 4 + 16 * 4 +(48 * 3); // Three matrices + three spotlights max + four floats
const materialUniformDataSize = (4 * 4); // 16 bytes
const rayTracerUniformDataSize = 224 + 16*4; // = 224 bytes + four floats

interface normalObjects extends PipelineResources
{
    uniformBuffer: GPUBuffer;

    perMaterialTopologies: PerMaterialTopologyInformation;

    materialUniforms: GPUBuffer[];  
    materialBindGroups: GPUBindGroup[];
    materialUniformBindGroupLayout: GPUBindGroupLayout;

    positionBuffers: GPUBuffer[];
    normalBuffers: GPUBuffer[];
    uvBuffers: GPUBuffer[];
    indexBuffers: GPUBuffer[];

    depthTexture: GPUTexture;
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
};

enum RayTracingMode
{
    NormalShading = 0,
    Normals = 1,
    Distance = 2,
    rayDirections = 3,
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
    private useRaytracing: boolean = true;
    private useRaytracingCheckBox: HTMLInputElement | null = null;

    private rayTracingModeSelect: HTMLSelectElement | null = null;
    private rayTracingMode: RayTracingMode = RayTracingMode.NormalShading;
    
    private sphereResolution: number = 8;
    private sphereResolutionSlider: HTMLInputElement | null = null;

    //================================//
    private spheresInfo: any;

    //================================//
    private activeContextMenu: HTMLDivElement | null = null;

    //================================//
    constructor () 
    {
        setCameraPosition(this.camera, 278, 500, -700);
        rotateCameraBy(this.camera, 0, -0.3);
        setCameraNearFar(this.camera, 0.1, 2000); // Increase far plane to see entire Cornell box
        this.camera.moveSpeed = 20.0;
        this.camera.rotateSpeed = 0.05;
        this.device = null;
        this.normalObjects = {} as normalObjects;
        this.rayTracerObjects = {} as rayTracerObjects;

        const light1 = {
            position: glm.vec3.fromValues(276.0, 450.0, 1.0), // Max depth is 559, Max X is 552.
            intensity: 1000.0,
            direction: glm.vec3.fromValues(0, -1, 0),
            coneAngle: Math.PI / 4,
            color: glm.vec3.fromValues(0.9, 0.9, 1.0),
            enabled: true
        };
        this.lights.push(light1);
    }

    //================================//
    initializeUtils()
    {
        const utilElement = getUtilElement();
        if (!utilElement) return;

        this.useRaytracingCheckBox = document.createElement('input');
        this.useRaytracingCheckBox.type = 'checkbox';
        this.useRaytracingCheckBox.checked = this.useRaytracing;
        this.useRaytracingCheckBox.id = 'useRaytracingCheckbox';
        this.useRaytracingCheckBox.tabIndex = -1;
        this.useRaytracingCheckBox.addEventListener('change', () => {
            this.useRaytracing = this.useRaytracingCheckBox!.checked;
        });
        const useRaytracingLabel = document.createElement('label');
        useRaytracingLabel.htmlFor = 'useRaytracingCheckbox';
        useRaytracingLabel.textContent = ' Use Raytracing';

        utilElement.appendChild(this.useRaytracingCheckBox);
        utilElement.appendChild(useRaytracingLabel);

        // UNDER THEM a SELECT FOR RAY TRACING MODE
        this.rayTracingModeSelect = document.createElement('select');
        this.rayTracingModeSelect.style.color = 'black';
        this.rayTracingModeSelect.tabIndex = -1;
        const modes = ['Normal Shading', 'Normals', 'Distance', 'Ray Directions'];
        modes.forEach((mode, index) => {
            const option = document.createElement('option');
            option.value = index.toString();
            option.text = mode;
            this.rayTracingModeSelect!.appendChild(option);
        });
        this.rayTracingModeSelect.value = this.rayTracingMode.toString();
        this.rayTracingModeSelect.addEventListener('change', () => {
            this.rayTracingMode = parseInt(this.rayTracingModeSelect!.value) as RayTracingMode;
        });

        utilElement.appendChild(document.createElement('br'));
        utilElement.appendChild(this.rayTracingModeSelect);

        // Resolution slider
        this.sphereResolutionSlider = document.createElement('input');
        this.sphereResolutionSlider.type = 'range';
        this.sphereResolutionSlider.min = '8';
        this.sphereResolutionSlider.max = '64';
        this.sphereResolutionSlider.step = '1';
        this.sphereResolutionSlider.value = this.sphereResolution.toString();
        this.sphereResolutionSlider.tabIndex = -1;
        this.sphereResolutionSlider.addEventListener('input', () => {
            this.sphereResolution = parseInt(this.sphereResolutionSlider!.value);
            this.startRendering();
        });
        const resolutionLabel = document.createElement('label');
        resolutionLabel.htmlFor = 'sphereResolutionSlider';
        resolutionLabel.textContent = ' Sphere Resolution';
        utilElement.appendChild(document.createElement('br'));
        utilElement.appendChild(this.sphereResolutionSlider);
        utilElement.appendChild(resolutionLabel);

        // Lights options
        // Three buttons, that spawn a context menu to edit each light

        this.lights.forEach((_, index) =>
        {
            const lightButton = document.createElement('button');
            lightButton.textContent = `Edit Light ${index + 1}`;
            lightButton.tabIndex = -1;
            lightButton.addEventListener('click', (e) => {
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
                    (newLight) => { this.lights[index] = newLight; this.activeContextMenu?.remove(); this.activeContextMenu = null; },
                    () => { this.activeContextMenu?.remove(); this.activeContextMenu = null; }
                );
                document.body.appendChild(this.activeContextMenu);
            });
            utilElement.appendChild(document.createElement('br'));
            utilElement.appendChild(lightButton);
        });
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
            entries: [{ // materials
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: "read-only-storage" },
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
            }]
        });

        this.normalObjects.pipelineLayout = this.device.createPipelineLayout({
            label: 'Normal Pipeline Layout',
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

        // Cached sphere materials?
        const sphereMaterials = this.spheresInfo?.sphereMaterials || [];

        const info: PerMaterialTopologyInformation = createCornellBox2(sphereMaterials, this.sphereResolution);
        this.normalObjects.perMaterialTopologies = info;
        this.spheresInfo = info.additionalInfo;
        const numMaterials = info.materials.length;

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
                size: materialUniformDataSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            }));
            const materialData = new Float32Array(info.materials[matNum].albedo);
            this.device.queue.writeBuffer(this.normalObjects.materialUniforms[matNum], 0, materialData as BufferSource);

            this.normalObjects.materialBindGroups.push(this.device.createBindGroup({
                label: 'Material Bind Group ' + matNum,
                layout: this.normalObjects.materialUniformBindGroupLayout,
                entries: [{
                    binding: 0,
                    resource: { buffer: this.normalObjects.materialUniforms[matNum] },
                }],
            }));

            this.normalObjects.positionBuffers.push(this.device.createBuffer({
                label: 'Normal Position Buffer ' + matNum,
                size: info.pmTopologies[matNum].vertexData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.positionBuffers[matNum], 0, info.pmTopologies[matNum].vertexData as BufferSource);

            this.normalObjects.indexBuffers.push(this.device.createBuffer({
                label: 'Normal Index Buffer ' + matNum,
                size: info.pmTopologies[matNum].indexData.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.indexBuffers[matNum], 0, info.pmTopologies[matNum].indexData as BufferSource);

            this.normalObjects.normalBuffers.push(this.device.createBuffer({
                label: 'Normal Normal Buffer ' + matNum,
                size: info.pmTopologies[matNum].normalData!.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.normalBuffers[matNum], 0, info.pmTopologies[matNum].normalData as BufferSource);

            this.normalObjects.uvBuffers.push(this.device.createBuffer({
                label: 'Normal UV Buffer ' + matNum,
                size: info.pmTopologies[matNum].uvData!.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.normalObjects.uvBuffers[matNum], 0, info.pmTopologies[matNum].uvData as BufferSource);
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

        // Ray Tracer Objects Buffers

        const flattenedPositions: number[] = [];
        const flattenedNormals: number[] = [];
        const flattenedUVs: number[] = [];
        const flattenedIndices: number[] = [];
        const flattenedMaterialIndices: number[] = [];
        let indexOffset = 0;
        for (let matNum = 0; matNum < numMaterials; matNum++)
        {
            let topo = info.pmTopologies[matNum];
            flattenedPositions.push(...topo.vertexData);
            flattenedNormals.push(...topo.normalData);
            flattenedUVs.push(...topo.uvData);

            for (let index of topo.indexData)
            {
                flattenedIndices.push(index + indexOffset);
            }
            indexOffset += topo.vertexData.length / 3;
            
            for (let i = 0; i < topo.indexData.length / 3; i++)
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
        const flattenedMaterials: number[] = [];
        for (let mat of this.normalObjects.perMaterialTopologies.materials)
        {
            flattenedMaterials.push(...mat.albedo);
        }
        const materialData = new Float32Array(flattenedMaterials);

        this.rayTracerObjects.materialBuffer = this.device.createBuffer({
            label: 'Ray Tracer Material Storage Buffer',
            size: materialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.rayTracerObjects.materialBuffer, 0, materialData as BufferSource);

        this.rayTracerObjects.materialBindGroup = this.device.createBindGroup({
            label: 'Ray Tracer Material Bind Group',
            layout: this.rayTracerObjects.materialBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.rayTracerObjects.materialBuffer },
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

        // Check if we clicked on the context menu first
        if (this.activeContextMenu !== null)
        {
            const rect = this.activeContextMenu.getBoundingClientRect();
            if (e.clientX >= rect.left && e.clientX <= rect.right &&
                e.clientY >= rect.top && e.clientY <= rect.bottom)
            return;
        }

        let sphereIndex = this.rayCastOnSpheres(e.clientX, e.clientY);
        if (sphereIndex !== -1)
        {
            console.log("Clicked on sphere index: ", sphereIndex);
            this.spawnMaterialContextMenu(sphereIndex, e.clientX, e.clientY);
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

        this.initializeBuffers();
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
            floatView.set(this.camera.position, 16); // vec3
            // After matrix (16) and camera (3), we set mode at 19
            uintView[19] =  this.rayTracingMode;
            floatView[20] = this.a_c;
            floatView[21] = this.a_l;
            floatView[22] = this.a_q;
            floatView[23] = 0.0; // pad

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
            floatView[48] = this.a_c;
            floatView[49] = this.a_l;
            floatView[50] = this.a_q;
            floatView[51] = 0.0; // pad

            for (let i = 0; i < 3; i++)
            {
                if (i >= this.lights.length)
                    break;

                const light = this.lights[i];
                const baseIndex = 52 + i * 12;

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
        // NONE FOR NOW
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
            this.animate();
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
                    clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1 }
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

                for (let matNum = 0; matNum < this.normalObjects.perMaterialTopologies.materials.length; matNum++)
                {
                    pass.setBindGroup(1, this.normalObjects.materialBindGroups[matNum]);
                    pass.setVertexBuffer(0, this.normalObjects.positionBuffers[matNum]);
                    pass.setVertexBuffer(1, this.normalObjects.normalBuffers[matNum]);
                    pass.setVertexBuffer(2, this.normalObjects.uvBuffers[matNum]);
                    pass.setIndexBuffer(this.normalObjects.indexBuffers[matNum], 'uint16');

                    pass.drawIndexed(this.normalObjects.indexBuffers[matNum].size / 2, 1, 0, 0, 0);
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
        // Clean handlers
        this.useRaytracingCheckBox?.removeEventListener('change', () => {
            this.useRaytracing = this.useRaytracingCheckBox!.checked;
        });
        this.rayTracingModeSelect?.removeEventListener('change', () => {
            this.rayTracingMode = parseInt(this.rayTracingModeSelect!.value) as RayTracingMode;
        });
        this.sphereResolutionSlider?.removeEventListener('input', () => {
            this.sphereResolution = parseInt(this.sphereResolutionSlider!.value);
            this.startRendering();
        });

        // Clean rest of handlers
        this.removeInputHandlers();

        const utilElement = getUtilElement();
        for (const child of Array.from(utilElement?.children || []))
        {
            child.remove();
        }

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
    changeSphereMaterial(sphereIndex: number, newMaterial: Material)
    {
        if (sphereIndex < 0 || sphereIndex >= (this.spheresInfo?.sphereMaterialIndices.length || 0)) return;

        // mat name
        const matName: string = newMaterial.name;
        const totalMaterialIndex = this.normalObjects.perMaterialTopologies.materials.findIndex(mat => mat.name === matName) || -1;
        if (totalMaterialIndex === -1) return;

        this.spheresInfo!.sphereMaterials[sphereIndex] = newMaterial;
        this.normalObjects.perMaterialTopologies.materials[totalMaterialIndex] = newMaterial;

        // Do not create a new buffer, just update existing one we know the offset in the buffer of the aforementioned material
        const materialBufferIndex = this.spheresInfo!.sphereMaterialIndices[sphereIndex];
        const materialData = new Float32Array(newMaterial.albedo);
        
        // Change it in normal pipeline, that is write on it's uniform buffer
        let buffer = this.normalObjects.materialUniforms[materialBufferIndex];
        this.device!.queue.writeBuffer(buffer, 0, materialData as BufferSource);

        // Change it in the ray tracing pipeline. Carful, here the offset in the buffer is materialBufferIndex * materialFlattenedSize
        const materialFlattenedSize = 3; // three floats for albedo
        const offset = materialBufferIndex * materialFlattenedSize * 4; // in bytes
        this.device!.queue.writeBuffer(this.rayTracerObjects.materialBuffer, offset, materialData as BufferSource);
    }

    //================================//
    // Test to see if we ray cast on one of the three spheres and return its index, else -1
    rayCastOnSpheres(screenX: number, screenY: number): number
    {
        if (this.canvas === null || this.camera === null || this.spheresInfo === null) return -1;

        const transforms: Transform[] = this.spheresInfo.sphereTransforms!;

        // Convert viewport coordinates to canvas-relative coordinates
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = screenX - rect.left;
        const canvasY = screenY - rect.top;

        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const ndcX = (2 * canvasX * scaleX) / this.canvas.width - 1;
        const ndcY = 1 - (2 * canvasY * scaleY) / this.canvas.height;

        const ray: Float32Array = cameraPointToRay(this.camera, ndcX, ndcY);

        // Now check if the ray intersects any of the spheres,
        // we know their world position and radius
        let currentClosestSphereIndex = -1;
        let currentClosestDistance = Number.POSITIVE_INFINITY;

        for (let i = 0; i < transforms.length; i++)
        {
            const transform = transforms[i];
            const sphereCenter = transform.translation;
            const sphereRadius = transform.scale[0]; // uniform scale anyways...

            const distance = rayIntersectsSphere(this.camera.position, ray, sphereCenter, sphereRadius);
            if (distance <= 0) continue;

            if (distance < currentClosestDistance)
            {
                currentClosestDistance = distance;
                currentClosestSphereIndex = i;
            }
        }

        return currentClosestSphereIndex;
    }

    //================================//
    spawnMaterialContextMenu(sphereIndex: number, screenX: number, screenY: number)
    {
        if (this.canvas === null) return;

        this.removeContextMenu();
        const currentMaterial: Material = this.spheresInfo?.sphereMaterials?.[sphereIndex];
        if (!currentMaterial) return;

        this.activeContextMenu = createMaterialContextMenu(
            {x: screenX, y: screenY},
            currentMaterial,
            (newMaterial: Material) => {
                this.changeSphereMaterial(sphereIndex, newMaterial);
                this.removeContextMenu();
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
        if (this.activeContextMenu) {
            this.activeContextMenu.remove();
            this.activeContextMenu = null;
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