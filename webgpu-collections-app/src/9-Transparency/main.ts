//================================//
import VertWGSL from './shader_vert.wgsl?raw';
import FragWGSL from './shader_frag.wgsl?raw';

//================================//
import { RequestWebGPUDevice, CreateShaderModule, mergeFloat32Arrays, mergeUint16Arrays } from '@src/helpers/WebGPUutils';
import type { ShaderModule, TimestampQuerySet } from '@src/helpers/WebGPUutils';
import { cleanUtilElement, getInfoElement, getUtilElement } from '@src/helpers/Others';
import { createQuad, createSphere, type TopologyInformation } from '@src/helpers/GeometryUtils';
import { createCamera, moveCameraLocal, rotateCameraBy, rotateCameraByMouse, setCameraAspect, setCameraNearFar, setCameraPosition } from '@src/helpers/CameraHelpers';
import * as glm from 'gl-matrix';

const uniformDataSize = (16 * 4 * 4) + (4 * 2); // = 224 bytes
//================================//
export async function startup_9(canvas: HTMLCanvasElement)
{
    const renderer = new TransparencyRenderer();
    await renderer.initialize(canvas);
    
    return renderer; // Return renderer for proper cleanup
}

//================================//
interface lightObject
{
    position: Float32Array;
    color: Float32Array;
    intensity: number;
};

enum cullModes
{
    NONE = "none",
    FRONT = "front",
    BACK = "back"
}

//================================//
class TransparencyRenderer
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
    private shaderModule: ShaderModule | null = null;
    private pipelineLayout: GPUPipelineLayout | null = null;
    private renderPipeline: GPURenderPipeline | null = null;
    private bindGroupLayout: GPUBindGroupLayout | null = null;
    private bindGroup: GPUBindGroup | null = null;

    //================================//
    private facesTopologyInformation: TopologyInformation[] = [];
    private spheresTopologyInformation: TopologyInformation[] = [];
    private currentSphereOrders: number[] = [];

    private uniformBuffer: GPUBuffer | null = null;
    private vertexBuffer: GPUBuffer | null = null;
    private indexBuffer: GPUBuffer | null = null;
    private colorBuffer: GPUBuffer | null = null;
    private normalBuffer: GPUBuffer | null = null;
    private uvBuffer: GPUBuffer | null = null;
    private totalIndices: number = 0;

    private depthTexture: GPUTexture | null = null;

    //================================//
    private keysPressed: Set<string> = new Set();
    private isMouseDown: boolean = false;
    private lastMouseX: number = 0;
    private lastMouseY: number = 0;

    //================================//
    private camera = createCamera(1.0);
    private cameraMoved: boolean = true;
    private light : lightObject;

    //================================//
    private wireFrameLabel: HTMLLabelElement | null = null;
    private wireFrameCheckbox: HTMLInputElement | null = null;
    private wireFrame: boolean = false;
    private cullMode: cullModes = cullModes.BACK;
    private cullModeSelect: HTMLSelectElement | null = null;
    private useSortingLabel: HTMLLabelElement | null = null;
    private useSortingCheckbox: HTMLInputElement | null = null;
    private useSorting: boolean = false;

    //================================//
    constructor () 
    {
        this.device = null;

        setCameraPosition(this.camera, 300, 200, 300);
        rotateCameraBy(this.camera, 9 * Math.PI / 12, -Math.PI / 6);
        setCameraNearFar(this.camera, 0.1, 2000);
        this.camera.moveSpeed = 20.0;
        this.camera.rotateSpeed = 0.05;

        this.light = {
            position: new Float32Array([380, 400, 220]),
            color: new Float32Array([1.0, 1.0, 1.0]),
            intensity: 1.0
        };
    }

    //================================//
    async initialize(canvas: HTMLCanvasElement) 
    {
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

        this.shaderModule = CreateShaderModule(this.device, VertWGSL, FragWGSL, 'Transparency Shader Module');
    }

    //================================//
    initializePipelines()
    {
        if (this.device === null || this.presentationFormat === null) return;

        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Transparency Bind Group Layout',
            entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform" },
        }]});

        this.pipelineLayout = this.device.createPipelineLayout({
            label: 'Transparency Pipeline Layout',
            bindGroupLayouts: [this.bindGroupLayout],
        });

        this.depthTexture = this.device.createTexture({
            size: [this.canvas!.width, this.canvas!.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        if (this.shaderModule !== null) {
            this.renderPipeline = this.device.createRenderPipeline({
                label: 'Transparency Pipeline',
                layout: this.pipelineLayout,
                vertex: {
                    module: this.shaderModule.vertex,
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
                    module: this.shaderModule.fragment,
                    entryPoint: 'fs',
                    targets: [
                        {
                            format: this.presentationFormat,
                            /*
                            FULL DEFAULTS ARE:
                            blend: {
                                color: {
                                    operation: 'add',
                                    srcFactor: 'one',
                                    dstFactor: 'zero',
                                },
                                alpha: {
                                    operation: 'add',
                                    srcFactor: 'one',
                                    dstFactor: 'zero',
                                },
                            }

                            where operation in { 'add', 'subtract', 'reverse-subtract', 'min', 'max' }
                            where srcFactor and dstFactor in { 'zero', 'one', 'src-color', 'one-minus-src-color', 
                                'dst-color', 'one-minus-dst-color', 'src-alpha', 'one-minus-src-alpha', 'dst-alpha', 
                                'one-minus-dst-alpha', 'constant-color', 'one-minus-constant-color', 'constant-alpha', 
                                'one-minus-constant-alpha', 'src-alpha-saturated' }
                            
                            result = operation((src * srcFactor),  (dst * dstFactor))
                            */
                            blend: {
                                color: {
                                    srcFactor: "one",
                                    dstFactor: "one-minus-src-alpha",
                                },
                                alpha: {
                                    srcFactor: "one",
                                    dstFactor: "one-minus-src-alpha",
                                },
                            }
                        }
                    ],
                },
                primitive: {
                    topology: this.wireFrame ? "line-list" : "triangle-list",
                    cullMode: this.cullMode.toLowerCase() as GPUCullMode,
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
    initializeUtils()
    {
        const utilElement = getUtilElement();
        if (!utilElement) return;

        // Wireframe checkbox
        this.wireFrameCheckbox = document.createElement('input');
        this.wireFrameCheckbox.type = 'checkbox';
        this.wireFrameCheckbox.checked = this.wireFrame;
        this.wireFrameCheckbox.id = 'wireframe-checkbox';
        this.wireFrameCheckbox.addEventListener('change', () => {
            this.wireFrame = this.wireFrameCheckbox!.checked;
            // Recreate pipeline
            this.initializePipelines();
        });
        this.wireFrameLabel = document.createElement('label');
        this.wireFrameLabel.htmlFor = 'wireframe-checkbox';
        this.wireFrameLabel.textContent = ' Wireframe Mode ';

        utilElement.appendChild(this.wireFrameCheckbox);
        utilElement.appendChild(this.wireFrameLabel);

        utilElement.appendChild(document.createElement('br'));

        // Cull mode select
        this.cullModeSelect = document.createElement('select');
        this.cullModeSelect.style.color = 'black';
        const cullModes = ["none", "front", "back"];
        cullModes.forEach(mode => {
            const option = document.createElement('option');
            option.value = mode;
            option.text = mode.charAt(0).toUpperCase() + mode.slice(1); // Capitalize display text
            this.cullModeSelect!.appendChild(option);
        });
        this.cullModeSelect.value = this.cullMode as string;
        this.cullModeSelect.addEventListener('change', () => {
            this.cullMode = this.cullModeSelect!.value as cullModes;
            this.initializePipelines();
        });
        utilElement.appendChild(this.cullModeSelect);

        // Use sorting checkbox
        this.useSortingCheckbox = document.createElement('input');
        this.useSortingCheckbox.type = 'checkbox';
        this.useSortingCheckbox.checked = this.useSorting;
        this.useSortingCheckbox.id = 'use-sorting-checkbox';
        this.useSortingCheckbox.addEventListener('change', () => {
            this.useSorting = this.useSortingCheckbox!.checked;
        });
        this.useSortingLabel = document.createElement('label');
        this.useSortingLabel.htmlFor = 'use-sorting-checkbox';
        this.useSortingLabel.textContent = ' Use Sorting (correct transparency) ';
        utilElement.appendChild(document.createElement('br'));
        utilElement.appendChild(this.useSortingCheckbox);
        utilElement.appendChild(this.useSortingLabel);
    }

    //================================//
    initializeScene()
    {
        const back = createQuad(
            {
                translation: glm.vec3.fromValues(0, 0, -100),
                rotation: glm.vec3.fromValues(0, 0, 0),
                scale: glm.vec3.fromValues(200, 200, 1)
            },
            [0.8, 0.8, 0.7] // light beige
        );
        back.additionalInfo = { order: 0, numVertices: back.vertexData.length / 3 }; // Keep track of order in final joint buffers
        this.facesTopologyInformation.push( back );

        const left = createQuad(
            {
                translation: glm.vec3.fromValues(-100, 0, 0),
                rotation: glm.vec3.fromValues(0, -Math.PI / 2, 0),
                scale: glm.vec3.fromValues(200, 200, 1)
            },
            [0.8, 0.8, 0.7]
        )
        left.additionalInfo = { order: 1, numVertices: left.vertexData.length / 3 };
        this.facesTopologyInformation.push( left );

        const floor = createQuad(
            {
                translation: glm.vec3.fromValues(0, -100, 0),
                rotation: glm.vec3.fromValues(Math.PI / 2, 0, 0),
                scale: glm.vec3.fromValues(200, 200, 1)
            },
            [0.8, 0.8, 0.7]
        );
        floor.additionalInfo = { order: 2, numVertices: floor.vertexData.length / 3 };
        this.facesTopologyInformation.push( floor );

        const sphereRadius = 25;
        const sphereDetails = 32;
        let currentSphereOrder = 3;
        let sphereID = 0;
        // Pyramid base
        const baseY = -100 + sphereRadius;
        for (let xi = -1; xi <= 1; xi++)
        {
            for (let zi = -1; zi <= 1; zi++)
            {
                const center: [number, number, number] = [xi * sphereRadius * 2, baseY, zi * sphereRadius * 2];
                const sphere = createSphere(
                    center,
                    sphereRadius,
                    [Math.random(), Math.random(), Math.random()],
                    sphereDetails,
                    sphereDetails
                );
                sphere.additionalInfo = { order: currentSphereOrder++, numVertices: sphere.vertexData.length / 3, id: sphereID++ };
                this.spheresTopologyInformation.push( sphere );
                this.currentSphereOrders.push(sphere.additionalInfo.id!);
            }
        }

        // Second layer
        const secondLayerY = baseY + sphereRadius * Math.sqrt(2);
        for (let xi = 0; xi <= 1; xi++)
        {
            for (let zi = 0; zi <= 1; zi++)
            {
                const center: [number, number, number] = [(xi - 0.5) * sphereRadius * 2, secondLayerY, (zi - 0.5) * sphereRadius * 2];
                const sphere = createSphere(
                    center,
                    sphereRadius,
                    [Math.random(), Math.random(), Math.random()],
                    sphereDetails,
                    sphereDetails
                );
                sphere.additionalInfo = { order: currentSphereOrder++, numVertices: sphere.vertexData.length / 3, id: sphereID++ };
                this.spheresTopologyInformation.push( sphere );
                this.currentSphereOrders.push(sphere.additionalInfo.id!);
            }
        }

        // Top sphere
        const topSphereCenter: [number, number, number] = [0, secondLayerY + sphereRadius * Math.sqrt(2), 0];
        const topSphere = createSphere(
            topSphereCenter,
            sphereRadius,
            [Math.random(), Math.random(), Math.random()],
            sphereDetails,
            sphereDetails
        );
        topSphere.additionalInfo = { order: currentSphereOrder++, numVertices: topSphere.vertexData.length / 3, id: sphereID++  };
        this.spheresTopologyInformation.push( topSphere );
        this.currentSphereOrders.push(topSphere.additionalInfo.id!);
    }

    //================================//
    initializeBuffers()
    {
        if (this.device === null) return;
        const queue = this.device.queue;

        // All topology information
        this.initializeScene();

        const totalVertexData: Float32Array[] = [];
        const totalIndexData: Uint16Array[] = [];
        const totalNormalData: Float32Array[] = [];
        const totalColorData: Float32Array[] = [];
        const totalUVData: Float32Array[] = [];

        for (let i = 0; i < this.facesTopologyInformation.length; i++)
        {
            const topoInfo = this.facesTopologyInformation[i];
            if (!topoInfo.additionalInfo) continue;

            totalVertexData.push(topoInfo.vertexData);
            totalIndexData.push(topoInfo.indexData);
            totalNormalData.push(topoInfo.normalData!);
            totalColorData.push(topoInfo.colorData!);
            totalUVData.push(topoInfo.uvData!);
        }

        // Draw the spheres first in random order, for random incorrect transparency at start
        const shuffledIndices = this.currentSphereOrders.slice();
        for (let i = shuffledIndices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]];
        }

        for (let i = 0; i < this.spheresTopologyInformation.length; i++)
        {
            const topoInfo = this.spheresTopologyInformation[shuffledIndices[i]];
            if (!topoInfo.additionalInfo) continue;

            totalVertexData.push(topoInfo.vertexData);
            totalIndexData.push(topoInfo.indexData);
            totalNormalData.push(topoInfo.normalData!);
            totalColorData.push(topoInfo.colorData!);
            totalUVData.push(topoInfo.uvData!);
        }
        const vertexCounts = totalVertexData.map(arr => arr.length / 3);

        const vertexData = mergeFloat32Arrays(totalVertexData);
        const indexData = mergeUint16Arrays(totalIndexData, vertexCounts);
        const normalData = mergeFloat32Arrays(totalNormalData);
        const colorData = mergeFloat32Arrays(totalColorData);
        const uvData = mergeFloat32Arrays(totalUVData);
        this.totalIndices = indexData.length;

        this.uniformBuffer = this.device.createBuffer({
            label: 'Uniform Buffer',
            size: uniformDataSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.vertexBuffer = this.device.createBuffer({
            label: 'Vertex Buffer',
            size: vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        queue.writeBuffer(this.vertexBuffer, 0, vertexData as BufferSource);

        this.normalBuffer = this.device.createBuffer({
            label: 'Normal Buffer',
            size: normalData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        queue.writeBuffer(this.normalBuffer, 0, normalData as BufferSource);

        this.colorBuffer = this.device.createBuffer({
            label: 'Color Buffer',
            size: colorData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        queue.writeBuffer(this.colorBuffer, 0, colorData as BufferSource);

        this.uvBuffer = this.device.createBuffer({
            label: 'UV Buffer',
            size: uvData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        queue.writeBuffer(this.uvBuffer, 0, uvData as BufferSource);

        this.indexBuffer = this.device.createBuffer({
            label: 'Index Buffer',
            size: indexData.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        queue.writeBuffer(this.indexBuffer, 0, indexData as BufferSource);

        this.bindGroup = this.device.createBindGroup({
            label: 'Transparency Bind Group',
            layout: this.bindGroupLayout!,
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer! },
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
    private onMouseUp = () => {
        this.isMouseDown = false;
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
            this.cameraMoved = true;
        }

        // rotation with arrow keys
        if (this.keysPressed.has('arrowleft')) rotateCameraByMouse(this.camera, -1, 0); // Yaw left
        if (this.keysPressed.has('arrowright')) rotateCameraByMouse(this.camera, 1, 0);
        if (this.keysPressed.has('arrowup')) rotateCameraByMouse(this.camera, 0, 1); // Pitch up
        if (this.keysPressed.has('arrowdown')) rotateCameraByMouse(this.camera, 0, -1);

        if (this.keysPressed.has('arrowleft') || this.keysPressed.has('arrowright') || this.keysPressed.has('arrowup') || this.keysPressed.has('arrowdown'))
            this.cameraMoved = true;
    }

    //================================//
    updateUniforms()
    {
        if (this.device === null || this.uniformBuffer === null) return;

        const data = new ArrayBuffer(uniformDataSize);
        const floatView = new Float32Array(data);

        const identityMatrix = glm.mat4.create();
        glm.mat4.identity(identityMatrix);

        floatView.set(identityMatrix, 0); // 16 floats
        floatView.set(this.camera.viewMatrix, 16); // 16 floats
        floatView.set(this.camera.projectionMatrix, 32); // 16 floats
        floatView.set(this.light.position, 48); // 4 floats
        floatView.set(this.light.color, 52); // 3 floats
        floatView[55] = this.light.intensity; // 1 float
        this.device.queue.writeBuffer(this.uniformBuffer, 0, data as BufferSource);
    }

    //================================//
    async startRendering()
    {
        await this.smallCleanup();

        this.initializeUtils();
        this.startMainLoop();
    }

    //================================//
    sortScene()
    {
        if (!this.useSorting) return;
        this.cameraMoved = false;

        const cameraPos = this.camera.position;

        const sphereDistances: { id: number; distance: number }[] = [];
        for (let i = 0; i < this.spheresTopologyInformation.length; i++)
        {
            const sphereTransform = this.spheresTopologyInformation[i].transform;
            const center: glm.vec3 = sphereTransform!.translation;

            const dx = center[0] - cameraPos[0];
            const dy = center[1] - cameraPos[1];
            const dz = center[2] - cameraPos[2];
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

            const sphereId = this.spheresTopologyInformation[i].additionalInfo!.id!;
            sphereDistances.push({ id: sphereId, distance: distance });
        }

        sphereDistances.sort((a, b) => b.distance - a.distance); // Farther spheres first
        this.currentSphereOrders = sphereDistances.map(item => item.id);

        const totalVertexData: Float32Array[] = [];
        const totalIndexData: Uint16Array[] = [];
        const totalNormalData: Float32Array[] = [];
        const totalColorData: Float32Array[] = [];
        const totalUVData: Float32Array[] = [];

        // Draw always faces first
        for (let i = 0; i < this.facesTopologyInformation.length; i++)
        {
            const topoInfo = this.facesTopologyInformation[i];
            if (!topoInfo.additionalInfo) continue;

            totalVertexData.push(topoInfo.vertexData);
            totalIndexData.push(topoInfo.indexData);
            totalNormalData.push(topoInfo.normalData!);
            totalColorData.push(topoInfo.colorData!);
            totalUVData.push(topoInfo.uvData!);
        }

        // Then draw spheres in sorted order
        for (let orderIndex = 0; orderIndex < this.currentSphereOrders.length; orderIndex++)
        {
            const sphereId = this.currentSphereOrders[orderIndex];
            const sphereInfo = this.spheresTopologyInformation.find(topo => topo.additionalInfo!.id === sphereId);
            if (!sphereInfo) continue;

            totalVertexData.push(sphereInfo.vertexData);
            totalIndexData.push(sphereInfo.indexData);
            totalNormalData.push(sphereInfo.normalData!);
            totalColorData.push(sphereInfo.colorData!);
            totalUVData.push(sphereInfo.uvData!);
        }

        const vertexCounts = totalVertexData.map(arr => arr.length / 3);
        const vertexData = mergeFloat32Arrays(totalVertexData);
        const indexData = mergeUint16Arrays(totalIndexData, vertexCounts);
        const normalData = mergeFloat32Arrays(totalNormalData);
        const colorData = mergeFloat32Arrays(totalColorData);
        const uvData = mergeFloat32Arrays(totalUVData);

        // Update buffers
        this.device!.queue.writeBuffer(this.vertexBuffer!, 0, vertexData as BufferSource);
        this.device!.queue.writeBuffer(this.indexBuffer!, 0, indexData as BufferSource);
        this.device!.queue.writeBuffer(this.normalBuffer!, 0, normalData as BufferSource);
        this.device!.queue.writeBuffer(this.colorBuffer!, 0, colorData as BufferSource);
        this.device!.queue.writeBuffer(this.uvBuffer!, 0, uvData as BufferSource);
    }

    //================================//
    startMainLoop()
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
            if (this.cameraMoved)
                this.sortScene();

            const textureView = this.context.getCurrentTexture().createView();
            const depthStencilAttachment: GPURenderPassDepthStencilAttachment = 
            {
                view: this.depthTexture!.createView(),
                depthLoadOp: 'clear' as const,
                depthStoreOp: 'store' as const,
                depthClearValue: 1.0,
            };
            const renderPassDescriptor: GPURenderPassDescriptor = {
                label: 'basic canvas renderPass',
                colorAttachments: [{
                    view: textureView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 }
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
            pass.setPipeline(this.renderPipeline!);
            pass.setBindGroup(0, this.bindGroup!);
            pass.setVertexBuffer(0, this.vertexBuffer!);
            pass.setVertexBuffer(1, this.normalBuffer!);
            pass.setVertexBuffer(2, this.uvBuffer!);
            pass.setVertexBuffer(3, this.colorBuffer!);
            pass.setIndexBuffer(this.indexBuffer!, 'uint16');
            pass.drawIndexed(this.totalIndices, 1, 0, 0, 0);
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
                    
                    // Prevent distortion AND update depth texture
                    setCameraAspect(this.camera, this.canvas.width / this.canvas.height);
                    
                    // Recreate depth texture with new size
                    if (this.depthTexture) {
                        this.depthTexture.destroy();
                        this.depthTexture = this.device.createTexture({
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
            while(this.infoElement.firstChild) {
                this.infoElement.removeChild(this.infoElement.firstChild);
            }
        }
    }

    //================================//
    async smallCleanup()
    {
        if (this.cullModeSelect)
        {
            this.cullModeSelect.removeEventListener('change', () => {
                this.cullMode = this.cullModeSelect!.value as cullModes;
                this.initializePipelines();
            });
        }
        if (this.wireFrameCheckbox)
        {
            this.wireFrameCheckbox.removeEventListener('change', () => {
                this.wireFrame = this.wireFrameCheckbox!.checked;
                this.initializePipelines();
            });
        }
        if (this.useSortingCheckbox)
        {
            this.useSortingCheckbox.removeEventListener('change', () => {
                this.useSorting = this.useSortingCheckbox!.checked;
            });
        }

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
}