
//============== START IMPORTS ==================//
import * as glm from 'gl-matrix';

//================================//
import rayTraceVertWGSL from './MainArchitectureShaders/raytrace_vert.wgsl?raw';
import rayTraceFragWGSL from './MainArchitectureShaders/raytrace_frag.wgsl?raw';

import rasterVertWgsl from './MainArchitectureShaders/raster_vert.wgsl?raw';
import rasterFragWgsl from './MainArchitectureShaders/raster_frag.wgsl?raw';

import bvhVertWGSL from './MainArchitectureShaders/bvh_vert.wgsl?raw';
import bvhFragWGSL from './MainArchitectureShaders/bvh_frag.wgsl?raw';

import sceneMinMaxWGSL from './FastBVHShaders/sceneMinMax.wgsl?raw';
import minMaxReduceWGSL from './FastBVHShaders/minMaxReduce.wgsl?raw';
import mortonCodeWGSL from './FastBVHShaders/mortonCode.wgsl?raw';
import radixBlockSumsWGSL from './FastBVHShaders/radixBlockSums.wgsl?raw';
import radixPrefixSum from './FastBVHShaders/radixPrefixSum.wgsl?raw';
import radixReorderWGSL from './FastBVHShaders/radixReorder.wgsl?raw';
import patriciaTreeWGSL from './FastBVHShaders/patriciaTree.wgsl?raw';
import AABBWGSL from './FastBVHShaders/AABB.wgsl?raw';
import wireframeWGSL from './FastBVHShaders/wireframe.wgsl?raw';

//================================//
import { RequestWebGPUDevice, CreateShaderModule, CreateTimestampQuerySet } from '@src/helpers/WebGPUutils';
import type { PipelineResources, ShaderModule, TimestampQuerySet } from '@src/helpers/WebGPUutils';
import { addButton, addCheckbox, addNumberInput, addProfilerFrameTime, addSlider, cleanUtilElement, createLightContextMenu, createMaterialContextMenu, getInfoElement, getUtilElement, type SpotLight } from '@src/helpers/Others';
import { createCamera, moveCameraLocal, rotateCameraByMouse, setCameraPosition, setCameraNearFar, setCameraAspect, computePixelToRayMatrix, rotateCameraBy, cameraPointToRay } from '@src/helpers/CameraHelpers';
import { fastBVHExampleScene, type Ray, type SceneInformation } from '@src/helpers/GeometryUtils';
import { createPlaceholderImage, createPlaceholderTexture, createTextureFromImage, loadImageFromUrl, resizeImage, TextureType } from '@src/helpers/ImageHelpers';
import { flattenMaterial, flattenMaterialArray, MATERIAL_SIZE, type Material } from '@src/helpers/MaterialUtils';
//============== END IMPORTS ==================//

//============== START PAPER IMPLEMENTATION ==================//
interface minMaxLevel
{
    workgroupCount: number;
    minMaxPipeline: GPUComputePipeline;
    minMaxBindGroup: GPUBindGroup;
    input: GPUBuffer;
    output: GPUBuffer;
    dispatchX: number;
    dispatchY: number;
}
interface PrefixSumLevel
{
    elementCount: number;
    workgroupCount: number;
    reducePipeline: GPUComputePipeline;
    addPipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    dataBuffer: GPUBuffer;
    blockSumBuffer: GPUBuffer;
    dispatchX: number;
    dispatchY: number;
}

const BVHNodeSize = 48;
const LeafAABBSize = 32;
const floatsPerNode = 24 * 3;
//================================//
class FastParallelBVH
{
    debug: boolean = false;

    //================================//
    private THREADS_PER_WORKGROUP = 256;
    private SIZE_X = 16;
    private SIZE_Y = 16;
    numTriangles: number = 0;
    private ITEMS_PER_WORKGROUP = 2 * this.THREADS_PER_WORKGROUP;
    private BIT_COUNT = 30;
    private NUM_PASSES = this.BIT_COUNT / 2;

    //=============== Min Max objects =================// 
    private minMaxPipelineLayout!: GPUPipelineLayout;
    private minMaxBindGroupLayout!: GPUBindGroupLayout;
    private minMaxReduceShaderModule!: GPUShaderModule;
    private minMaxSceneShaderModule!: GPUShaderModule;
    private minMaxLevels: minMaxLevel[] = [];
    minMaxReadbackBuffer: GPUBuffer | null = null;

    //=============== Morton code computation objects =================//
    private mortonPipelineLayout!: GPUPipelineLayout;
    private mortonPipeline!: GPUComputePipeline;
    private mortonBindGroupLayout!: GPUBindGroupLayout;
    private mortonShaderModule!: GPUShaderModule;
    private mortonBindGroup!: GPUBindGroup;
    private mortonOutputBitsBuffer!: GPUBuffer;
    private mortonOutputTriangleIndexBuffer!: GPUBuffer;

    //============== Radix Sort objects ==================//
    private WORKGROUP_COUNT!: number;
    private radixSortBindGroupLayout!: GPUBindGroupLayout;
    private reorderBindGroupLayout!: GPUBindGroupLayout;
    private prefixSumBindGroupLayout!: GPUBindGroupLayout;
    private radixSortPipelineLayout!: GPUPipelineLayout;
    private reorderPipelineLayout!: GPUPipelineLayout;
    private prefixSumPipelineLayout!: GPUPipelineLayout;
    private radixSortShaderModule!: GPUShaderModule;
    private reorderShaderModule!: GPUShaderModule;
    private prefixSumShaderModule!: GPUShaderModule;

    private prefixSumLevels: PrefixSumLevel[] = [];
    private radixSortPipeline!: GPUComputePipeline;
    private reorderPipeline!: GPUComputePipeline;
    
    private radixSortBindGroups!:   [GPUBindGroup, GPUBindGroup];
    private reorderBindGroups!:     [GPUBindGroup, GPUBindGroup];

    private keysBufferA!: GPUBuffer;
    private keysBufferB!: GPUBuffer;
    private valuesBufferA!: GPUBuffer;
    private valuesBufferB!: GPUBuffer;
    private localPrefixSumBuffer!: GPUBuffer;
    private blockSumBuffer!: GPUBuffer;

    private uniformBuffers!: GPUBuffer[];
    private uniformBindGroups!: GPUBindGroup[];
    private uniformBindGroupLayout!: GPUBindGroupLayout;

    //============== Patricia Tree objects ==================//
    private patriciaTreeShaderModule!: GPUShaderModule;
    private patriciaTreeBindGroupLayout!: GPUBindGroupLayout;
    private patriciaTreePipelineLayout!: GPUPipelineLayout;
    private patriciaTreePipeline!: GPUComputePipeline;
    private patriciaTreeBindGroup!: GPUBindGroup;

    private mortonCodesBuffer!: GPUBuffer;
    private internalNodesBuffer!: GPUBuffer;
    private leafNodesBuffer!: GPUBuffer;

    //=============== AABB objects =================//
    private aabbShaderModule!: GPUShaderModule;
    private aabbPipelineLayout!: GPUPipelineLayout;
    private aabbBindGroupLayout!: GPUBindGroupLayout;
    private aabbPipeline!: GPUComputePipeline;
    private aabbBindGroup!: GPUBindGroup;

    private aabbInternalNodesBuffer!: GPUBuffer;
    private aabbLeafNodesBuffer!: GPUBuffer;
    private aabbAtomicCountersBuffer!: GPUBuffer;
    private aabbVertexBuffer!: GPUBuffer;
    private aabbIndexBuffer!: GPUBuffer;
    private aabbSortedIndexBuffer!: GPUBuffer;
    private leafAABBsBuffer!: GPUBuffer;

    //=============== Wireframe Visualization objects =================//
    private wireframeShaderModule!: GPUShaderModule;
    private wireframePipelineLayout!: GPUPipelineLayout;
    private wireframeBindGroupLayout!: GPUBindGroupLayout;
    private wireframePipeline!: GPUComputePipeline;
    private wireframeBindGroup!: GPUBindGroup;
    public wireframeVertexBuffer!: GPUBuffer;
    public wireframeVertexCount: number = 0;
    public wireframeDepthBuffer!: GPUBuffer;

    //================================//
    constructor()
    {
    }

    //================================//
    dispatch(commandEncoder: GPUComputePassEncoder)
    {
        this.dispatchMinMaxPass(commandEncoder);
        this.dispatchMortonPass(commandEncoder);
        this.dispatchRadixSort(commandEncoder);
        this.dispatchPatriciaTreePass(commandEncoder);
        this.dispatchAABBPass(commandEncoder);
        this.dispatchWireframePass(commandEncoder);
    }

    //================================//
    dispatchSize(workgroupCount: number): [number, number]
    {
        const x = Math.min(workgroupCount, 65535);
        const y = Math.ceil(workgroupCount / 65535);
        return [x, y];
    }

    //============== Min Max Methods ==================//
    initializeMinMaxPipeline(device: GPUDevice, topLevelVertexBuffer: GPUBuffer, totalVertexCount: number)
    {
        this.minMaxLevels = [];
        
        this.minMaxReduceShaderModule = device.createShaderModule({label: 'Scene Min Max Reduce Shader Module', code: minMaxReduceWGSL});
        this.minMaxSceneShaderModule = device.createShaderModule({label: 'Scene Min Max Scene Shader Module', code: sceneMinMaxWGSL});

        this.minMaxBindGroupLayout = device.createBindGroupLayout({
            label: 'Scene Min Max Bind Group Layout',
            entries: [  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // input buffer
                        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },]           // output buffer
        });
        this.minMaxPipelineLayout = device.createPipelineLayout({label: 'Scene Min Max Pipeline Layout', bindGroupLayouts: [this.minMaxBindGroupLayout]});

        // Level 0 : First pass of vertices -> min/max output weaved buffer
        let workGroupCount = Math.ceil(totalVertexCount / this.THREADS_PER_WORKGROUP);
        let [dx, dy] = this.dispatchSize(workGroupCount);
        const level0OutputBuffer = device.createBuffer({
            label: 'Scene Min Max Level 0 Output Buffer',
            size: workGroupCount * 6 * 4, // Buffer is in shape [ minX, minY, minZ, maxX, maxY, maxZ ] for each workgroup
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const level0BindGroup = device.createBindGroup({
            label: 'Scene Min Max Level 0 Bind Group',
            layout: this.minMaxBindGroupLayout,
            entries: [ { binding: 0, resource: { buffer: topLevelVertexBuffer } }, { binding: 1, resource: { buffer: level0OutputBuffer } } ]
        });
        const level0Pipeline = device.createComputePipeline({
            label: 'Scene Min Max Level 0 Pipeline',
            layout: this.minMaxPipelineLayout,
            compute: {
                module: this.minMaxSceneShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, SIZE_X: this.SIZE_X, SIZE_Y: this.SIZE_Y, TOTAL_VERTICES: totalVertexCount }
            }
        });
        this.minMaxLevels.push({ workgroupCount: workGroupCount, minMaxPipeline: level0Pipeline, minMaxBindGroup: level0BindGroup, input: topLevelVertexBuffer, output: level0OutputBuffer, dispatchX: dx, dispatchY: dy });

        let currentWorkGroupCount = workGroupCount;
        let currentInputBuffer = level0OutputBuffer;

        // Level 1 - N : reduces minMaxs into final scene min max
        while (true)
        {
            const workGroupCount = Math.ceil(currentWorkGroupCount / this.THREADS_PER_WORKGROUP);
            const [dx, dy] = this.dispatchSize(workGroupCount);

            const outputBuffer = device.createBuffer({
                label: `Scene Min Max Level ${this.minMaxLevels.length} Output Buffer`,
                size: workGroupCount * 6 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            });
            const bindGroup = device.createBindGroup({
                label: `Scene Min Max Level ${this.minMaxLevels.length} Bind Group`,
                layout: this.minMaxBindGroupLayout,
                entries: [ { binding: 0, resource: { buffer: currentInputBuffer } }, { binding: 1, resource: { buffer: outputBuffer } } ]
            });
            const pipeline = device.createComputePipeline({
                label: `Scene Min Max Level ${this.minMaxLevels.length} Pipeline`,
                layout: this.minMaxPipelineLayout,
                compute: {
                    module: this.minMaxReduceShaderModule,
                    entryPoint: 'cs',
                    constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, SIZE_X: this.SIZE_X, SIZE_Y: this.SIZE_Y, ELEMENT_COUNT: currentWorkGroupCount }
                }
            });

            this.minMaxLevels.push({ workgroupCount: workGroupCount, minMaxPipeline: pipeline, minMaxBindGroup: bindGroup, input: currentInputBuffer, output: outputBuffer, dispatchX: dx, dispatchY: dy });

            if (workGroupCount <= 1) break;

            currentWorkGroupCount = workGroupCount;
            currentInputBuffer = outputBuffer;
        }

        this.minMaxReadbackBuffer = device.createBuffer({
            label: 'MinMax Readback Buffer',
            size: 6 * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }

    //================================//
    copyResultForReadback(encoder: GPUCommandEncoder)
    {
        if (!this.minMaxReadbackBuffer || this.minMaxLevels.length === 0) return;
        const lastOutput = this.minMaxLevels[this.minMaxLevels.length - 1].output;
        encoder.copyBufferToBuffer(lastOutput, 0, this.minMaxReadbackBuffer, 0, 6 * 4);
    }

    //================================//
    dispatchMinMaxPass(commandEncoder: GPUComputePassEncoder)
    {
        for (const level of this.minMaxLevels)
        {
            commandEncoder.setPipeline(level.minMaxPipeline);
            commandEncoder.setBindGroup(0, level.minMaxBindGroup);
            commandEncoder.dispatchWorkgroups(level.dispatchX, level.dispatchY);
        }
    }

    //============== Morton Code Methods ==================//
    initializeMortonPipeline(device: GPUDevice, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer, numTriangles: number)
    {
        this.numTriangles = numTriangles;
        this.mortonShaderModule = device.createShaderModule({ label: 'Morton Code Shader Module', code: mortonCodeWGSL });
        this.mortonBindGroupLayout = device.createBindGroupLayout({
            label: 'Morton Code Bind Group Layout',
            entries: [  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // vertex buffer
                        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // index buffer
                        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // scene min max buffer
                        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // output morton bits buffer
                        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },]           // output triangle index buffer
        });
        this.mortonPipelineLayout = device.createPipelineLayout({ label: 'Morton Code Pipeline Layout', bindGroupLayouts: [this.mortonBindGroupLayout] });
        this.mortonPipeline = device.createComputePipeline({
            label: 'Morton Code Pipeline',
            layout: this.mortonPipelineLayout,
            compute: {
                module: this.mortonShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, SIZE_X: this.SIZE_X, SIZE_Y: this.SIZE_Y, TRIANGLE_COUNT: numTriangles },
            }
        });

        this.mortonOutputBitsBuffer = device.createBuffer({
            label: 'Morton Code Output Bits Buffer',
            size: numTriangles * 4, // u32 == 4 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        this.mortonOutputTriangleIndexBuffer = device.createBuffer({
            label: 'Morton Code Output Triangle Index Buffer',
            size: numTriangles * 4, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.mortonBindGroup = device.createBindGroup({
            label: 'Morton Code Bind Group',
            layout: this.mortonBindGroupLayout,
            entries: [ { binding: 0, resource: { buffer: vertexBuffer } },
                        { binding: 1, resource: { buffer: indexBuffer } },
                        { binding: 2, resource: { buffer: this.minMaxLevels[this.minMaxLevels.length - 1].output } },
                        { binding: 3, resource: { buffer: this.mortonOutputBitsBuffer } },
                        { binding: 4, resource: { buffer: this.mortonOutputTriangleIndexBuffer } } ]
        });
    }

    //================================//
    dispatchMortonPass(commandEncoder: GPUComputePassEncoder)
    {
        if (!this.mortonShaderModule || !this.mortonBindGroup) return;

        const [dx, dy] = this.dispatchSize(Math.ceil(this.numTriangles / this.THREADS_PER_WORKGROUP));
        commandEncoder.setPipeline(this.mortonPipeline);
        commandEncoder.setBindGroup(0, this.mortonBindGroup);
        commandEncoder.dispatchWorkgroups(dx, dy);
    }

    //============ Radix sort methods ====================//
    initializeRadixSortPipelines(device: GPUDevice)
    {
        const totalElements = this.numTriangles;
        this.WORKGROUP_COUNT = Math.ceil(totalElements / this.THREADS_PER_WORKGROUP);

        this.radixSortShaderModule = device.createShaderModule({ label: 'Radix Sort Shader Module', code: radixBlockSumsWGSL });
        this.reorderShaderModule = device.createShaderModule({ label: 'Radix Reorder Shader Module', code: radixReorderWGSL });
        this.prefixSumShaderModule = device.createShaderModule({ label: 'Prefix Sum Reduce Shader Module', code: radixPrefixSum });

        // Uniform bind group layout
        this.uniformBindGroupLayout = device.createBindGroupLayout({
            label: 'Uniform Bind Group Layout',
            entries: [  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } } ]
        });

        this.uniformBuffers = [];
        this.uniformBindGroups = [];
        for( let i = 0; i < this.NUM_PASSES; i++)
        {
            const buffer = device.createBuffer({ label: `Radix Sort Uniform Buffer Pass ${i}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(buffer, 0, new Uint32Array([i * 2]));
            const bindGroup = device.createBindGroup({ label: `Radix Sort Uniform Bind Group Pass ${i}`, layout: this.uniformBindGroupLayout, entries: [ { binding: 0, resource: { buffer: buffer } } ] });
            this.uniformBindGroups.push(bindGroup);
            this.uniformBuffers.push(buffer);
        }

        // Prefix sum bind group layout
        this.prefixSumBindGroupLayout = device.createBindGroupLayout({
            label: 'Prefix Sum Bind Group Layout',
                entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // Data to scan / add to
                            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }} ] // block sums buffer
        });
        this.prefixSumPipelineLayout = device.createPipelineLayout({ label: 'Prefix Sum Pipeline Layout', bindGroupLayouts: [this.prefixSumBindGroupLayout] });

        // Shared buffers
        this.keysBufferA = this.mortonOutputBitsBuffer;
        this.keysBufferB = device.createBuffer({
            label: 'Radix Sort Keys Buffer B',
            size: this.numTriangles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.valuesBufferA = this.mortonOutputTriangleIndexBuffer;
        this.valuesBufferB = device.createBuffer({
            label: 'Radix Sort Values Buffer B',
            size: this.numTriangles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.localPrefixSumBuffer = device.createBuffer({
            label: 'Radix Sort Local Prefix Sum Buffer',
            size: this.numTriangles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.blockSumBuffer = device.createBuffer({
            label: 'Radix Sort Block Sum Buffer',
            size: 4 * this.WORKGROUP_COUNT * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // [1] Radix sort pipeline
        this.radixSortBindGroupLayout = device.createBindGroupLayout({
            label: 'Radix Sort Bind Group Layout',
            entries: [  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input array
                        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // local prefix sum buffer
                        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },]           // block sums buffer
        });
        this.radixSortPipelineLayout = device.createPipelineLayout({ label: 'Radix Sort Pipeline Layout', bindGroupLayouts: [this.radixSortBindGroupLayout, this.uniformBindGroupLayout] });
        this.radixSortPipeline = device.createComputePipeline({
            label: 'Radix Sort Pipeline',
            layout: this.radixSortPipelineLayout,
            compute: {
                module: this.radixSortShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, X_SIZE: this.SIZE_X, Y_SIZE: this.SIZE_Y, ELEMENT_COUNT: this.numTriangles, WORKGROUP_COUNT: this.WORKGROUP_COUNT }
            },
        });
        this.radixSortBindGroups = [
            device.createBindGroup({
                label: 'Radix Sort Bind Group',
                layout: this.radixSortBindGroupLayout,
                entries: [  { binding: 0, resource: { buffer: this.keysBufferA } }, 
                            { binding: 1, resource: { buffer: this.localPrefixSumBuffer } }, 
                            { binding: 2, resource: { buffer: this.blockSumBuffer } } ]
            }),
            device.createBindGroup({
                label: 'Radix Sort Bind Group 2',
                layout: this.radixSortBindGroupLayout,
                entries: [  { binding: 0, resource: { buffer: this.keysBufferB } },
                            { binding: 1, resource: { buffer: this.localPrefixSumBuffer } },
                            { binding: 2, resource: { buffer: this.blockSumBuffer } } ]
            })
        ];

        // [2] Reorder pipeline
        this.reorderBindGroupLayout = device.createBindGroupLayout({
            label: 'Reorder Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input keys
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // local prefix sum buffer
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // block sums buffer
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // input values
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },]          // output values
        });
        this.reorderPipelineLayout = device.createPipelineLayout({ label: 'Reorder Pipeline Layout', bindGroupLayouts: [this.reorderBindGroupLayout, this.uniformBindGroupLayout] });
        this.reorderPipeline = device.createComputePipeline({
            label: 'Reorder Pipeline',
            layout: this.reorderPipelineLayout,
            compute: {
                module: this.reorderShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, X_SIZE: this.SIZE_X, Y_SIZE: this.SIZE_Y, ELEMENT_COUNT: this.numTriangles, WORKGROUP_COUNT: this.WORKGROUP_COUNT }
            },
        });
        this.reorderBindGroups = [
            device.createBindGroup({
                label: 'Reorder Bind Group',
                layout: this.reorderBindGroupLayout,
                entries: [  { binding: 0, resource: { buffer: this.keysBufferA } },
                            { binding: 1, resource: { buffer: this.keysBufferB } },
                            { binding: 2, resource: { buffer: this.localPrefixSumBuffer } },
                            { binding: 3, resource: { buffer: this.blockSumBuffer } },
                            { binding: 4, resource: { buffer: this.valuesBufferA } },
                            { binding: 5, resource: { buffer: this.valuesBufferB } } ]
            }),
            device.createBindGroup({
                label: 'Reorder Bind Group 2',
                layout: this.reorderBindGroupLayout,
                entries: [  { binding: 0, resource: { buffer: this.keysBufferB } },
                            { binding: 1, resource: { buffer: this.keysBufferA } },
                            { binding: 2, resource: { buffer: this.localPrefixSumBuffer } },
                            { binding: 3, resource: { buffer: this.blockSumBuffer } },
                            { binding: 4, resource: { buffer: this.valuesBufferB } },
                            { binding: 5, resource: { buffer: this.valuesBufferA } } ]
            })
        ];

        // [3] Prefix sum pipeline
        this.prefixSumLevels = [];
        let currentElementCount = 4 * this.WORKGROUP_COUNT;
        let currentDataBuffer = this.blockSumBuffer;

        while(true)
        {
            const workgroupCount = Math.ceil(currentElementCount / this.ITEMS_PER_WORKGROUP);
            const [dx, dy] = this.dispatchSize(workgroupCount);

            const blockSumBuffer = device.createBuffer({ label: `Prefix Sum Block Sum Buffer Level ${this.prefixSumLevels.length}`, size: Math.max(workgroupCount, 1) * 4, usage: GPUBufferUsage.STORAGE });
            const reducePipeline = device.createComputePipeline({
                label: `Prefix Sum Reduce Pipeline Level ${this.prefixSumLevels.length}`,
                layout: this.prefixSumPipelineLayout,
                compute: {
                    module: this.prefixSumShaderModule,
                    entryPoint: 'cs_reduce',
                    constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, X_SIZE: this.SIZE_X, Y_SIZE: this.SIZE_Y, ITEMS_PER_WORKGROUP: this.ITEMS_PER_WORKGROUP,ELEMENT_COUNT: currentElementCount }
                }
            });
            const addPipeline = device.createComputePipeline({
                label: `Prefix Sum Add Pipeline Level ${this.prefixSumLevels.length}`,
                layout: this.prefixSumPipelineLayout,
                compute: {
                    module: this.prefixSumShaderModule,
                    entryPoint: 'cs_add',
                    constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, X_SIZE: this.SIZE_X, Y_SIZE: this.SIZE_Y, ITEMS_PER_WORKGROUP: this.ITEMS_PER_WORKGROUP,ELEMENT_COUNT: currentElementCount }
                }
            });

            const bindGroup = device.createBindGroup({
                label: `Prefix Sum Bind Group Level ${this.prefixSumLevels.length}`,
                layout: this.prefixSumBindGroupLayout,
                entries: [ { binding: 0, resource: { buffer: currentDataBuffer } }, { binding: 1, resource: { buffer: blockSumBuffer } } ]
            });

            this.prefixSumLevels.push({ elementCount: currentElementCount, workgroupCount: workgroupCount, reducePipeline: reducePipeline, addPipeline: addPipeline, bindGroup: bindGroup, dataBuffer: currentDataBuffer, blockSumBuffer: blockSumBuffer, dispatchX: dx, dispatchY: dy });

            if (workgroupCount <= 1) break;

            currentElementCount = workgroupCount;
            currentDataBuffer = blockSumBuffer;
        }
    }

    //================================//
    dispatchRadixSort(commandEncoder: GPUComputePassEncoder)
    {
        const [dx, dy] = this.dispatchSize(this.WORKGROUP_COUNT);

        for (let pass = 0; pass < this.NUM_PASSES; pass++)
        {
            const isEvenPass = (pass % 2 === 0);
            let uniformBindGroup = this.uniformBindGroups[pass];

            // [1] Radix sort pass
            commandEncoder.setPipeline(this.radixSortPipeline);
            commandEncoder.setBindGroup(0, isEvenPass ? this.radixSortBindGroups[0] : this.radixSortBindGroups[1]);
            commandEncoder.setBindGroup(1, uniformBindGroup);
            commandEncoder.dispatchWorkgroups(dx, dy, 1);

            // [2] Prefix sum passes
            const N = this.prefixSumLevels.length;
            
            for (let i = 0; i < N; i++)
            {
                const level = this.prefixSumLevels[i];
                commandEncoder.setPipeline(level.reducePipeline);
                commandEncoder.setBindGroup(0, level.bindGroup);
                commandEncoder.dispatchWorkgroups(level.dispatchX, level.dispatchY, 1);
            }

            for (let i = N - 2; i >= 0; i--)
            {
                const level = this.prefixSumLevels[i];
                commandEncoder.setPipeline(level.addPipeline);
                commandEncoder.setBindGroup(0, level.bindGroup);
                commandEncoder.dispatchWorkgroups(level.dispatchX, level.dispatchY, 1);
            }

            // [3] Reorder pass
            commandEncoder.setPipeline(this.reorderPipeline);
            commandEncoder.setBindGroup(0, isEvenPass ? this.reorderBindGroups[0] : this.reorderBindGroups[1]);
            commandEncoder.setBindGroup(1, uniformBindGroup);
            commandEncoder.dispatchWorkgroups(dx, dy, 1);
        }
    }

    //============= Patricia Tree Methods ===================//
    initializePatriciaTreePipeline(device: GPUDevice)
    {
        this.patriciaTreeShaderModule = device.createShaderModule({ label: 'Patricia Tree Shader Module', code: patriciaTreeWGSL });
        this.patriciaTreeBindGroupLayout = device.createBindGroupLayout({
            label: 'Patricia Tree Bind Group Layout',
            entries: [  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // morton codes buffer
                        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // internal nodes buffer
                        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },]           // leaf nodes buffer
        });
        this.patriciaTreePipelineLayout = device.createPipelineLayout({ label: 'Patricia Tree Pipeline Layout', bindGroupLayouts: [this.patriciaTreeBindGroupLayout] });
        this.patriciaTreePipeline = device.createComputePipeline({
            label: 'Patricia Tree Pipeline',
            layout: this.patriciaTreePipelineLayout,
            compute: {
                module: this.patriciaTreeShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, INTERNAL_NODE_COUNT: this.numTriangles - 1, LEAF_NODE_COUNT: this.numTriangles }
            },
        });

        this.mortonCodesBuffer = this.keysBufferB; // Results of the radix sort live in B (15 passes -> ends in B)

        this.internalNodesBuffer = device.createBuffer({
            label: 'BVH Internal Nodes Buffer',
            size: (this.numTriangles - 1) * BVHNodeSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        this.leafNodesBuffer = device.createBuffer({
            label: 'BVH Leaf Nodes Buffer',
            size: this.numTriangles * 4, // u32 array
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.patriciaTreeBindGroup = device.createBindGroup({
            label: 'Patricia Tree Bind Group',
            layout: this.patriciaTreeBindGroupLayout,
            entries: [  { binding: 0, resource: { buffer: this.mortonCodesBuffer } },
                        { binding: 1, resource: { buffer: this.internalNodesBuffer } },
                        { binding: 2, resource: { buffer: this.leafNodesBuffer } } ]
        });
    }

    //================================//
    dispatchPatriciaTreePass(commandEncoder: GPUComputePassEncoder)
    {
        if (!this.patriciaTreePipeline || !this.patriciaTreeBindGroup) return;

        const internalNodeCount = this.numTriangles - 1;
        const dispatchX = Math.ceil(internalNodeCount / this.THREADS_PER_WORKGROUP);

        commandEncoder.setPipeline(this.patriciaTreePipeline);
        commandEncoder.setBindGroup(0, this.patriciaTreeBindGroup);
        commandEncoder.dispatchWorkgroups(dispatchX, 1, 1);
    }

    //============== AABB Construction Methods ==================//
    initializeAABBPipeline(device: GPUDevice, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer)
    {
        this.aabbShaderModule = device.createShaderModule({ label: 'AABB Shader Module', code: AABBWGSL });
        this.aabbBindGroupLayout = device.createBindGroupLayout({
            label: 'AABB Bind Group Layout',
            entries: [  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // internal nodes buffer
                        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // leaf nodes buffer (containing parent pointers)
                        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // atomic counters
                        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // vertex buffer
                        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // index buffer
                        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // sorted index buffer (output of radix sort pass)
                        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },] // leaf AABBs buffer
        });
        this.aabbPipelineLayout = device.createPipelineLayout({ label: 'AABB Pipeline Layout', bindGroupLayouts: [this.aabbBindGroupLayout] });
        this.aabbPipeline = device.createComputePipeline({
            label: 'AABB Pipeline',
            layout: this.aabbPipelineLayout,
            compute: {
                module: this.aabbShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, INTERNAL_NODE_COUNT: this.numTriangles - 1, LEAF_NODE_COUNT: this.numTriangles }
            },
        });
        
        this.aabbInternalNodesBuffer = this.internalNodesBuffer;
        this.aabbLeafNodesBuffer = this.leafNodesBuffer;
        this.aabbAtomicCountersBuffer = device.createBuffer({
            label: 'AABB Atomic Counters Buffer',
            size: 4 * (this.numTriangles - 1), // counters for internal nodes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST // We need Copy_dst so I can clear it to 0s ech pass
        });
        this.aabbVertexBuffer = vertexBuffer;
        this.aabbIndexBuffer = indexBuffer;
        this.aabbSortedIndexBuffer = this.valuesBufferB;
        this.leafAABBsBuffer = device.createBuffer({
            label: 'Leaf AABBs Buffer',
            size: this.numTriangles * LeafAABBSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.aabbBindGroup = device.createBindGroup({
            label: 'AABB Bind Group',
            layout: this.aabbBindGroupLayout,
            entries: [  { binding: 0, resource: { buffer: this.aabbInternalNodesBuffer } },
                        { binding: 1, resource: { buffer: this.aabbLeafNodesBuffer } },
                        { binding: 2, resource: { buffer: this.aabbAtomicCountersBuffer } },
                        { binding: 3, resource: { buffer: this.aabbVertexBuffer } },
                        { binding: 4, resource: { buffer: this.aabbIndexBuffer } },
                        { binding: 5, resource: { buffer: this.aabbSortedIndexBuffer } },
                        { binding: 6, resource: { buffer: this.leafAABBsBuffer } } ]
        });
    }

    //================================//
    clearAtomicCounters(encoder: GPUCommandEncoder)
    {
        if (!this.aabbAtomicCountersBuffer) return;
        encoder.clearBuffer(this.aabbAtomicCountersBuffer);
    }

    //================================//
    dispatchAABBPass(commandEncoder: GPUComputePassEncoder)
    {
        if (!this.aabbPipeline || !this.aabbBindGroup) return;

        const dispatchX = Math.ceil((this.numTriangles) / this.THREADS_PER_WORKGROUP);
        commandEncoder.setPipeline(this.aabbPipeline);
        commandEncoder.setBindGroup(0, this.aabbBindGroup);
        commandEncoder.dispatchWorkgroups(dispatchX, 1, 1);
    }

    //============== Wireframe Visualization Methods ==================//
    initializeWireframePipeline(device: GPUDevice, initialDepth: number)
    {
        const totalNodes = (this.numTriangles - 1) + this.numTriangles; // internal + leaf
        this.wireframeVertexCount = totalNodes * 24;

        this.wireframeVertexBuffer = device.createBuffer({
            label: 'FastBVH Wireframe Vertex Buffer',
            size: totalNodes * floatsPerNode * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
        });

        this.wireframeDepthBuffer = device.createBuffer({
            label: 'FastBVH Wireframe Depth Uniform Buffer',
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.wireframeDepthBuffer, 0, new Uint32Array([initialDepth]));

        this.wireframeShaderModule = device.createShaderModule({ label: 'Wireframe Shader Module', code: wireframeWGSL });
        this.wireframeBindGroupLayout = device.createBindGroupLayout({
            label: 'Wireframe Bind Group Layout',
            entries: [  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // internalNodes
                        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // leafAABBs
                        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // wireframeVerts
                        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // leafParents
                        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }]           // depthUniforms
        });
        this.wireframePipelineLayout = device.createPipelineLayout({ label: 'Wireframe Pipeline Layout', bindGroupLayouts: [this.wireframeBindGroupLayout] });
        this.wireframePipeline = device.createComputePipeline({
            label: 'Wireframe Pipeline',
            layout: this.wireframePipelineLayout,
            compute: {
                module: this.wireframeShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, INTERNAL_NODE_COUNT: this.numTriangles - 1, LEAF_NODE_COUNT: this.numTriangles }
            }
        });
        this.wireframeBindGroup = device.createBindGroup({
            label: 'Wireframe Bind Group',
            layout: this.wireframeBindGroupLayout,
            entries: [  { binding: 0, resource: { buffer: this.aabbInternalNodesBuffer } },
                        { binding: 1, resource: { buffer: this.leafAABBsBuffer } },
                        { binding: 2, resource: { buffer: this.wireframeVertexBuffer } },
                        { binding: 3, resource: { buffer: this.aabbLeafNodesBuffer } }, // leafParents (u32 per-leaf parent index)
                        { binding: 4, resource: { buffer: this.wireframeDepthBuffer } }]
        });
    }

    //================================//
    dispatchWireframePass(commandEncoder: GPUComputePassEncoder)
    {
        if (!this.wireframePipeline || !this.wireframeBindGroup) return;

        const totalNodes = (this.numTriangles - 1) + this.numTriangles;
        const dispatchX = Math.ceil(totalNodes / this.THREADS_PER_WORKGROUP);
        commandEncoder.setPipeline(this.wireframePipeline);
        commandEncoder.setBindGroup(0, this.wireframeBindGroup);
        commandEncoder.dispatchWorkgroups(dispatchX, 1, 1);
    }
}
//============== END PAPER IMPLEMENTATION ==================//

//============== START OF MAIN ARCHITECTURE (NOT PAPER IMPLEMENTATION) ==================//
export async function startup_14(canvas: HTMLCanvasElement)
{
    const renderer = new RayTracer();
    await renderer.initialize(canvas);
    
    return renderer;
}

//================================//
const normalUniformDataSize = (16 * 2) * 4 + (2 * 4) * 4 + (48 * 3);
const rayTracerUniformDataSize = 224 + 16*4 + 16;
const meshInstanceSize = 24 * 4;

//================================//
enum RayTracerMode
{
    raytrace = 0,
    BVHVisualization = 1,
    normal = 2,
    distance = 3,
    rayDirections = 4
}

//================================//
interface NO extends PipelineResources
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

interface RO extends PipelineResources
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

    // Utils for the FastBVH implementation
    worldPositionStorageBuffer: GPUBuffer;
    perMeshWorldPositionOffsets: number[];
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
    private NO: NO;
    private RO: RO;

    //================================//
    private useRaytracing: boolean = true;
    private rayTracerMode: RayTracerMode = RayTracerMode.raytrace;
    private numBounces: number = 3;
    private numSpheres: number = 100;
    private meshesInfo: any;
    private activeContextMenu: HTMLDivElement | null = null;
    private seed = 0;

    //================================//
    private showBVH: boolean = false;
    private showFastBVH: boolean = false;
    private bvhDepth: number = Infinity;
    private fastBVHDepth: number = 10;
    private minMaxBoundsText: string = '';

    //================================//
    private fastBVHIdentityBuffer: GPUBuffer | null = null;
    private fastBVHWireframeBindGroup: GPUBindGroup | null = null;

    //================================//
    private fastBVH: FastParallelBVH = new FastParallelBVH();

    //================================//
    constructor () 
    {
        setCameraPosition(this.camera, 0, 100, -200);
        rotateCameraBy(this.camera, 0, -0.5);
        setCameraNearFar(this.camera, 0.1, 2000);
        this.camera.moveSpeed = 5.0;
        this.camera.rotateSpeed = 0.02;
        this.device = null;
        this.NO = {} as NO;
        this.RO = {} as RO;

        const light1 = {
            position: glm.vec3.fromValues(0, 100, 0),
            intensity: 200.0,
            direction: glm.vec3.fromValues(0, -1, 0),
            coneAngle: Math.PI / 2,
            color: glm.vec3.fromValues(0.1, 0.1, 0.85),
            enabled: true
        };
        this.lights.push(light1);

        const light2 = {
            position: glm.vec3.fromValues(100.0, 100.0, 0), 
            intensity: 1000.0,
            direction: glm.vec3.fromValues(-1, -3, 0),
            coneAngle: Math.PI / 5,
            color: glm.vec3.fromValues(0.1, 0.85, 0.1),
            enabled: true
        };
        this.lights.push(light2);

        const light3 = {
            position: glm.vec3.fromValues(-100.0, 100.0, 0),
            intensity: 1000.0,
            direction: glm.vec3.fromValues(1, -3, 0),
            coneAngle: Math.PI / 5,
            color: glm.vec3.fromValues(0.85, 0.1, 0.1),
            enabled: true
        };
        this.lights.push(light3);
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
        addCheckbox('Show BVH', this.showBVH, utilElement, (value) => { this.showBVH = value; this.rayTracerMode = value ? RayTracerMode.BVHVisualization : RayTracerMode.raytrace; });
        utilElement.appendChild(document.createElement('br'));
        addCheckbox('Show FastBVH', this.showFastBVH, utilElement, (value) => { this.showFastBVH = value; });
        utilElement.appendChild(document.createElement('br'));
        addSlider('FastBVH Depth', this.fastBVHDepth, 1, 30, 1, utilElement, (value) => { this.fastBVHDepth = value; if (this.device) this.device.queue.writeBuffer(this.fastBVH.wireframeDepthBuffer, 0, new Uint32Array([value])); });
        utilElement.appendChild(document.createElement('br'));
        addSlider('BVH Depth', this.bvhDepth === Infinity ? 32 : this.bvhDepth, 1, 32, 1, utilElement, (value) => { this.bvhDepth = value === 32 ? Infinity : value; this.rebuildBVHBuffer(); });
        utilElement.appendChild(document.createElement('br'));
        addNumberInput('Random Seed', this.seed, 0, 10<<20, 1, utilElement, (value) => { this.seed = value; this.initializeBuffers(); });
        utilElement.appendChild(document.createElement('br'));
        addSlider('Number of Spheres', this.numSpheres, 1, 200, 1, utilElement, (value) => { this.numSpheres = value; this.initializeBuffers(); });
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

        if (!this.context) 
        {
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

        this.RO.shaderModule = CreateShaderModule(this.device, rayTraceVertWGSL, rayTraceFragWGSL, 'Ray Trace Shader Module');
        this.NO.shaderModule = CreateShaderModule(this.device, rasterVertWgsl, rasterFragWgsl, 'Normal Shader Module');
        this.NO.bvhShaderModule = CreateShaderModule(this.device, bvhVertWGSL, bvhFragWGSL, 'BVH Draw Shader Module');
    }

    //================================//
    initializePipelines()
    {
        if (this.device === null || this.presentationFormat === null) return;

        // RAY TRACE PIPELINE
        this.RO.bindGroupLayout = this.device.createBindGroupLayout({
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
        this.RO.materialBindGroupLayout = this.device.createBindGroupLayout({
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
                    targets: [
                        {
                            format: this.presentationFormat
                        }
                    ],
                }
            });
        }

        // NORMAL PIPELINE
        this.NO.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Normal Bind Group Layout',
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            }]
        });
        this.NO.materialUniformBindGroupLayout = this.device.createBindGroupLayout({
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

        this.NO.pipelineLayout = this.device.createPipelineLayout({
            label: 'Normal Pipeline Layout',
            bindGroupLayouts: [this.NO.bindGroupLayout, this.NO.materialUniformBindGroupLayout],
        });

        this.NO.bvhDrawPipelineLayout = this.device.createPipelineLayout({
            label: 'BVH Draw Pipeline Layout',
            bindGroupLayouts: [this.NO.bindGroupLayout, this.NO.materialUniformBindGroupLayout],
        });

        this.NO.depthTexture = this.device.createTexture({
            size: [this.canvas!.width, this.canvas!.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        if (this.NO.shaderModule !== null) {
            this.NO.pipeline = this.device.createRenderPipeline({
                label: 'Normal Pipeline',
                layout: this.NO.pipelineLayout,
                vertex: {
                    module: this.NO.shaderModule.vertex,
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
                    module: this.NO.shaderModule.fragment,
                    entryPoint: 'fs',
                    targets: [
                        {
                            format: this.presentationFormat
                        }
                    ],
                },
                primitive: {
                    topology: "triangle-list",
                    cullMode: "back",
                },
                    depthStencil: {
                    format: "depth24plus",
                    depthWriteEnabled: true,
                    depthCompare: "less",
                },
            });

            this.NO.bvhDrawPipeline = this.device.createRenderPipeline({
                label: 'BVH Draw Pipeline',
                layout: this.NO.bvhDrawPipelineLayout,
                vertex: {
                    module: this.NO.bvhShaderModule!.vertex,
                    entryPoint: 'vsBVH',
                    buffers: [
                        {
                            arrayStride: 3*4,
                            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }]
                        }
                    ]
                },
                fragment: {
                    module: this.NO.bvhShaderModule!.fragment,
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
                    format: "depth24plus",
                    depthWriteEnabled: false,
                    depthCompare: "less",
                },
            });
        }

        this.timestampQuerySet = CreateTimestampQuerySet(this.device, 2);

        // Samplers
        this.NO.sampler = this.device.createSampler({
            label: 'Normal Objects Sampler',
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
            addressModeU: "repeat",
            addressModeV: "repeat",
        });

        this.RO.sampler = this.device.createSampler({
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
        const info: SceneInformation = await fastBVHExampleScene(meshMaterials, this.seed, this.numSpheres);
        this.NO.sceneInformation = info;
        this.meshesInfo = info.additionalInfo;
        const worldPositionData: Float32Array = this.meshesInfo.worldPositionData;
        this.RO.perMeshWorldPositionOffsets = this.meshesInfo.perMeshWorldPositionOffsets;

        const numMaterials = info.meshes.length;
        this.NO.materialUniforms = [];
        this.NO.materialBindGroups = [];
        this.NO.positionBuffers = [];
        this.NO.normalBuffers = [];
        this.NO.uvBuffers = [];
        this.NO.indexBuffers = [];

        this.NO.meshesModelMatrixBuffers = [];
        this.NO.meshesNormalMatrixBuffers = [];

        for (let matNum = 0; matNum < numMaterials; matNum++)
        {
            this.NO.meshesModelMatrixBuffers.push(this.device.createBuffer({
                label: 'Mesh Model Matrix Buffer ' + matNum,
                size: 16 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.NO.meshesModelMatrixBuffers[matNum], 0, info.meshes[matNum].GetFlatWorldMatrix() as BufferSource);

            this.NO.meshesNormalMatrixBuffers.push(this.device.createBuffer({
                label: 'Mesh Normal Matrix Buffer ' + matNum,
                size: 16 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.NO.meshesNormalMatrixBuffers[matNum], 0, info.meshes[matNum].GetFlatNormalMatrix() as BufferSource);

            this.NO.materialUniforms.push(this.device.createBuffer({
                label: 'Material Uniform Buffer ' + matNum,
                size: MATERIAL_SIZE * 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            }));

            const materialData = info.meshes[matNum].GetFlattenedMaterial();
            this.device.queue.writeBuffer(this.NO.materialUniforms[matNum], 0, materialData as BufferSource);

            this.NO.materialBindGroups.push(this.device.createBindGroup({
                label: 'Material Bind Group ' + matNum,
                layout: this.NO.materialUniformBindGroupLayout,
                entries: [{
                    binding: 0,
                    resource: { buffer: this.NO.materialUniforms[matNum] },
                },
                {
                    binding: 1,
                    resource: this.NO.sampler,
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
                    resource: { buffer: this.NO.meshesModelMatrixBuffers[matNum] },
                },
                {
                        binding: 7,
                        resource: { buffer: this.NO.meshesNormalMatrixBuffers[matNum] },
                }],
            }));

            const vertData = info.meshes[matNum].getVertexData();
            this.NO.positionBuffers.push(this.device.createBuffer({
                label: 'Normal Position Buffer ' + matNum,
                size: vertData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.NO.positionBuffers[matNum], 0, vertData as BufferSource);

            const indexData = info.meshes[matNum].getIndexData16();
            this.NO.indexBuffers.push(this.device.createBuffer({
                label: 'Normal Index Buffer ' + matNum,
                size: indexData.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.NO.indexBuffers[matNum], 0, indexData as BufferSource);

            const normalData = info.meshes[matNum].getNormalData();
            this.NO.normalBuffers.push(this.device.createBuffer({
                label: 'Normal Normal Buffer ' + matNum,
                size: normalData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.NO.normalBuffers[matNum], 0, normalData as BufferSource);

            const uvData = info.meshes[matNum].getUVData();
            this.NO.uvBuffers.push(this.device.createBuffer({
                label: 'Normal UV Buffer ' + matNum,
                size: uvData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            }));
            this.device.queue.writeBuffer(this.NO.uvBuffers[matNum], 0, uvData as BufferSource);
        }

        this.NO.uniformBuffer = this.device.createBuffer({
            label: 'Normal Uniform Buffer',
            size: normalUniformDataSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.NO.bindGroup = this.device.createBindGroup({
            label: 'Normal Bind Group',
            layout: this.NO.bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.NO.uniformBuffer },
            }],
        });

        const lineData: Float32Array[] = this.getBVHGeometry(Infinity); // This way create buffer at max capacity
        this.NO.bvhLineGeometryBuffers = [];
        for (let i = 0; i < lineData.length; i++)
        {
            this.NO.bvhLineGeometryBuffers[i] = this.device.createBuffer({
                label: `BVH Line Geometry Buffer ${i}`,
                size: lineData[i].byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(this.NO.bvhLineGeometryBuffers[i], 0, lineData[i] as BufferSource);
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
            uint32View[20] = bvhNodeCount;
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

        this.RO.uniformBuffer = this.device.createBuffer({
            label: 'Ray Tracer Uniform Buffer',
            size: rayTracerUniformDataSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.RO.positionStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Position Storage Buffer',
            size: positionData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.positionStorageBuffer, 0, positionData as BufferSource);

        this.RO.worldPositionStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer World Position Storage Buffer',
            size: worldPositionData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.worldPositionStorageBuffer, 0, worldPositionData as BufferSource);

        this.RO.normalStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Normal Storage Buffer',
            size: normalData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.normalStorageBuffer, 0, normalData as BufferSource);

        this.RO.uvStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer UV Storage Buffer',
            size: uvData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.uvStorageBuffer, 0, uvData as BufferSource);

        this.RO.indexStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Index Storage Buffer',
            size: indexData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.indexStorageBuffer, 0, indexData as BufferSource);

        this.RO.bvhNodesStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer BVH Nodes Storage Buffer',
            size: bvhNodesData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.bvhNodesStorageBuffer, 0, bvhNodesData as BufferSource);

        this.RO.meshInstancesStorageBuffer = this.device.createBuffer({
            label: 'Ray Tracer Mesh Instances Storage Buffer',
            size: meshInstancesData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.meshInstancesStorageBuffer, 0, meshInstancesData as BufferSource);

        this.RO.bindGroup = this.device.createBindGroup({
            label: 'Ray Tracer Bind Group',
            layout: this.RO.bindGroupLayout,
            entries: [{
                    binding: 0,
                    resource: { buffer: this.RO.uniformBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: this.RO.positionStorageBuffer },
                },
                {
                    binding: 2,
                    resource: { buffer: this.RO.normalStorageBuffer },
                },
                {
                    binding: 3,
                    resource: { buffer: this.RO.uvStorageBuffer },
                },
                {
                    binding: 4,
                    resource: { buffer: this.RO.indexStorageBuffer },
                },
                {
                    binding: 5,
                    resource: { buffer: this.RO.bvhNodesStorageBuffer },
                },
                {
                    binding: 6,
                    resource: { buffer: this.RO.meshInstancesStorageBuffer },
                }
            ],
        });

        // FastBVH Pipeline
        this.fastBVH.initializeMinMaxPipeline(this.device, this.RO.worldPositionStorageBuffer, worldPositionData.length / 3);
        const numTriangles = indexData.length / 3;
        this.fastBVH.initializeMortonPipeline(this.device, this.RO.worldPositionStorageBuffer, this.RO.indexStorageBuffer, numTriangles);
        this.fastBVH.initializeRadixSortPipelines(this.device);
        this.fastBVH.initializePatriciaTreePipeline(this.device);
        this.fastBVH.initializeAABBPipeline(this.device, this.RO.worldPositionStorageBuffer, this.RO.indexStorageBuffer);
        this.fastBVH.initializeWireframePipeline(this.device, this.fastBVHDepth);

        // Identity matrix buffer
        const identityMat = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
        this.fastBVHIdentityBuffer = this.device.createBuffer({
            label: 'FastBVH Identity Matrix Buffer',
            size: 16 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.fastBVHIdentityBuffer, 0, identityMat);

        const placeholderTex = createPlaceholderTexture(this.device, 1024, 32);
        this.fastBVHWireframeBindGroup = this.device.createBindGroup({
            label: 'FastBVH Wireframe Draw Bind Group',
            layout: this.NO.materialUniformBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.NO.materialUniforms[0] } },
                { binding: 1, resource: this.NO.sampler },
                { binding: 2, resource: placeholderTex.createView() },
                { binding: 3, resource: placeholderTex.createView() },
                { binding: 4, resource: placeholderTex.createView() },
                { binding: 5, resource: placeholderTex.createView() },
                { binding: 6, resource: { buffer: this.fastBVHIdentityBuffer } },
                { binding: 7, resource: { buffer: this.fastBVHIdentityBuffer } },
            ],
        });

        // material buffer for ray tracer
        const materials = info.meshes.map(mesh => mesh.Material);
        const materialData = flattenMaterialArray(materials);
        this.RO.materialBuffer = this.device.createBuffer({
            label: 'Ray Tracer Material Storage Buffer',
            size: materialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.RO.materialBuffer, 0, materialData as BufferSource);

        // Texture array creation
        const numTexturesPerMaterial = 4;
        var numTexturedMaterials = this.meshesInfo?.meshMaterials.filter((mat: Material) => mat.textureIndex >= 0).length || 0;
        if (numTexturedMaterials === 0) numTexturedMaterials = 1;

        const commonW = 1024;
        const commonH = 1024;

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
        for (let matNum = 0; matNum < numTexturedMaterials; matNum++)
        {
            const albedoImage = this.meshesInfo?.meshMaterials[matNum].albedoImage ? this.meshesInfo.meshMaterials[matNum].albedoImage : placeHolderImage;
            const metalnessImage = this.meshesInfo?.meshMaterials[matNum].metalnessImage ? this.meshesInfo.meshMaterials[matNum].metalnessImage : placeHolderImage;
            const roughnessImage = this.meshesInfo?.meshMaterials[matNum].roughnessImage ? this.meshesInfo.meshMaterials[matNum].roughnessImage : placeHolderImage;
            const normalImage = this.meshesInfo?.meshMaterials[matNum].normalImage ? this.meshesInfo.meshMaterials[matNum].normalImage : placeHolderImage;

            this.device.queue.copyExternalImageToTexture(
                { source: albedoImage },
                { texture: this.RO.textureArray, origin: [0, 0, matNum * numTexturesPerMaterial] },
                [commonW, commonH]
            );
            this.device.queue.copyExternalImageToTexture(
                { source: metalnessImage },
                { texture: this.RO.textureArray, origin: [0, 0, matNum * numTexturesPerMaterial + 1] },
                [commonW, commonH]
            );
            this.device.queue.copyExternalImageToTexture(
                { source: roughnessImage },
                { texture: this.RO.textureArray, origin: [0, 0, matNum * numTexturesPerMaterial + 2] },
                [commonW, commonH]
            );
            this.device.queue.copyExternalImageToTexture(
                { source: normalImage },
                { texture: this.RO.textureArray, origin: [0, 0, matNum * numTexturesPerMaterial + 3] },
                [commonW, commonH]
            );
        }

        this.RO.materialBindGroup = this.device.createBindGroup({
            label: 'Ray Tracer Material Bind Group',
            layout: this.RO.materialBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.RO.materialBuffer },
            },
            {
                binding: 1,
                resource: this.RO.sampler,
            },
            {
                binding: 2,
                resource: this.RO.textureArray.createView(),
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
            uintView[19] =  this.rayTracerMode;
            floatView[20] = this.a_c;
            floatView[21] = this.a_l;
            floatView[22] = this.a_q;
            floatView[23] = this.bvhDepth;

            uintView[24] = this.numBounces;
            floatView[25] = 0.0;
            floatView[26] = 0.0;
            floatView[27] = 0.0;

            // All lights
            for (let i = 0; i < 3; i++)
            {
                if (i >= this.lights.length)
                    break;

                const light = this.lights[i];
                const baseIndex = 28 + i * 12; // Each light is in total 48 bytes, 12 floats

                floatView.set(light.position, baseIndex);
                floatView[baseIndex + 3] = light.intensity;

                floatView.set(light.direction, baseIndex + 4);
                floatView[baseIndex + 7] = light.coneAngle;

                floatView.set(light.color, baseIndex + 8);
                floatView[baseIndex + 11] = light.enabled ? 1.0 : 0.0;
                // pad
            }
            this.device.queue.writeBuffer(this.RO.uniformBuffer, 0, data);
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
            this.device.queue.writeBuffer(this.NO.uniformBuffer, 0, data as BufferSource);
        }
    }

    //================================//
    animate()
    {
        const meshes = this.NO.sceneInformation.meshes;
        const totalMeshes = meshes.length;
        const time = performance.now() * 0.001;

        let s = (this.seed + 777) | 0;
        const random = (): number => 
        {
            s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        const randomRange = (min: number, max: number) => random() * (max - min) + min;
        const maxAnimated = Math.min(10, totalMeshes - 1);

        for (let i = 0; i < maxAnimated; i++)
        {
            const meshIndex = i + 1;
            const mesh = meshes[meshIndex];

            const orbitRadius = randomRange(20, 80);
            const speed = randomRange(0.3, 1.5);
            const phaseOffset = randomRange(0, Math.PI * 2);
            const pattern = random();

            const baseY = mesh.GetTransform().scale[0];
            const baseX = randomRange(-60, 60);
            const baseZ = randomRange(-60, 60);

            let x: number, y: number, z: number;
            const t = time * speed + phaseOffset;

            if (pattern < 0.25)
            {
                x = baseX + orbitRadius * Math.sin(t);
                z = baseZ + orbitRadius * Math.sin(t) * Math.cos(t);
                y = baseY;
            }
            else if (pattern < 0.5)
            {
                const eccentricity = randomRange(0.3, 0.8);
                x = baseX + orbitRadius * Math.cos(t);
                z = baseZ + orbitRadius * eccentricity * Math.sin(t);
                y = baseY + Math.abs(Math.sin(t * 2.0)) * 15.0;
            }
            else if (pattern < 0.75)
            {
                const axis = randomRange(0, Math.PI * 2);
                const dist = Math.sin(t) * orbitRadius;
                x = baseX + Math.cos(axis) * dist;
                z = baseZ + Math.sin(axis) * dist;
                y = baseY + Math.abs(Math.sin(t * 1.5)) * 8.0;
            }
            else
            {
                const spiralRadius = orbitRadius * (0.5 + 0.5 * Math.sin(t * 0.3));
                x = baseX + spiralRadius * Math.cos(t);
                z = baseZ + spiralRadius * Math.sin(t);
                y = baseY;
            }

            mesh.SetTranslation(glm.vec3.fromValues(x, y, z));

            this.device?.queue.writeBuffer(this.NO.meshesModelMatrixBuffers[meshIndex], 0, mesh.GetFlatWorldMatrix() as BufferSource);
            this.device?.queue.writeBuffer(this.NO.meshesNormalMatrixBuffers[meshIndex], 0, mesh.GetFlatNormalMatrix() as BufferSource);
            this.device?.queue.writeBuffer(this.RO.meshInstancesStorageBuffer, meshIndex * meshInstanceSize, mesh.GetFlatInverseWorldMatrix() as BufferSource);

            const worldPositions = mesh.getWorldVertexData();
            const meshOffset = this.RO.perMeshWorldPositionOffsets[meshIndex];
            this.device?.queue.writeBuffer(this.RO.worldPositionStorageBuffer, meshOffset, worldPositions as BufferSource);
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
            this.animate();

            const textureView = this.context.getCurrentTexture().createView();
            const depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined = !this.useRaytracing ? {
                view: this.NO.depthTexture.createView(),
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

            this.fastBVH.clearAtomicCounters(encoder);
            const computePass = encoder.beginComputePass({ label: 'Fast parallel BVH Compute Pass' });
            this.fastBVH.dispatch(computePass);
            computePass.end();

            if (this.fastBVH.minMaxReadbackBuffer?.mapState === 'unmapped' && this.fastBVH.debug)
                this.fastBVH.copyResultForReadback(encoder);

            const pass = encoder.beginRenderPass(renderPassDescriptor);

            if (this.useRaytracing)
            {
                pass.setPipeline(this.RO.pipeline);
                pass.setBindGroup(0, this.RO.bindGroup);
                pass.setBindGroup(1, this.RO.materialBindGroup);
                pass.draw(6); // Fullscreen quad
            }
            else
            {
                pass.setPipeline(this.NO.pipeline);
                pass.setBindGroup(0, this.NO.bindGroup);
                for (let matNum = 0; matNum < this.NO.sceneInformation.meshes.length; matNum++)
                {
                    pass.setBindGroup(1, this.NO.materialBindGroups[matNum]);
                    pass.setVertexBuffer(0, this.NO.positionBuffers[matNum]);
                    pass.setVertexBuffer(1, this.NO.normalBuffers[matNum]);
                    pass.setVertexBuffer(2, this.NO.uvBuffers[matNum]);
                    pass.setIndexBuffer(this.NO.indexBuffers[matNum], 'uint16');

                    pass.drawIndexed(this.NO.indexBuffers[matNum].size / 2, 1, 0, 0, 0);
                }

                // CPU BVH Rendering
                if (this.showBVH)
                {
                    pass.setPipeline(this.NO.bvhDrawPipeline);
                    pass.setBindGroup(0, this.NO.bindGroup);
                    for (let i = 0; i < this.NO.bvhLineGeometryBuffers.length; i++)
                    {
                        pass.setBindGroup(1, this.NO.materialBindGroups[i]);
                        pass.setVertexBuffer(0, this.NO.bvhLineGeometryBuffers[i]);
                        pass.draw(this.NO.bvhLineCounts[i]);
                    }
                }

                // FastBVH (GPU-built) Wireframe Rendering
                if (this.showFastBVH && this.fastBVHWireframeBindGroup)
                {
                    pass.setPipeline(this.NO.bvhDrawPipeline);
                    pass.setBindGroup(0, this.NO.bindGroup);
                    pass.setBindGroup(1, this.fastBVHWireframeBindGroup);
                    pass.setVertexBuffer(0, this.fastBVH.wireframeVertexBuffer);
                    pass.draw(this.fastBVH.wireframeVertexCount);
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

            if (this.fastBVH.minMaxReadbackBuffer?.mapState === 'unmapped' && this.fastBVH.debug)
            {
                this.fastBVH.minMaxReadbackBuffer.mapAsync(GPUMapMode.READ).then(() =>
                {
                    const d = new Float32Array(this.fastBVH.minMaxReadbackBuffer!.getMappedRange());
                    this.minMaxBoundsText = `Min: (${d[0].toFixed(1)}, ${d[1].toFixed(1)}, ${d[2].toFixed(1)}) Max: (${d[3].toFixed(1)}, ${d[4].toFixed(1)}, ${d[5].toFixed(1)})`;
                    this.fastBVH.minMaxReadbackBuffer!.unmap();
                });
            }

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
                Num Triangles: ${this.fastBVH.numTriangles}
                ${this.fastBVH.debug ? this.minMaxBoundsText : ''}
                `
                this.infoElement.textContent = content;
            }

            if (1000/dt <= 300) 
                addProfilerFrameTime(1000/dt);

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
                    if (this.NO.depthTexture) {
                        this.NO.depthTexture.destroy();
                        this.NO.depthTexture = this.device.createTexture({
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
        const totalMaterialIndex = this.NO.sceneInformation.meshes.findIndex(mesh => mesh.Material.name === matName);
        if (totalMaterialIndex === -1) return;

        this.meshesInfo!.meshMaterials[meshIndex] = newMaterial;
        this.NO.sceneInformation.meshes[totalMaterialIndex].Material = newMaterial;

        const materialBufferIndex = this.meshesInfo!.meshIndices[meshIndex];
        const materialData = flattenMaterial(newMaterial);
        
        // NORMAL
        let buffer = this.NO.materialUniforms[materialBufferIndex];
        this.device!.queue.writeBuffer(buffer, 0, materialData as BufferSource);

        // RAY TRACING
        const offset = materialBufferIndex * MATERIAL_SIZE * 4;
        this.device!.queue.writeBuffer(this.RO.materialBuffer, offset, materialData as BufferSource);
    }

    //================================//
    recreateBindGroup(material: Material)
    {
        const matName: string = material.name;
        const totalMaterialIndex = this.NO.sceneInformation.meshes.findIndex(mesh => mesh.Material.name === matName);
        if (totalMaterialIndex === -1) return;

        const newBindGroup = this.device!.createBindGroup({
            label: 'Material Bind Group ' + totalMaterialIndex,
            layout: this.NO.materialUniformBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.NO.materialUniforms[totalMaterialIndex] },
            },
            {
                binding: 1,
                resource: this.NO.sampler,
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
                    resource: { buffer: this.NO.meshesModelMatrixBuffers[totalMaterialIndex] },
            },
            {
                    binding: 7,
                    resource: { buffer: this.NO.meshesNormalMatrixBuffers[totalMaterialIndex] },
            }],
        });

        this.NO.materialBindGroups[totalMaterialIndex] = newBindGroup;

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
                { texture: this.RO.textureArray, origin: [0, 0, index * 4 + typeIndex] },
                [1024, 1024]
            );
        }
    }

    //================================//
    getBVHGeometry(desiredDepth: number): Float32Array[]
    {
        if (this.NO.sceneInformation.meshes.length === 0) return [];

        this.NO.bvhLineCounts = [];
        const chunks: Float32Array[] = [];
        for (let matNum = 0; matNum < this.NO.sceneInformation.meshes.length; matNum++)
        {
            const { vertexData, count } = this.NO.sceneInformation.meshes[matNum].GetBVHGeometry(desiredDepth);
            chunks.push(vertexData);
            this.NO.bvhLineCounts.push(count);
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
            this.NO.bvhLineGeometryBuffers[i] = this.device.createBuffer({
                label: `BVH Line Geometry Buffer ${i}`,
                size: lineData[i].byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(this.NO.bvhLineGeometryBuffers[i], 0, lineData[i] as BufferSource);
        }
    }

    //================================//
    rayCastOnMeshes(screenX: number, screenY: number): number
    {
        if (this.canvas === null || this.camera === null || this.meshesInfo === null) return -1;

        const potentialMeshesIndices: number[] = this.meshesInfo!.meshIndices;
        const potentialMeshes = potentialMeshesIndices.map(index => this.NO.sceneInformation.meshes[index]);

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
//============== END OF MAIN ARCHITECTURE (NOT PAPER IMPLEMENTATION) ==================//