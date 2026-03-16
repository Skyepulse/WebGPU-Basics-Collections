import { RequestWebGPUDevice, CreateTimestampQuerySet, CreateShaderModule } from '@src/helpers/WebGPUutils';
import type { ComputePipelineResources, PipelineResources, TimestampQuerySet } from '@src/helpers/WebGPUutils';
import { addButton, addCheckbox, addNumberInput, cleanUtilElement, getInfoElement, getUtilElement } from '@src/helpers/Others';

//================================//
import radixSortWGSL from './radixSort.wgsl?raw';
import prefixSumWGSL from './prefixSum.wgsl?raw';
import reorderWGSL from './reorder.wgsl?raw';

import render_vertWGSL from './vert.wgsl?raw';
import render_fragWGSL from './frag.wgsl?raw';

//================================//
export async function startup_13(canvas: HTMLCanvasElement)
{
    const renderer = new ComputeRenderer();
    await renderer.initialize(canvas);
    
    return renderer;
}

//================================//
interface RenderResources extends PipelineResources
{
    inputArrayBuffer: GPUBuffer;
}

//================================//
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

//================================//
class ComputeRenderer
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

    private desiredNewElementCount: number = 1 << 20;

    //================================//
    /* CONSTANTS TO OVERRIDE IN SHADERS */
    private THREADS_PER_WORKGROUP: number = 256;
    private X_SIZE: number = 16;
    private Y_SIZE: number = 16;
    private ELEMENT_COUNT: number = 1 << 20;
    private ITEMS_PER_WORKGROUP: number = 2 * this.THREADS_PER_WORKGROUP;
    private readonly BIT_COUNT: number = 30;
    private readonly NUM_PASSES: number = this.BIT_COUNT / 2;

    private GRID_SIZE!: number;
    private WORKGROUP_COUNT!: number; // MATH.CEIL(ELEMENT_COUNT / THREADS_PER_WORKGROUP)

    //================================//
    /* A list of unordered or then ordered elements (30 bit integers max) */
    private elements: number[] = [];
    private radixSort: boolean = true;
    private sortFlag: boolean = false;
    private sortedThisFrame: boolean = false;
    private lastSortTime: number = 0;

    //================================//
    // PIPELINES AND RESOURCES
    //================================//
    private radixSortResources: ComputePipelineResources    = {} as ComputePipelineResources;
    private reorderResources: ComputePipelineResources      = {} as ComputePipelineResources;
    private renderResources: RenderResources                = {} as RenderResources;

    private prefixSumBindGroupLayout!: GPUBindGroupLayout;
    private prefixSumLevels: PrefixSumLevel[] = [];

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

    private radixDispatchX!: number;
    private radixDispatchY!: number;

    //================================//
    constructor () 
    {
        this.device = null;

        this.computeConstants();
    }

    //================================//
    computeConstants()
    {
        this.WORKGROUP_COUNT = Math.ceil(this.ELEMENT_COUNT / this.THREADS_PER_WORKGROUP);
        this.GRID_SIZE = Math.ceil(Math.sqrt(this.ELEMENT_COUNT));

        const [dx, dy] = this.dispatchSize(this.WORKGROUP_COUNT);
        this.radixDispatchX = dx;
        this.radixDispatchY = dy;
    }

    //================================//
    initializeUtils()
    {
        const utilElement = getUtilElement();
        if (!utilElement) return;

        
        addNumberInput('Element Count', this.ELEMENT_COUNT, 0, 1 << 30, 10, utilElement, (value) =>
        {            
            this.desiredNewElementCount = value;
            this.resizeElementCount(this.desiredNewElementCount);
        });
        utilElement.appendChild(document.createElement('br'));
        addButton('Randomize', utilElement, () => 
        {
            this.shuffle();
        });
        utilElement.appendChild(document.createElement('br'));
        addCheckbox('Radix Sort', this.radixSort, utilElement, (value) => this.radixSort = value);
        utilElement.appendChild(document.createElement('br'));
        addButton('Sort', utilElement, () => 
        {
            this.sortFlag = true;
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

        this.radixSortResources.shaderModule    = this.device.createShaderModule({ label: 'Radix Sort Shader Module', code: radixSortWGSL });
        this.reorderResources.shaderModule      = this.device.createShaderModule({ label: 'Reorder Shader Module', code: reorderWGSL });

        this.renderResources.shaderModule       = CreateShaderModule(this.device, render_vertWGSL, render_fragWGSL, 'Render Shader Module');
    }

    //================================//
    initializePipelines()
    {
        if (this.device === null || this.presentationFormat === null) return;

        // Uniform bind group layout
        this.uniformBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Uniform Bind Group Layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        // Prefix sum bind group layout
        this.prefixSumBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Prefix Sum Bind Group Layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' } // data to scan / add to
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' } // block sums
                }
            ]
        });

        // [1] RADIX SORT PIPELINE
        this.radixSortResources.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Radix Sort Bind Group Layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' } // input array
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' } // local prefix sum array
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' } // block sum array
                }
            ]
        });
        this.radixSortResources.pipelineLayout = this.device.createPipelineLayout({
            label: 'Radix Sort Pipeline Layout',
            bindGroupLayouts: [this.radixSortResources.bindGroupLayout, this.uniformBindGroupLayout]
        });
        this.radixSortResources.pipeline = this.device.createComputePipeline({
            label: 'Radix Sort Compute Pipeline',
            layout: this.radixSortResources.pipelineLayout,
            compute: {
                module: this.radixSortResources.shaderModule!,
                entryPoint: 'cs',
                constants: 
                {
                    THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP,
                    X_SIZE: this.X_SIZE,
                    Y_SIZE: this.Y_SIZE,
                    ELEMENT_COUNT: this.ELEMENT_COUNT,
                    WORKGROUP_COUNT: this.WORKGROUP_COUNT
                }
            },
        });

        // [2] REORDER PIPELINE
        this.reorderResources.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Reorder Bind Group Layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' } // input keys
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' } // output keys (sorted)
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' } // local prefix sum array
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' } // block sum array
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' } // input values (IN CASE OF KEY-VALUE SORTING)
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' } // output values (sorted, IN CASE OF KEY-VALUE SORTING)
                }
            ]
        });
        this.reorderResources.pipelineLayout = this.device.createPipelineLayout({
            label: 'Reorder Pipeline Layout',
            bindGroupLayouts: [this.reorderResources.bindGroupLayout, this.uniformBindGroupLayout]
        });
        this.reorderResources.pipeline = this.device.createComputePipeline({
            label: 'Reorder Compute Pipeline',
            layout: this.reorderResources.pipelineLayout,
            compute: {
                module: this.reorderResources.shaderModule!,
                entryPoint: 'cs',
                constants: 
                {
                    THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP,
                    X_SIZE: this.X_SIZE,
                    Y_SIZE: this.Y_SIZE,
                    ELEMENT_COUNT: this.ELEMENT_COUNT,
                    WORKGROUP_COUNT: this.WORKGROUP_COUNT
                }
            }
        });

        // [3] RENDER PIPELINE
        this.renderResources.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Render Bind Group Layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: 'read-only-storage' } // input array
                }
            ]
        });
        this.renderResources.pipelineLayout = this.device.createPipelineLayout({
            label: 'Render Pipeline Layout',
            bindGroupLayouts: [this.renderResources.bindGroupLayout]
        });
        this.renderResources.pipeline = this.device.createRenderPipeline({
            label: 'Render Pipeline',
            layout: this.renderResources.pipelineLayout,
            vertex: {
                module: this.renderResources.shaderModule!.vertex,
                entryPoint: 'vs',
                constants:
                {
                    ELEMENT_COUNT: this.ELEMENT_COUNT,
                    GRID_SIZE: this.GRID_SIZE
                }
            },
            fragment: {
                module: this.renderResources.shaderModule!.fragment,
                entryPoint: 'fs',
                targets: [
                    {
                        format: this.presentationFormat
                    }
                ]
            },
            primitive: {
                topology: 'triangle-list',
            }
        });

        this.timestampQuerySet = CreateTimestampQuerySet(this.device, 4);
    }

    //================================//
    // Build the hierarchical prefix sum: pipelines, buffers, and bind groups.
    //
    // The block_sums buffer from the radix sort has 4 * WORKGROUP_COUNT entries.
    // Each Blelloch workgroup can scan ITEMS_PER_WORKGROUP (512) entries.
    //
    // Larger than that, a recursive hierarchical approach is needed to sweep
    // through those levels and then add.
    //================================//
    initializePrefixSum(topLevelDataBuffer: GPUBuffer, topLevelElementCount: number)
    {
        if (this.device === null) return;

        this.prefixSumLevels = [];
        const shaderModule = this.device.createShaderModule({ label: 'Prefix Sum Shader Module', code: prefixSumWGSL });
        const pipelineLayout = this.device.createPipelineLayout({
            label: 'Prefix Sum Pipeline Layout',
            bindGroupLayouts: [this.prefixSumBindGroupLayout]
        });

        let currentElementCount = topLevelElementCount;
        let currentDataBuffer = topLevelDataBuffer;

        while(true)
        {
            const workgroupCount = Math.ceil(currentElementCount / this.ITEMS_PER_WORKGROUP);
            const [dx, dy] = this.dispatchSize(workgroupCount);

            const blockSumBuffer = this.device.createBuffer({
                label: `Block Sum Buffer (Level ${this.prefixSumLevels.length})`,
                size: Math.max(workgroupCount, 1) * 4,
                usage: GPUBufferUsage.STORAGE
            });

            const reducePipeline = this.device.createComputePipeline({
                label: `Prefix Sum Reduce (level ${this.prefixSumLevels.length})`,
                layout: pipelineLayout,
                compute: {
                    module: shaderModule,
                    entryPoint: 'cs_reduce',
                    constants: {
                        THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP,
                        X_SIZE: this.X_SIZE,
                        Y_SIZE: this.Y_SIZE,
                        ITEMS_PER_WORKGROUP: this.ITEMS_PER_WORKGROUP,
                        ELEMENT_COUNT: currentElementCount,
                    }
                }
            });
 
            const addPipeline = this.device.createComputePipeline({
                label: `Prefix Sum Add (level ${this.prefixSumLevels.length})`,
                layout: pipelineLayout,
                compute: {
                    module: shaderModule,
                    entryPoint: 'cs_add',
                    constants: {
                        THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP,
                        X_SIZE: this.X_SIZE,
                        Y_SIZE: this.Y_SIZE,
                        ITEMS_PER_WORKGROUP: this.ITEMS_PER_WORKGROUP,
                        ELEMENT_COUNT: currentElementCount,
                    }
                }
            });

            const bindGroup = this.device.createBindGroup({
                label: `Prefix Sum Bind Group (level ${this.prefixSumLevels.length})`,
                layout: this.prefixSumBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: currentDataBuffer } },
                    { binding: 1, resource: { buffer: blockSumBuffer } }
                ]
            });

            this.prefixSumLevels.push({
                elementCount: currentElementCount,
                workgroupCount,
                reducePipeline,
                addPipeline,
                bindGroup,
                dataBuffer: currentDataBuffer,
                blockSumBuffer,
                dispatchX: dx,
                dispatchY: dy
            });

            if (workgroupCount <= 1) break; // No more levels needed

            currentDataBuffer = blockSumBuffer;
            currentElementCount = workgroupCount;
        }
    }

    //================================//
    async initializeBuffers()
    {
        if (this.device === null) return;

        // SHARED BUFFERS
        this.keysBufferA = this.device.createBuffer(
            {
                label: 'Keys Buffer A',
                size: this.ELEMENT_COUNT * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }
        )
        this.keysBufferB = this.device.createBuffer(
            {
                label: 'Keys Buffer B',
                size: this.ELEMENT_COUNT * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }
        );
        this.shuffle();

        this.valuesBufferA = this.device.createBuffer(
            {
                label: 'Values A',
                size: this.ELEMENT_COUNT * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }
        );
        this.valuesBufferB = this.device.createBuffer(
            {
                label: 'Values B',
                size: this.ELEMENT_COUNT * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }
        );

        this.localPrefixSumBuffer = this.device.createBuffer(
            {
                label: 'Local Prefix Sum Buffer',
                size: this.ELEMENT_COUNT * 4,
                usage: GPUBufferUsage.STORAGE
            }
        );

        this.blockSumBuffer = this.device.createBuffer(
            {
                label: 'Block Sum Buffer',
                size: 4 * this.WORKGROUP_COUNT * 4,
                usage: GPUBufferUsage.STORAGE
            }
        );
        const prefixSumElementCount = 4 * this.WORKGROUP_COUNT;

        // this is the top level buffer for the prefix sum.
        // In prefix sum level pipelines we will sweep over all buffers, writing out 
        this.initializePrefixSum(this.blockSumBuffer, prefixSumElementCount);

        // UNIFORM BUFFERS
        this.uniformBuffers = [];
        this.uniformBindGroups = [];
        for (let pass = 0; pass < 15; pass++) 
        {
            const buf = this.device.createBuffer({
                label: `Uniform Buffer pass ${pass}`,
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(buf, 0, new Uint32Array([pass * 2])); // 0, 2, ... 30 for up to 32 bit morton code sort
            this.uniformBuffers.push(buf);

            const bg = this.device!.createBindGroup({
                label: `Uniform Bind Group for pass ${pass}`,
                layout: this.uniformBindGroupLayout,
                entries: [{ binding: 0, resource: { buffer: buf } }]
            });
            this.uniformBindGroups.push(bg);
        }

        // BIND GROUPS
        this.radixSortBindGroups = [
            this.device.createBindGroup({
                label: 'Radix Sort Bind Group (A)',
                layout: this.radixSortResources.bindGroupLayout!,
                entries: [
                    { binding: 0, resource: { buffer: this.keysBufferA } },
                    { binding: 1, resource: { buffer: this.localPrefixSumBuffer } },
                    { binding: 2, resource: { buffer: this.blockSumBuffer } }
                ]
            }),
            this.device.createBindGroup({
                label: 'Radix Sort Bind Group (B)',
                layout: this.radixSortResources.bindGroupLayout!,
                entries: [
                    { binding: 0, resource: { buffer: this.keysBufferB } },
                    { binding: 1, resource: { buffer: this.localPrefixSumBuffer } },
                    { binding: 2, resource: { buffer: this.blockSumBuffer } }
                ]
            }),
        ];

        this.reorderBindGroups = [
            this.device.createBindGroup({
                label: 'Reorder Bind Group (A -> B)',
                layout: this.reorderResources.bindGroupLayout!,
                entries: [
                    { binding: 0, resource: { buffer: this.keysBufferA } },
                    { binding: 1, resource: { buffer: this.keysBufferB } },
                    { binding: 2, resource: { buffer: this.localPrefixSumBuffer } },
                    { binding: 3, resource: { buffer: this.blockSumBuffer } },
                    { binding: 4, resource: { buffer: this.valuesBufferA } },
                    { binding: 5, resource: { buffer: this.valuesBufferB } }
                ]
            }),
            this.device.createBindGroup({
                label: 'Reorder Bind Group (B -> A)',
                layout: this.reorderResources.bindGroupLayout!,
                entries: [
                    { binding: 0, resource: { buffer: this.keysBufferB } },
                    { binding: 1, resource: { buffer: this.keysBufferA } },
                    { binding: 2, resource: { buffer: this.localPrefixSumBuffer } },
                    { binding: 3, resource: { buffer: this.blockSumBuffer } },
                    { binding: 4, resource: { buffer: this.valuesBufferB } },
                    { binding: 5, resource: { buffer: this.valuesBufferA } }
                ]
            }),
        ];
        
        this.renderResources.bindGroup = this.device.createBindGroup({
            label: 'Render Bind Group',
            layout: this.renderResources.bindGroupLayout!,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.keysBufferB }
                }
            ]
        });
    }

    //================================//
    async startRendering()
    {
        await this.smallCleanup();

        await this.initializeBuffers();
        this.initializeUtils();

        this.mainLoop();
    }

    //================================//
    updateUniforms()
    {
        if (this.device === null) return;
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

            this.updateUniforms();

            const textureView = this.context.getCurrentTexture().createView();
            const renderPassDescriptor: GPURenderPassDescriptor = {
                label: 'basic canvas renderPass',
                colorAttachments: [{
                    view: textureView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 }
                }],
                ... (this.timestampQuerySet != null && {
                    timestampWrites: {
                        querySet: this.timestampQuerySet.querySet,
                        beginningOfPassWriteIndex: 0,
                        endOfPassWriteIndex: 1,
                    }
                }),
            };

            const encoder = this.device.createCommandEncoder({label: 'Main encoder'});

            if (this.sortFlag)
            {
                this.sortFlag = false;
                this.sort(encoder);
            }

            const pass = encoder.beginRenderPass(renderPassDescriptor);

            pass.setPipeline(this.renderResources.pipeline!);
            pass.setBindGroup(0, this.renderResources.bindGroup!);
            pass.draw(6, this.ELEMENT_COUNT, 0, 0);

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

                    if (this.sortedThisFrame)
                    {
                        this.sortedThisFrame = false;
                        this.lastSortTime = Number(times[3] - times[2]) / 1e6;
                    }
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
                Last Sort Time: ${this.lastSortTime > 0 ? this.lastSortTime.toFixed(2) : 'N/A'} ${this.lastSortTime > 0 ? 'ms' : ''}
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
    shuffle()
    {
        this.elements = [];
        for (let i = 0; i < this.ELEMENT_COUNT; i++)
        {
            this.elements.push(Math.floor(Math.random() * (1 << 30)));
        }

        this.device?.queue.writeBuffer(this.keysBufferA, 0, new Uint32Array(this.elements));
        this.device?.queue.writeBuffer(this.keysBufferB, 0, new Uint32Array(this.elements));
    }

    //================================//
    resizeElementCount(newCount: number)
    {
        if (this.device === null) return;
        if (newCount === this.ELEMENT_COUNT) return;
        if (newCount < 1) return;

        this.ELEMENT_COUNT = newCount;
        this.computeConstants();

        this.destroyBuffers();
        this.initializePipelines();
        this.initializeBuffers();
    }

    //================================//
    destroyBuffers()
    {
        this.keysBufferA?.destroy();
        this.keysBufferB?.destroy();
        this.valuesBufferA?.destroy();
        this.valuesBufferB?.destroy();
        this.localPrefixSumBuffer?.destroy();
        this.blockSumBuffer?.destroy();

        if (this.uniformBuffers)
        {
            for (const buf of this.uniformBuffers) buf.destroy();
        }

        for (const level of this.prefixSumLevels)
        {
            level.blockSumBuffer?.destroy();
        }
        this.prefixSumLevels = [];
    }

    //================================//
    async cleanup() 
    {
        await this.smallCleanup();
        this.destroyBuffers();

        if (this.infoElement)
        {
            while(this.infoElement.firstChild) 
            {
                this.infoElement.removeChild(this.infoElement.firstChild);
            }
        }
    }

    //================================//
    dispatchSize(workgroupCount: number): [number, number]
    {
        const x = Math.min(workgroupCount, 65535);
        const y = Math.ceil(workgroupCount / 65535);
        return [x, y];
    }

    //================================//
    async sort(encoder: GPUCommandEncoder)
    {
        if (this.device === null) return;

        if (!this.radixSort)
        {
            const start = performance.now();
            this.elements.sort((a, b) => a - b);
            const end = performance.now();
            this.lastSortTime = end - start;

            this.device.queue.writeBuffer(this.keysBufferA, 0, new Uint32Array(this.elements));
            this.device.queue.writeBuffer(this.keysBufferB, 0, new Uint32Array(this.elements));

            return;
        }

        this.sortedThisFrame = true;
        let computePassDescriptor: GPUComputePassDescriptor = {
            label: 'Radix Sort Compute Pass',
            ... (this.timestampQuerySet != null && {
                timestampWrites: {
                    querySet: this.timestampQuerySet.querySet,
                    beginningOfPassWriteIndex: 2,
                    endOfPassWriteIndex: 3,
                }
            }),
        };

        const computePass = encoder.beginComputePass(computePassDescriptor);

        for (let pass = 0; pass < this.NUM_PASSES; pass++)
        {
            const isEvenPass = (pass % 2 === 0);
            let uniformBindGroup: GPUBindGroup = this.uniformBindGroups[pass];

            // [1.] RADIX SORT
            computePass.setPipeline(this.radixSortResources.pipeline!);
            computePass.setBindGroup(0, isEvenPass ? this.radixSortBindGroups[0] : this.radixSortBindGroups[1]);
            computePass.setBindGroup(1, uniformBindGroup);
            computePass.dispatchWorkgroups(this.radixDispatchX, this.radixDispatchY, 1);

            // [2.] PREFIX SUM
            this.dispatchPrefixSum(computePass);

            // [3.] REORDER
            computePass.setPipeline(this.reorderResources.pipeline!);
            computePass.setBindGroup(0, isEvenPass ? this.reorderBindGroups[0] : this.reorderBindGroups[1]);
            computePass.setBindGroup(1, uniformBindGroup);
            computePass.dispatchWorkgroups(this.radixDispatchX, this.radixDispatchY, 1);
        }

        computePass.end();
    }

    //================================//
    dispatchPrefixSum(pass: GPUComputePassEncoder)
    {
        // EXAMPLE OF HOW IT WORKS IN HIERARCHICAL PREFIX SUM: with 4 items possible per WG
        // Input data[32]: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]
        // Therefore, we need 8 workgroups.
        // Final expected output: Expected output: [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31]

        // Size of the buffers, top down:
        // Level 0:
        // items     = data[32]             ← the actual block sums from radix sort
        // blockSums = L0_blockSums[8]      ← one total per WG (8 workgroups needed)

        // Level 1:
        // items     = L0_blockSums[8]      ← SAME buffer as level 0's blockSums
        // blockSums = L1_blockSums[2]      ← one total per WG (2 workgroups needed)

        // Level 2:
        // items     = L1_blockSums[2]      ← SAME buffer as level 1's blockSums
        // blockSums = L2_blockSums[1]      ← one total per WG (1 workgroup, we stop)

        // LEVEL 0 REDUCE: data[32]: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]
        //                 blockSums[8]: [4, 4, 4, 4, 4, 4, 4, 4]
        // LEVEL 1 REDUCE: data[8]: [0, 4, 8, 12, 0, 4, 8, 12]
        //                 blockSums[2]: [16, 16]
        // LEVEL 2 REDUCE: data[2]: [0, 16]
        //                 blockSums[1]: [32] (will go unused though)

        // THEN COMES THE ADD PIPELINE
        // Other way around, we skip level_2_blocksum, we go straight to level 1.
        // Algo:
        // let blockSum = L1_blockSums[WORKGROUP_ID];
        // L0_blockSums[ELM_ID] += blockSum;
        // L0_blockSums[ELM_ID + 1] += blockSum;

        // I.E:
        // WG0: blockSum = L1_blockSums[0] = 0
        //     L0_blockSums[0..3] += 0  →  [0, 4, 8, 12]    (no change)

        // WG1: blockSum = L1_blockSums[1] = 16
        //     L0_blockSums[4..7] += 16 →  [0+16, 4+16, 8+16, 12+16] = [16, 20, 24, 28]

        // LEVEL 1 ADD: data[8]: [0, 4, 8, 12, 16, 20, 24, 28]
        //                blockSums[2]: [0, 16]
        // LEVEL 0 ADD: data[32]: [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31]
        //                blockSums[8]: [0, 4, 8, 12, 16, 20, 24, 28]

        // DONE
        
        const N = this.prefixSumLevels.length;

        // Sweep first
        for (let i = 0; i < N; i++)
        {
            const level = this.prefixSumLevels[i];
            pass.setPipeline(level.reducePipeline);
            pass.setBindGroup(0, level.bindGroup);
            pass.dispatchWorkgroups(level.dispatchX, level.dispatchY, 1);
        }

        // Then add
        for (let i = N - 2; i >= 0; i--)
        {
            const level = this.prefixSumLevels[i];
            pass.setPipeline(level.addPipeline);
            pass.setBindGroup(0, level.bindGroup);
            pass.dispatchWorkgroups(level.dispatchX, level.dispatchY, 1);
        }
    }

    //================================//
    async smallCleanup()
    {
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
}