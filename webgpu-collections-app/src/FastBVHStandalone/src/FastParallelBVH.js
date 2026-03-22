const BVHNodeSize    = 48;
const flatBVHNodeSize = 32;
const LeafAABBSize   = 32;

//================================//
export class FastParallelBVH
{
    //================================//
    static async create()
    {
        const load = (path) => fetch(path).then(r => { if (!r.ok) throw new Error(`Failed to load ${path}`); return r.text(); });
        const shaders = {
            sceneMinMax:    await load('./Shaders/FastBVHShaders/sceneMinMax.wgsl'),
            minMaxReduce:   await load('./Shaders/FastBVHShaders/minMaxReduce.wgsl'),
            mortonCode:     await load('./Shaders/FastBVHShaders/mortonCode.wgsl'),
            radixBlockSums: await load('./Shaders/FastBVHShaders/radixBlockSums.wgsl'),
            radixPrefixSum: await load('./Shaders/FastBVHShaders/radixPrefixSum.wgsl'),
            radixReorder:   await load('./Shaders/FastBVHShaders/radixReorder.wgsl'),
            patriciaTree:   await load('./Shaders/FastBVHShaders/patriciaTree.wgsl'),
            AABB:           await load('./Shaders/FastBVHShaders/AABB.wgsl'),
            AABBFinalize:   await load('./Shaders/FastBVHShaders/AABBFinalize.wgsl'),
            DFSFlattening:  await load('./Shaders/FastBVHShaders/DFSFlattening.wgsl'),
        };
        return new FastParallelBVH(shaders);
    }

    //================================//
    constructor(shaders)
    {
        this._shaders = shaders;

        this.debug = false;

        this.THREADS_PER_WORKGROUP = 256;
        this.SIZE_X = 16;
        this.SIZE_Y = 16;
        this.numTriangles = 0;
        this.ITEMS_PER_WORKGROUP = 2 * this.THREADS_PER_WORKGROUP;
        this.BIT_COUNT = 30;
        this.NUM_PASSES = this.BIT_COUNT / 2;

        this.minMaxLevels = [];
        this.minMaxReadbackBuffer = null;
        this.prefixSumLevels = [];
    }

    //================================//
    dispatch(commandEncoder)
    {
        this.dispatchMinMaxPass(commandEncoder);
        this.dispatchMortonPass(commandEncoder);
        this.dispatchRadixSort(commandEncoder);
        this.dispatchPatriciaTreePass(commandEncoder);
        this.dispatchAABBPass(commandEncoder);
        this.dispatchAABBFinalizePass(commandEncoder);
        this.dispatchDFSFlatteningPass(commandEncoder);
    }

    //================================//
    dispatchSize(workgroupCount)
    {
        const x = Math.min(workgroupCount, 65535);
        const y = Math.ceil(workgroupCount / 65535);
        return [x, y];
    }

    //================================//
    initializeMinMaxPipeline(device, topLevelVertexBuffer, totalVertexCount)
    {
        this.minMaxLevels = [];

        this.minMaxReduceShaderModule = device.createShaderModule({ label: 'Scene Min Max Reduce Shader Module', code: this._shaders.minMaxReduce });
        this.minMaxSceneShaderModule  = device.createShaderModule({ label: 'Scene Min Max Scene Shader Module',  code: this._shaders.sceneMinMax });

        this.minMaxBindGroupLayout = device.createBindGroupLayout({
            label: 'Scene Min Max Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.minMaxPipelineLayout = device.createPipelineLayout({ label: 'Scene Min Max Pipeline Layout', bindGroupLayouts: [this.minMaxBindGroupLayout] });

        // Level 0: vertices → per-workgroup min/max
        let workGroupCount = Math.ceil(totalVertexCount / this.THREADS_PER_WORKGROUP);
        let [dx, dy] = this.dispatchSize(workGroupCount);
        const level0OutputBuffer = device.createBuffer({
            label: 'Scene Min Max Level 0 Output Buffer',
            size: workGroupCount * 6 * 4,
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

        // Reduction levels
        while (true)
        {
            const wgc = Math.ceil(currentWorkGroupCount / this.THREADS_PER_WORKGROUP);
            const [dx2, dy2] = this.dispatchSize(wgc);
            const outputBuffer = device.createBuffer({
                label: `Scene Min Max Level ${this.minMaxLevels.length} Output Buffer`,
                size: wgc * 6 * 4,
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
            this.minMaxLevels.push({ workgroupCount: wgc, minMaxPipeline: pipeline, minMaxBindGroup: bindGroup, input: currentInputBuffer, output: outputBuffer, dispatchX: dx2, dispatchY: dy2 });
            if (wgc <= 1) break;
            currentWorkGroupCount = wgc;
            currentInputBuffer = outputBuffer;
        }

        this.minMaxReadbackBuffer = device.createBuffer({
            label: 'MinMax Readback Buffer',
            size: 6 * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }

    //================================//
    copyResultForReadback(encoder)
    {
        if (!this.minMaxReadbackBuffer || this.minMaxLevels.length === 0) return;
        const lastOutput = this.minMaxLevels[this.minMaxLevels.length - 1].output;
        encoder.copyBufferToBuffer(lastOutput, 0, this.minMaxReadbackBuffer, 0, 6 * 4);
    }

    //================================//
    dispatchMinMaxPass(commandEncoder)
    {
        for (const level of this.minMaxLevels)
        {
            commandEncoder.setPipeline(level.minMaxPipeline);
            commandEncoder.setBindGroup(0, level.minMaxBindGroup);
            commandEncoder.dispatchWorkgroups(level.dispatchX, level.dispatchY);
        }
    }

    //================================//
    initializeMortonPipeline(device, vertexBuffer, indexBuffer, numTriangles)
    {
        this.numTriangles = numTriangles;
        this.mortonShaderModule = device.createShaderModule({ label: 'Morton Code Shader Module', code: this._shaders.mortonCode });
        this.mortonBindGroupLayout = device.createBindGroupLayout({
            label: 'Morton Code Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.mortonPipelineLayout = device.createPipelineLayout({ label: 'Morton Code Pipeline Layout', bindGroupLayouts: [this.mortonBindGroupLayout] });
        this.mortonPipeline = device.createComputePipeline({
            label: 'Morton Code Pipeline',
            layout: this.mortonPipelineLayout,
            compute: {
                module: this.mortonShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, SIZE_X: this.SIZE_X, SIZE_Y: this.SIZE_Y, TRIANGLE_COUNT: numTriangles }
            }
        });
        this.mortonOutputBitsBuffer = device.createBuffer({
            label: 'Morton Code Output Bits Buffer',
            size: numTriangles * 4,
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
            entries: [
                { binding: 0, resource: { buffer: vertexBuffer } },
                { binding: 1, resource: { buffer: indexBuffer } },
                { binding: 2, resource: { buffer: this.minMaxLevels[this.minMaxLevels.length - 1].output } },
                { binding: 3, resource: { buffer: this.mortonOutputBitsBuffer } },
                { binding: 4, resource: { buffer: this.mortonOutputTriangleIndexBuffer } },
            ]
        });
    }

    //================================//
    dispatchMortonPass(commandEncoder)
    {
        if (!this.mortonShaderModule || !this.mortonBindGroup) return;
        const [dx, dy] = this.dispatchSize(Math.ceil(this.numTriangles / this.THREADS_PER_WORKGROUP));
        commandEncoder.setPipeline(this.mortonPipeline);
        commandEncoder.setBindGroup(0, this.mortonBindGroup);
        commandEncoder.dispatchWorkgroups(dx, dy);
    }

    //================================//
    initializeRadixSortPipelines(device)
    {
        const totalElements = this.numTriangles;
        this.WORKGROUP_COUNT = Math.ceil(totalElements / this.THREADS_PER_WORKGROUP);

        this.radixSortShaderModule  = device.createShaderModule({ label: 'Radix Sort Shader Module',          code: this._shaders.radixBlockSums });
        this.reorderShaderModule    = device.createShaderModule({ label: 'Radix Reorder Shader Module',        code: this._shaders.radixReorder });
        this.prefixSumShaderModule  = device.createShaderModule({ label: 'Prefix Sum Reduce Shader Module',    code: this._shaders.radixPrefixSum });

        // Uniform bind group layout (one per pass)
        this.uniformBindGroupLayout = device.createBindGroupLayout({
            label: 'Uniform Bind Group Layout',
            entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } } ]
        });
        this.uniformBuffers = [];
        this.uniformBindGroups = [];
        for (let i = 0; i < this.NUM_PASSES; i++)
        {
            const buffer = device.createBuffer({ label: `Radix Sort Uniform Buffer Pass ${i}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(buffer, 0, new Uint32Array([i * 2]));
            const bindGroup = device.createBindGroup({ label: `Radix Sort Uniform Bind Group Pass ${i}`, layout: this.uniformBindGroupLayout, entries: [{ binding: 0, resource: { buffer } }] });
            this.uniformBindGroups.push(bindGroup);
            this.uniformBuffers.push(buffer);
        }

        // Prefix sum bind group layout
        this.prefixSumBindGroupLayout = device.createBindGroupLayout({
            label: 'Prefix Sum Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.prefixSumPipelineLayout = device.createPipelineLayout({ label: 'Prefix Sum Pipeline Layout', bindGroupLayouts: [this.prefixSumBindGroupLayout] });

        // Double-buffered keys/values
        this.keysBufferA   = this.mortonOutputBitsBuffer;
        this.keysBufferB   = device.createBuffer({ label: 'Radix Sort Keys Buffer B',   size: this.numTriangles * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.valuesBufferA = this.mortonOutputTriangleIndexBuffer;
        this.valuesBufferB = device.createBuffer({ label: 'Radix Sort Values Buffer B', size: this.numTriangles * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        this.localPrefixSumBuffer = device.createBuffer({ label: 'Radix Sort Local Prefix Sum Buffer', size: this.numTriangles * 4,          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.blockSumBuffer       = device.createBuffer({ label: 'Radix Sort Block Sum Buffer',        size: 4 * this.WORKGROUP_COUNT * 4,   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        // [1] Radix sort pipeline
        this.radixSortBindGroupLayout = device.createBindGroupLayout({
            label: 'Radix Sort Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.radixSortPipelineLayout = device.createPipelineLayout({ label: 'Radix Sort Pipeline Layout', bindGroupLayouts: [this.radixSortBindGroupLayout, this.uniformBindGroupLayout] });
        this.radixSortPipeline = device.createComputePipeline({
            label: 'Radix Sort Pipeline',
            layout: this.radixSortPipelineLayout,
            compute: {
                module: this.radixSortShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, X_SIZE: this.SIZE_X, Y_SIZE: this.SIZE_Y, ELEMENT_COUNT: this.numTriangles, WORKGROUP_COUNT: this.WORKGROUP_COUNT }
            }
        });
        this.radixSortBindGroups = [
            device.createBindGroup({ label: 'Radix Sort Bind Group',   layout: this.radixSortBindGroupLayout, entries: [{ binding: 0, resource: { buffer: this.keysBufferA } }, { binding: 1, resource: { buffer: this.localPrefixSumBuffer } }, { binding: 2, resource: { buffer: this.blockSumBuffer } }] }),
            device.createBindGroup({ label: 'Radix Sort Bind Group 2', layout: this.radixSortBindGroupLayout, entries: [{ binding: 0, resource: { buffer: this.keysBufferB } }, { binding: 1, resource: { buffer: this.localPrefixSumBuffer } }, { binding: 2, resource: { buffer: this.blockSumBuffer } }] }),
        ];

        // [2] Reorder pipeline
        this.reorderBindGroupLayout = device.createBindGroupLayout({
            label: 'Reorder Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.reorderPipelineLayout = device.createPipelineLayout({ label: 'Reorder Pipeline Layout', bindGroupLayouts: [this.reorderBindGroupLayout, this.uniformBindGroupLayout] });
        this.reorderPipeline = device.createComputePipeline({
            label: 'Reorder Pipeline',
            layout: this.reorderPipelineLayout,
            compute: {
                module: this.reorderShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, X_SIZE: this.SIZE_X, Y_SIZE: this.SIZE_Y, ELEMENT_COUNT: this.numTriangles, WORKGROUP_COUNT: this.WORKGROUP_COUNT }
            }
        });
        this.reorderBindGroups = [
            device.createBindGroup({ label: 'Reorder Bind Group',   layout: this.reorderBindGroupLayout, entries: [{ binding: 0, resource: { buffer: this.keysBufferA } }, { binding: 1, resource: { buffer: this.keysBufferB } }, { binding: 2, resource: { buffer: this.localPrefixSumBuffer } }, { binding: 3, resource: { buffer: this.blockSumBuffer } }, { binding: 4, resource: { buffer: this.valuesBufferA } }, { binding: 5, resource: { buffer: this.valuesBufferB } }] }),
            device.createBindGroup({ label: 'Reorder Bind Group 2', layout: this.reorderBindGroupLayout, entries: [{ binding: 0, resource: { buffer: this.keysBufferB } }, { binding: 1, resource: { buffer: this.keysBufferA } }, { binding: 2, resource: { buffer: this.localPrefixSumBuffer } }, { binding: 3, resource: { buffer: this.blockSumBuffer } }, { binding: 4, resource: { buffer: this.valuesBufferB } }, { binding: 5, resource: { buffer: this.valuesBufferA } }] }),
        ];

        // [3] Prefix sum pipeline
        this.prefixSumLevels = [];
        let currentElementCount = 4 * this.WORKGROUP_COUNT;
        let currentDataBuffer = this.blockSumBuffer;

        while (true)
        {
            const wgc = Math.ceil(currentElementCount / this.ITEMS_PER_WORKGROUP);
            const [dx, dy] = this.dispatchSize(wgc);
            const blockSumBuf = device.createBuffer({ label: `Prefix Sum Block Sum Buffer Level ${this.prefixSumLevels.length}`, size: Math.max(wgc, 1) * 4, usage: GPUBufferUsage.STORAGE });
            const reducePipeline = device.createComputePipeline({
                label: `Prefix Sum Reduce Pipeline Level ${this.prefixSumLevels.length}`,
                layout: this.prefixSumPipelineLayout,
                compute: { module: this.prefixSumShaderModule, entryPoint: 'cs_reduce', constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, X_SIZE: this.SIZE_X, Y_SIZE: this.SIZE_Y, ITEMS_PER_WORKGROUP: this.ITEMS_PER_WORKGROUP, ELEMENT_COUNT: currentElementCount } }
            });
            const addPipeline = device.createComputePipeline({
                label: `Prefix Sum Add Pipeline Level ${this.prefixSumLevels.length}`,
                layout: this.prefixSumPipelineLayout,
                compute: { module: this.prefixSumShaderModule, entryPoint: 'cs_add', constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, X_SIZE: this.SIZE_X, Y_SIZE: this.SIZE_Y, ITEMS_PER_WORKGROUP: this.ITEMS_PER_WORKGROUP, ELEMENT_COUNT: currentElementCount } }
            });
            const bindGroup = device.createBindGroup({ label: `Prefix Sum Bind Group Level ${this.prefixSumLevels.length}`, layout: this.prefixSumBindGroupLayout, entries: [{ binding: 0, resource: { buffer: currentDataBuffer } }, { binding: 1, resource: { buffer: blockSumBuf } }] });
            this.prefixSumLevels.push({ elementCount: currentElementCount, workgroupCount: wgc, reducePipeline, addPipeline, bindGroup, dataBuffer: currentDataBuffer, blockSumBuffer: blockSumBuf, dispatchX: dx, dispatchY: dy });
            if (wgc <= 1) break;
            currentElementCount = wgc;
            currentDataBuffer = blockSumBuf;
        }
    }

    //================================//
    dispatchRadixSort(commandEncoder)
    {
        const [dx, dy] = this.dispatchSize(this.WORKGROUP_COUNT);

        for (let pass = 0; pass < this.NUM_PASSES; pass++)
        {
            const isEvenPass = (pass % 2 === 0);
            const uniformBindGroup = this.uniformBindGroups[pass];

            commandEncoder.setPipeline(this.radixSortPipeline);
            commandEncoder.setBindGroup(0, isEvenPass ? this.radixSortBindGroups[0] : this.radixSortBindGroups[1]);
            commandEncoder.setBindGroup(1, uniformBindGroup);
            commandEncoder.dispatchWorkgroups(dx, dy, 1);

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

            commandEncoder.setPipeline(this.reorderPipeline);
            commandEncoder.setBindGroup(0, isEvenPass ? this.reorderBindGroups[0] : this.reorderBindGroups[1]);
            commandEncoder.setBindGroup(1, uniformBindGroup);
            commandEncoder.dispatchWorkgroups(dx, dy, 1);
        }
    }

    //================================//
    initializePatriciaTreePipeline(device)
    {
        this.patriciaTreeShaderModule = device.createShaderModule({ label: 'Patricia Tree Shader Module', code: this._shaders.patriciaTree });
        this.patriciaTreeBindGroupLayout = device.createBindGroupLayout({
            label: 'Patricia Tree Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.patriciaTreePipelineLayout = device.createPipelineLayout({ label: 'Patricia Tree Pipeline Layout', bindGroupLayouts: [this.patriciaTreeBindGroupLayout] });
        this.patriciaTreePipeline = device.createComputePipeline({
            label: 'Patricia Tree Pipeline',
            layout: this.patriciaTreePipelineLayout,
            compute: {
                module: this.patriciaTreeShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, INTERNAL_NODE_COUNT: this.numTriangles - 1, LEAF_NODE_COUNT: this.numTriangles }
            }
        });

        // After 15 (NUM_PASSES=odd) radix sort passes, sorted keys/values are in B buffers
        this.mortonCodesBuffer = this.keysBufferB;

        this.internalNodesBuffer = device.createBuffer({ label: 'BVH Internal Nodes Buffer', size: (this.numTriangles - 1) * BVHNodeSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.leafNodesBuffer     = device.createBuffer({ label: 'BVH Leaf Nodes Buffer',      size: this.numTriangles * 4,                   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        this.patriciaTreeBindGroup = device.createBindGroup({
            label: 'Patricia Tree Bind Group',
            layout: this.patriciaTreeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.mortonCodesBuffer } },
                { binding: 1, resource: { buffer: this.internalNodesBuffer } },
                { binding: 2, resource: { buffer: this.leafNodesBuffer } },
            ]
        });
    }

    //================================//
    dispatchPatriciaTreePass(commandEncoder)
    {
        if (!this.patriciaTreePipeline || !this.patriciaTreeBindGroup) return;
        const dispatchX = Math.ceil((this.numTriangles - 1) / this.THREADS_PER_WORKGROUP);
        commandEncoder.setPipeline(this.patriciaTreePipeline);
        commandEncoder.setBindGroup(0, this.patriciaTreeBindGroup);
        commandEncoder.dispatchWorkgroups(dispatchX, 1, 1);
    }

    //================================//
    initializeAABBPipeline(device, vertexBuffer, indexBuffer)
    {
        this.aabbShaderModule = device.createShaderModule({ label: 'AABB Shader Module', code: this._shaders.AABB });
        this.aabbBindGroupLayout = device.createBindGroupLayout({
            label: 'AABB Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.aabbPipelineLayout = device.createPipelineLayout({ label: 'AABB Pipeline Layout', bindGroupLayouts: [this.aabbBindGroupLayout] });
        this.aabbPipeline = device.createComputePipeline({
            label: 'AABB Pipeline',
            layout: this.aabbPipelineLayout,
            compute: {
                module: this.aabbShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, INTERNAL_NODE_COUNT: this.numTriangles - 1, LEAF_NODE_COUNT: this.numTriangles }
            }
        });

        this.aabbInternalNodesBuffer = this.internalNodesBuffer;
        this.aabbLeafNodesBuffer     = this.leafNodesBuffer;
        this.aabbVertexBuffer        = vertexBuffer;
        this.aabbIndexBuffer         = indexBuffer;
        this.aabbSortedIndexBuffer   = this.valuesBufferB; // sorted values also in B after 15 passes

        this.leafAABBsBuffer   = device.createBuffer({ label: 'Leaf AABBs Buffer',   size: this.numTriangles * LeafAABBSize,          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.aabbReadyBufferA  = device.createBuffer({ label: 'AABB Ready Buffer A', size: Math.max(this.numTriangles - 1, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.aabbReadyBufferB  = device.createBuffer({ label: 'AABB Ready Buffer B', size: Math.max(this.numTriangles - 1, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

        this.aabbBindGroup = device.createBindGroup({
            label: 'AABB Bind Group',
            layout: this.aabbBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.aabbVertexBuffer } },
                { binding: 1, resource: { buffer: this.aabbIndexBuffer } },
                { binding: 2, resource: { buffer: this.aabbSortedIndexBuffer } },
                { binding: 3, resource: { buffer: this.leafAABBsBuffer } },
            ]
        });
    }

    //================================//
    initializeAABBFinalizePipeline(device)
    {
        this.aabbFinalizeShaderModule = device.createShaderModule({ label: 'AABB Finalize Shader Module', code: this._shaders.AABBFinalize });
        this.aabbFinalizeBindGroupLayout = device.createBindGroupLayout({
            label: 'AABB Finalize Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.aabbFinalizePipelineLayout = device.createPipelineLayout({ label: 'AABB Finalize Pipeline Layout', bindGroupLayouts: [this.aabbFinalizeBindGroupLayout] });
        this.aabbFinalizePipeline = device.createComputePipeline({
            label: 'AABB Finalize Pipeline',
            layout: this.aabbFinalizePipelineLayout,
            compute: {
                module: this.aabbFinalizeShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, INTERNAL_NODE_COUNT: this.numTriangles - 1 }
            }
        });
        this.aabbFinalizeBindGroups = [
            device.createBindGroup({ label: 'AABB Finalize Bind Group 0', layout: this.aabbFinalizeBindGroupLayout, entries: [{ binding: 0, resource: { buffer: this.aabbInternalNodesBuffer } }, { binding: 1, resource: { buffer: this.leafAABBsBuffer } }, { binding: 2, resource: { buffer: this.aabbReadyBufferA } }, { binding: 3, resource: { buffer: this.aabbReadyBufferB } }] }),
            device.createBindGroup({ label: 'AABB Finalize Bind Group 1', layout: this.aabbFinalizeBindGroupLayout, entries: [{ binding: 0, resource: { buffer: this.aabbInternalNodesBuffer } }, { binding: 1, resource: { buffer: this.leafAABBsBuffer } }, { binding: 2, resource: { buffer: this.aabbReadyBufferB } }, { binding: 3, resource: { buffer: this.aabbReadyBufferA } }] }),
        ];
    }

    //================================//
    clearAtomicCounters(encoder)
    {
        if (!this.aabbReadyBufferA || !this.aabbReadyBufferB) return;
        encoder.clearBuffer(this.aabbReadyBufferA);
        encoder.clearBuffer(this.aabbReadyBufferB);
    }

    //================================//
    dispatchAABBPass(commandEncoder)
    {
        if (!this.aabbPipeline || !this.aabbBindGroup) return;
        const dispatchX = Math.ceil(this.numTriangles / this.THREADS_PER_WORKGROUP);
        commandEncoder.setPipeline(this.aabbPipeline);
        commandEncoder.setBindGroup(0, this.aabbBindGroup);
        commandEncoder.dispatchWorkgroups(dispatchX, 1, 1);
    }

    //================================//
    // BIT_COUNT=30 iterations is the proven upper bound for 30-bit Morton codes.
    dispatchAABBFinalizePass(commandEncoder)
    {
        if (!this.aabbFinalizePipeline || !this.aabbFinalizeBindGroups) return;
        const dispatchX = Math.ceil((this.numTriangles - 1) / this.THREADS_PER_WORKGROUP);
        if (dispatchX <= 0) return;
        commandEncoder.setPipeline(this.aabbFinalizePipeline);
        for (let i = 0; i < this.BIT_COUNT; i++)
        {
            commandEncoder.setBindGroup(0, this.aabbFinalizeBindGroups[i % 2]);
            commandEncoder.dispatchWorkgroups(dispatchX, 1, 1);
        }
    }

    //================================//
    initializeDFSFlatteningPipeline(device)
    {
        this.dfsFlatteningShaderModule = device.createShaderModule({ label: 'DFS Flattening Shader Module', code: this._shaders.DFSFlattening });
        this.dfsFlatteningBindGroupLayout = device.createBindGroupLayout({
            label: 'DFS Flattening Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });
        this.dfsFlatteningPipelineLayout = device.createPipelineLayout({ label: 'DFS Flattening Pipeline Layout', bindGroupLayouts: [this.dfsFlatteningBindGroupLayout] });
        this.dfsFlatteningPipeline = device.createComputePipeline({
            label: 'DFS Flattening Pipeline',
            layout: this.dfsFlatteningPipelineLayout,
            compute: {
                module: this.dfsFlatteningShaderModule,
                entryPoint: 'cs',
                constants: { THREADS_PER_WORKGROUP: this.THREADS_PER_WORKGROUP, INTERNAL_NODE_COUNT: this.numTriangles - 1, LEAF_NODE_COUNT: this.numTriangles }
            }
        });

        this.dfsFlattenedNodesBuffer = device.createBuffer({
            label: 'DFS Flattened BVH Buffer',
            size: (this.numTriangles * 2 - 1) * flatBVHNodeSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.dfsFlatteningBindGroup = device.createBindGroup({
            label: 'DFS Flattening Bind Group',
            layout: this.dfsFlatteningBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.aabbInternalNodesBuffer } },
                { binding: 1, resource: { buffer: this.aabbLeafNodesBuffer } },
                { binding: 2, resource: { buffer: this.leafAABBsBuffer } },
                { binding: 3, resource: { buffer: this.aabbSortedIndexBuffer } },
                { binding: 4, resource: { buffer: this.dfsFlattenedNodesBuffer } },
            ]
        });
    }

    //================================//
    dispatchDFSFlatteningPass(commandEncoder)
    {
        if (!this.dfsFlatteningPipeline || !this.dfsFlatteningBindGroup) return;
        const dispatchX = Math.ceil((this.numTriangles * 2 - 1) / this.THREADS_PER_WORKGROUP);
        commandEncoder.setPipeline(this.dfsFlatteningPipeline);
        commandEncoder.setBindGroup(0, this.dfsFlatteningBindGroup);
        commandEncoder.dispatchWorkgroups(dispatchX, 1, 1);
    }

    //================================//
    getFinalFlattenedBVHBuffer()    { return this.dfsFlattenedNodesBuffer; }
    getFinalFlattenedBVHNodeCount() { return this.numTriangles * 2 - 1; }
}
