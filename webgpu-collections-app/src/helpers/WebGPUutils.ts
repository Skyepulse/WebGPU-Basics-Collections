///<reference types="@webgpu/types" />

//============== STRUCTS ==================//
export interface ShaderModule
{
    vertex: GPUShaderModule;
    fragment: GPUShaderModule;
}

export interface TimestampQuerySet
{
    querySet: GPUQuerySet;
    resolveBuffer: GPUBuffer;
    resultBuffer: GPUBuffer;
}

//============== METHODS ==================//

/*
 * Request access to WebGPU in browser.
 * @returns device
 */
export async function RequestWebGPUDevice(features: GPUFeatureName[] = []): Promise<GPUDevice | null> 
{
    if (!navigator.gpu) {
        alert("WebGPU is not supported in this browser.");
        console.error("WebGPU is not supported in this browser.");
        return null;
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        alert("This browser supports WebGPU, but it appears disabled.");
        console.error("This browser supports WebGPU, but it appears disabled.");
        return null;
    }

    const logFeatureSupport = (name: GPUFeatureName): boolean =>
    {
        const supported = adapter.features.has(name);
        if (!supported) console.warn(`WebGPU feature not supported: ${name}`);
        else console.log(`WebGPU feature supported: ${name}`);
        return supported;
    };
    features = features.filter(f => logFeatureSupport(f));

    const device = await adapter.requestDevice(
        {
            requiredFeatures: features
        }
    );
    device.lost.then((info) => {
        console.error(`WebGPU device was lost: ${info.message}`);
    });

    return device;
}

/*
 *
 * Creates vertex and fragment shaders from WGSL source code.
 *
 */
export function CreateShaderModule(device: GPUDevice, vertexSource: string, fragmentSource: string, labelName: string = "shader module"): ShaderModule | null {
    const vertexShaderModule = device.createShaderModule({
        label: `${labelName} - vertex`,
        code: vertexSource
    });

    const fragmentShaderModule = device.createShaderModule({
        label: `${labelName} - fragment`,
        code: fragmentSource
    });

    return {
        vertex: vertexShaderModule,
        fragment: fragmentShaderModule
    };
}

/*
 * Creates a timestamp query set and associated buffers.
 * @param device The GPU device.
 * @param numQueries The number of timestamp queries.
 * @returns An object containing the query set and buffers.
 */
export function CreateTimestampQuerySet(device: GPUDevice, numQueries: number): TimestampQuerySet | null
{
    if (!device) return null;

    const querySet = device.createQuerySet({
        label: 'timestamp-query-set',
        type: 'timestamp',
        count: numQueries
    });

    const resolveBuffer = device.createBuffer({
        label: 'timestamp-query-resolve-buffer',
        size: numQueries * 8, // 8 bytes per timestamp
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    });

    const resultBuffer = device.createBuffer({
        label: 'timestamp-query-result-buffer',
        size: numQueries * 8, // 8 bytes per timestamp
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    return {
        querySet,
        resolveBuffer,
        resultBuffer
    };
}

/*
 * Resolves and reads timestamp query results.
 * @Param the timestamp query set object.
 * @Param the encoder to use for resolving the queries.
 */
export function ResolveTimestampQuery(timestampQuerySet: TimestampQuerySet, encoder: GPUCommandEncoder): boolean
{
    if (!timestampQuerySet || !encoder) return false;

    encoder.resolveQuerySet(
        timestampQuerySet.querySet,
        0, timestampQuerySet.querySet.count,
        timestampQuerySet.resolveBuffer,
        0
    );

    if (timestampQuerySet.resultBuffer.mapState === 'unmapped')
        encoder.copyBufferToBuffer(
            timestampQuerySet.resolveBuffer,
            0,
            timestampQuerySet.resultBuffer,
            0,
            timestampQuerySet.resultBuffer.size
        );

    return true;
}