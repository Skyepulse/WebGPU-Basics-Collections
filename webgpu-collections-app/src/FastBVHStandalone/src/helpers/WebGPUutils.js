//================================//
export async function RequestWebGPUDevice(features = [])
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

    const logFeatureSupport = (name) => {
        const supported = adapter.features.has(name);
        if (!supported) console.warn(`WebGPU feature not supported: ${name}`);
        else console.log(`WebGPU feature supported: ${name}`);
        return supported;
    };
    features = features.filter(f => logFeatureSupport(f));

    const device = await adapter.requestDevice({ requiredFeatures: features });
    device.lost.then((info) => {
        console.error(`WebGPU device was lost: ${info.message}`);
    });

    return device;
}

//================================//
export function CreateShaderModule(device, vertexSource, fragmentSource, labelName = "shader module")
{
    const vertex = device.createShaderModule({
        label: `${labelName} - vertex`,
        code: vertexSource
    });
    const fragment = device.createShaderModule({
        label: `${labelName} - fragment`,
        code: fragmentSource
    });
    return { vertex, fragment };
}

//================================//
export function CreateTimestampQuerySet(device, numQueries)
{
    if (!device) return null;

    const querySet = device.createQuerySet({
        label: 'timestamp-query-set',
        type: 'timestamp',
        count: numQueries
    });
    const resolveBuffer = device.createBuffer({
        label: 'timestamp-query-resolve-buffer',
        size: numQueries * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    });
    const resultBuffer = device.createBuffer({
        label: 'timestamp-query-result-buffer',
        size: numQueries * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    return { querySet, resolveBuffer, resultBuffer };
}

//================================//
export function ResolveTimestampQuery(timestampQuerySet, encoder)
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
            timestampQuerySet.resolveBuffer, 0,
            timestampQuerySet.resultBuffer, 0,
            timestampQuerySet.resultBuffer.size
        );

    return true;
}
