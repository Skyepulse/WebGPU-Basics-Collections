//================================//
import compWGSL from './basic_comp.wgsl?raw';

//================================//
export async function startup_2(canvas: HTMLCanvasElement)
{
    const adaptor = await navigator.gpu?.requestAdapter();
    const device = await adaptor?.requestDevice();

    if (!device) 
    {
        console.log("WebGPU is not supported on this device.");
        return;
    } else {
        console.log("WebGPU is supported on this device.");
    }

    const basicModule: GPUShaderModule = createBasicModule(device);
    const basicPipeline: GPUComputePipeline = createBasicPipeline(device, basicModule);

    const input = new Float32Array([1, 3, 5]);
    const workerBuffer = createWorkerBuffer(device, input);
    const resultBuffer = createReadBuffer(device, input.byteLength);

    // getBindGroupLayout(0) represents @group(0)
    const bindGroup = createBasicBindGroup(device, basicPipeline.getBindGroupLayout(0), workerBuffer);

    const encoder = device.createCommandEncoder({label: 'command encoder'});
    const pass = encoder.beginComputePass({label: 'basic compute pass'});
    pass.setPipeline(basicPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(input.length);
    pass.end();

    // At the end we copy to the read buffer
    encoder.copyBufferToBuffer(workerBuffer, 0, resultBuffer, 0, resultBuffer.size);

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    // We await the results
    console.log("We send this Input: ", input);
    const startTime = performance.now();

    await resultBuffer.mapAsync(GPUMapMode.READ); // Await for it to be accessible
    const result = new Float32Array(resultBuffer.getMappedRange()); // This returns an Array Buffer of the entire buffer

    console.log("Computation took: ", performance.now() - startTime, "ms");
    console.log("We got this Result: ", result);

    resultBuffer.unmap(); // length will be set to 0 and data not accessible

    return null; // No renderer
}

//================================//
function createBasicModule(device: GPUDevice) {
    return device.createShaderModule({
        label: 'basic compute module',
        code: `${compWGSL}`
    });
}

//================================//
function createBasicPipeline(device: GPUDevice, Cmodule: GPUShaderModule) {
    return device.createComputePipeline({
        label: 'doubling compute pipeline',
        layout: 'auto',
        compute: {
            module: Cmodule,
            entryPoint: 'computeSomething'
        }
    });
}

//================================//
function createWorkerBuffer(device: GPUDevice, data: BufferSource)
{
    const buffer = device.createBuffer({
        label: 'work buffer',
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
}

//================================//
function createReadBuffer(device: GPUDevice, size: number)
{
    return device.createBuffer({
        label: 'result buffer',
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
}

//================================//
function createBasicBindGroup(device: GPUDevice, layout: GPUBindGroupLayout, workerBuffer: GPUBuffer)
{
    return device.createBindGroup({
        label: 'basic bind group',
        layout: layout,
        entries: [
            {
                // binding: 0 represents @binding(0)
                binding: 0,
                resource: { buffer: workerBuffer }
            }
        ]
    });
}