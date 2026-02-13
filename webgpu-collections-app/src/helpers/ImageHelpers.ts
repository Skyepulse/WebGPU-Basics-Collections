///<reference types="@webgpu/types" />

export enum TextureType
{
    Albedo,
    Metalness,
    Roughness,
    Normal
}

/*
 * Load file from URL and return as HTMLImageElement.
 * @param url The URL of the image to load.
 * @returns A promise that resolves to an HTMLImageElement containing the loaded image.
 */
export function loadImageFromUrl(url: string): Promise<HTMLImageElement>
{
    return new Promise((resolve, reject) =>
    {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = url;
    });
}

/*
 * Creates a GPUTexture from an HTMLImageElement.
 * @param device The GPU device to create the texture on.
 * @param image The HTMLImageElement containing the image data to create the texture from.
 * @param labelName An optional label for the texture for debugging purposes.
 * @return A GPUTexture created from the provided image, ready to be used in rendering or as a bindable resource.
 */
export function createTextureFromImage(device: GPUDevice, image: HTMLImageElement, labelName: string = "texture"): GPUTexture
{
    // Check if the image is sane
    if (image.width <= 0 || image.height <= 0)
    {
        // return placeholder texture
        console.warn(`Image has invalid dimensions (${image.width}x${image.height}). Using placeholder texture instead.`);
        return createPlaceholderTexture(device);
    }
    
    const texture: GPUTexture = device.createTexture({
        label: labelName,
        size: {
            width: image.width,
            height: image.height,
            depthOrArrayLayers: 1
        },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });
    device.queue.copyExternalImageToTexture(
        { source: image },
        { texture: texture },
        [image.width, image.height]
    );
    return texture;
}

/*
 * Resize image to target dimensions.
 */
export function resizeImage(image: HTMLImageElement, targetWidth: number, targetHeight: number): HTMLImageElement
{
    // We need to pass through a canvas to resize the image
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = targetWidth;
    resizeCanvas.height = targetHeight;

    const ctx = resizeCanvas.getContext('2d');
    if (!ctx) 
    {
        console.error("Failed to get 2D context for image resizing.");
        return image;
    }

    ctx.drawImage(image, 0, 0, targetWidth, targetHeight);

    const resizedImage = new Image();
    resizedImage.src = resizeCanvas.toDataURL();

    return resizedImage;
}

/*
 * Create a simple placeholder texture for binding purposes
 */
export function createPlaceholderTexture(device: GPUDevice, size: number = 256, cells: number = 32): GPUTexture
{
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;

    const cellSize = size / cells;
    for (let y = 0; y < cells; y++)
    {
        for (let x = 0; x < cells; x++)
        {
            ctx.fillStyle = (x + y) % 2 === 0 ? "#FF00FF" : "#000000";
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }

    const texture = device.createTexture({
        label: "placeholder-texture",
        size: [size, size],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    device.queue.copyExternalImageToTexture(
        { source: canvas },
        { texture },
        [size, size]
    );

    return texture;
}

//================================//
export function createPlaceholderImage(size: number = 256, cells: number = 32): HTMLCanvasElement
{
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;

    const cellSize = size / cells;
    for (let y = 0; y < cells; y++)
    {
        for (let x = 0; x < cells; x++)
        {
            ctx.fillStyle = (x + y) % 2 === 0 ? "#FF00FF" : "#000000";
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }

    return canvas;
}