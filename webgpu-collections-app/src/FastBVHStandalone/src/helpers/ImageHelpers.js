//================================//
export const TextureType = { Albedo: 0, Metalness: 1, Roughness: 2, Normal: 3 };

//================================//
export function loadImageFromUrl(url)
{
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = url;
    });
}

//================================//
export function createTextureFromImage(device, image, labelName = "texture")
{
    if (image.width <= 0 || image.height <= 0) {
        console.warn(`Image has invalid dimensions (${image.width}x${image.height}). Using placeholder texture instead.`);
        return createPlaceholderTexture(device);
    }

    const texture = device.createTexture({
        label: labelName,
        size: { width: image.width, height: image.height, depthOrArrayLayers: 1 },
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

//================================//
export function resizeImage(image, targetWidth, targetHeight)
{
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = targetWidth;
    resizeCanvas.height = targetHeight;

    const ctx = resizeCanvas.getContext('2d');
    if (!ctx) {
        console.error("Failed to get 2D context for image resizing.");
        return resizeCanvas;
    }

    ctx.drawImage(image, 0, 0, targetWidth, targetHeight);
    return resizeCanvas;
}

//================================//
export function createPlaceholderTexture(device, size = 256, cells = 32)
{
    const canvas = createPlaceholderImage(size, cells);
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
export function createPlaceholderImage(size = 256, cells = 32)
{
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");

    const cellSize = size / cells;
    for (let y = 0; y < cells; y++) {
        for (let x = 0; x < cells; x++) {
            ctx.fillStyle = (x + y) % 2 === 0 ? "#FF00FF" : "#000000";
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
    return canvas;
}
