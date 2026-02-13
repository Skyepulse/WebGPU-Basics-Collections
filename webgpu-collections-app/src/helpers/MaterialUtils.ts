//================================//
export const MATERIAL_SIZE = 16; // 16 floats = 64 bytes

//================================//
export interface Material
{
    name: string;
    albedo: [number, number, number]; // vec3
    roughness: number;
    usePerlinRoughness: boolean;
    metalness: number;
    usePerlinMetalness: boolean;
    perlinFreq: number;

    albedoTexture?: HTMLImageElement;
    useAlbedoTexture: boolean;
    metalnessTexture?: HTMLImageElement;
    useMetalnessTexture: boolean;
    roughnessTexture?: HTMLImageElement;
    useRoughnessTexture: boolean;
    useNormalTexture: boolean;
    normalTexture?: HTMLImageElement;

    albedoGPUTexture?: GPUTexture;
    metalnessGPUTexture?: GPUTexture;
    roughnessGPUTexture?: GPUTexture;
    normalGPUTexture?: GPUTexture;

    textureIndex: number; // Possible textureArray index
};

//================================//
export function createDefaultMaterial( {
    name = "default",
    albedo = [1.0, 1.0, 1.0] as [number, number, number],
    roughness = 0.98,
    metalness = 0.0,
    usePerlinRoughness = false,
    usePerlinMetalness = false,
    perlinFreq = 2.0,
    useAlbedoTexture = false,
    useMetalnessTexture = false,
    useRoughnessTexture = false,
    useNormalTexture = false,
    textureIndex = -1
} ): Material
{
    return {
        name: name,
        albedo: albedo,
        roughness: roughness,
        usePerlinRoughness: usePerlinRoughness,
        metalness: metalness,
        usePerlinMetalness: usePerlinMetalness,
        perlinFreq: perlinFreq,

        useAlbedoTexture: useAlbedoTexture,
        useMetalnessTexture: useMetalnessTexture,
        useRoughnessTexture: useRoughnessTexture,
        useNormalTexture: useNormalTexture,

        textureIndex: textureIndex
    };
}

//================================//
export function flattenMaterial(material: Material): Float32Array
{
    const array = new Array(MATERIAL_SIZE);
    const float32View = new Float32Array(array);

    float32View.set(material.albedo, 0);
    float32View[3] = material.metalness;
    float32View[4] = material.usePerlinMetalness ? 1.0 : 0.0;
    float32View[5] = material.roughness;
    float32View[6] = material.usePerlinRoughness ? 1.0 : 0.0;
    float32View[7] = material.perlinFreq;
    float32View[8] = material.useAlbedoTexture ? 1.0 : 0.0;
    float32View[9] = material.useMetalnessTexture ? 1.0 : 0.0;
    float32View[10] = material.useRoughnessTexture ? 1.0 : 0.0;
    float32View[11] = material.useNormalTexture ? 1.0 : 0.0;
    float32View[12] = material.textureIndex;
    
    return float32View;
}

//================================//
export function flattenMaterialArray(materials: Material[]): Float32Array
{
    const flattenedMaterials: number[] = [];
    for (const mat of materials)
    {
        flattenedMaterials.push(...mat.albedo);
        flattenedMaterials.push(mat.metalness);
        flattenedMaterials.push(mat.usePerlinMetalness ? 1.0 : 0.0);
        flattenedMaterials.push(mat.roughness);
        flattenedMaterials.push(mat.usePerlinRoughness ? 1.0 : 0.0);
        flattenedMaterials.push(mat.perlinFreq);
        flattenedMaterials.push(mat.useAlbedoTexture ? 1.0 : 0.0);
        flattenedMaterials.push(mat.useMetalnessTexture ? 1.0 : 0.0);
        flattenedMaterials.push(mat.useRoughnessTexture ? 1.0 : 0.0);
        flattenedMaterials.push(mat.useNormalTexture ? 1.0 : 0.0);
        flattenedMaterials.push(mat.textureIndex);
        flattenedMaterials.push(0.0);
        flattenedMaterials.push(0.0);
        flattenedMaterials.push(0.0);
    }

    return new Float32Array(flattenedMaterials);
}