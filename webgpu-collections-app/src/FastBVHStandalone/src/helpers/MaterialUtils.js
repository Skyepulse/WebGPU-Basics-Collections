//================================//
export const MATERIAL_SIZE = 16;

//================================//
export function createDefaultMaterial({
    name = "default",
    albedo = [1.0, 1.0, 1.0],
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
} = {})
{
    return {
        name,
        albedo,
        roughness,
        usePerlinRoughness,
        metalness,
        usePerlinMetalness,
        perlinFreq,
        useAlbedoTexture,
        useMetalnessTexture,
        useRoughnessTexture,
        useNormalTexture,
        textureIndex
    };
}

//================================//
export function flattenMaterial(material)
{
    const float32View = new Float32Array(MATERIAL_SIZE);

    float32View.set(material.albedo, 0);
    float32View[3]  = material.metalness;
    float32View[4]  = material.usePerlinMetalness ? 1.0 : 0.0;
    float32View[5]  = material.roughness;
    float32View[6]  = material.usePerlinRoughness ? 1.0 : 0.0;
    float32View[7]  = material.perlinFreq;
    float32View[8]  = material.useAlbedoTexture ? 1.0 : 0.0;
    float32View[9]  = material.useMetalnessTexture ? 1.0 : 0.0;
    float32View[10] = material.useRoughnessTexture ? 1.0 : 0.0;
    float32View[11] = material.useNormalTexture ? 1.0 : 0.0;
    float32View[12] = material.textureIndex;

    return float32View;
}

//================================//
export function flattenMaterialArray(materials)
{
    const flattenedMaterials = [];
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
