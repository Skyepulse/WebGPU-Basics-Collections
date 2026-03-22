import { createDefaultMaterial } from './MaterialUtils.js';

//================================//
export async function fastBVHExampleScene(meshMaterials, seed, numSpheres, withPlane = true)
{
    // seed
    let s = seed | 0;
    const random = () => {
        s = (s + 0x6D2B79F5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const randomRange = (min, max) => random() * (max - min) + min;

    const allPositions = [];
    const allNormals   = [];
    const allUVs       = [];
    const allIndices   = [];
    const perTriangleMaterialIndices = [];
    const perMeshWorldPositionOffsets = [];
    const materials = [];
    const perMeshData = [];
    const sphereCenters = [];

    let globalVertexOffset = 0;

    function currentByteOffset() { return globalVertexOffset * 3 * 4; }

    // ============== GROUND PLANE ============== //
    const planeMin = -100;
    const planeMax = 100;
    if (withPlane)
    {
        const pY = 0;
        const planeMat = meshMaterials.find(m => m.name === 'ground') ||
            createDefaultMaterial({ albedo: [0.9, 0.9, 0.9], name: 'ground', roughness: 1.0, metalness: 0.0 });
        materials.push(planeMat);
        perMeshWorldPositionOffsets.push(currentByteOffset());

        const planePositions = new Float32Array([
            planeMin, pY, planeMin,
            planeMax, pY, planeMin,
            planeMax, pY, planeMax,
            planeMin, pY, planeMax,
        ]);
        const planeNormals = new Float32Array([
            0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0,
        ]);
        const planeUVs = new Float32Array([
            0, 0,  1, 0,  1, 1,  0, 1,
        ]);
        const planeIndices = new Uint16Array([0, 2, 1, 0, 3, 2]);
        perMeshData.push({ positions: planePositions, normals: planeNormals, uvs: planeUVs, indices: planeIndices });

        allPositions.push(...planePositions);
        allNormals.push(...planeNormals);
        allUVs.push(...planeUVs);
        allIndices.push(globalVertexOffset+0, globalVertexOffset+2, globalVertexOffset+1);
        perTriangleMaterialIndices.push(0);
        allIndices.push(globalVertexOffset+0, globalVertexOffset+3, globalVertexOffset+2);
        perTriangleMaterialIndices.push(0);
        globalVertexOffset += 4;
    }

    //================================//
    const latBands = 16;
    const lonBands = 16;
    const vertsPerSphere = (latBands + 1) * (lonBands + 1);

    for (let i = 0; i < numSpheres; i++)
    {
        const sphereMat = meshMaterials.find(m => m.name === `sphere${i}`) ||
            createDefaultMaterial({
                albedo: [random(), random(), random()],
                name: `sphere${i}`,
                roughness: randomRange(0.01, 1.0),
                metalness: random() > 0.5 ? randomRange(0.8, 1.0) : randomRange(0.0, 0.2)
            });

        const materialIndex = materials.length;
        materials.push(sphereMat);
        perMeshWorldPositionOffsets.push(currentByteOffset());

        const radius = randomRange(2, 8);
        const cx = randomRange(planeMin + radius, planeMax - radius);
        const cy = radius;
        const cz = randomRange(planeMin + radius, planeMax - radius);
        sphereCenters.push({ cx, cy, cz, radius });

        const meshPositions = new Float32Array(vertsPerSphere * 3);
        const meshNormals   = new Float32Array(vertsPerSphere * 3);
        const meshUVs       = new Float32Array(vertsPerSphere * 2);
        let vi = 0, ni = 0, ui = 0;

        for (let lat = 0; lat <= latBands; lat++)
        {
            const theta = lat * Math.PI / latBands;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            for (let lon = 0; lon <= lonBands; lon++)
            {
                const phi = lon * 2 * Math.PI / lonBands;
                const nx = Math.cos(phi) * sinTheta;
                const ny = cosTheta;
                const nz = Math.sin(phi) * sinTheta;

                meshPositions[vi++] = cx + radius * nx;
                meshPositions[vi++] = cy + radius * ny;
                meshPositions[vi++] = cz + radius * nz;
                meshNormals[ni++] = nx;
                meshNormals[ni++] = ny;
                meshNormals[ni++] = nz;
                meshUVs[ui++] = 1 - (lon / lonBands);
                meshUVs[ui++] = 1 - (lat / latBands);
            }
        }

        const triCount = latBands * lonBands * 2;
        const meshIndices = new Uint16Array(triCount * 3);
        let idx = 0;
        for (let lat = 0; lat < latBands; lat++)
        {
            for (let lon = 0; lon < lonBands; lon++)
            {
                const first  = lat * (lonBands + 1) + lon;
                const second = first + lonBands + 1;
                meshIndices[idx++] = first;
                meshIndices[idx++] = first + 1;
                meshIndices[idx++] = second;
                meshIndices[idx++] = second;
                meshIndices[idx++] = first + 1;
                meshIndices[idx++] = second + 1;
            }
        }
        perMeshData.push({ positions: meshPositions, normals: meshNormals, uvs: meshUVs, indices: meshIndices });

        // Add to global flat arrays
        const base = globalVertexOffset;
        allPositions.push(...meshPositions);
        allNormals.push(...meshNormals);
        allUVs.push(...meshUVs);
        for (let t = 0; t < triCount; t++)
        {
            const a = meshIndices[t * 3 + 0];
            const b = meshIndices[t * 3 + 1];
            const c = meshIndices[t * 3 + 2];
            allIndices.push(base + a, base + b, base + c);
            perTriangleMaterialIndices.push(materialIndex);
        }
        globalVertexOffset += vertsPerSphere;
    }

    //================================//
    const worldPositionData           = new Float32Array(allPositions);
    const worldNormalData             = new Float32Array(allNormals);
    const worldUVData                 = new Float32Array(allUVs);
    const worldIndexData              = new Uint32Array(allIndices);
    const perTriangleMaterialTyped    = new Uint32Array(perTriangleMaterialIndices);
    const totalTriangleCount          = allIndices.length / 3;

    return {
        worldPositionData,
        worldNormalData,
        worldUVData,
        worldIndexData,
        perTriangleMaterialIndices: perTriangleMaterialTyped,
        perMeshWorldPositionOffsets,
        materials,
        totalTriangleCount,
        perMeshData,
        sphereCenters,
    };
}
