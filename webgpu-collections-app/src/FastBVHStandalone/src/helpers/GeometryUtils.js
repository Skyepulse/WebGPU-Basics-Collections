import { createDefaultMaterial } from './MaterialUtils.js';

//================================//
// Builds the example scene (ground plane + N spheres) as flat GPU-ready arrays.
// Returns:
//   worldPositionData           : Float32Array  - flat [x,y,z, ...]   (all vertex positions, world space)
//   worldNormalData             : Float32Array  - flat [nx,ny,nz, ...] (all vertex normals)
//   worldUVData                 : Float32Array  - flat [u,v, ...]      (all vertex UVs)
//   worldIndexData              : Uint32Array   - flat [i0,i1,i2, ...] (triangle indices into vertex arrays)
//   perTriangleMaterialIndices  : Uint32Array   - one material index per triangle
//   perMeshWorldPositionOffsets : number[]      - byte offset of each mesh's vertex start in worldPositionData
//   materials                   : Material[]    - one per mesh (ground first, then spheres)
//   totalTriangleCount          : number
//   perMeshData                 : Array<{positions, normals, uvs, indices(Uint16Array)}>
//                                 - per-mesh local (=world since identity transform) data for rasterizer
export async function fastBVHExampleScene(meshMaterials, seed, numSpheres)
{
    // Mulberry32 seeded PRNG
    let s = seed | 0;
    const random = () => {
        s = (s + 0x6D2B79F5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const randomRange = (min, max) => random() * (max - min) + min;

    // Global flat accumulators
    const allPositions = [];
    const allNormals   = [];
    const allUVs       = [];
    const allIndices   = [];
    const perTriangleMaterialIndices = [];
    const perMeshWorldPositionOffsets = [];
    const materials = [];
    const perMeshData = [];

    let globalVertexOffset = 0;

    function currentByteOffset() { return globalVertexOffset * 3 * 4; }

    //================================//
    // ============== GROUND PLANE ============== //
    const planeMin = -100;
    const planeMax = 100;
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

    const g = globalVertexOffset;
    allPositions.push(...planePositions);
    allNormals.push(...planeNormals);
    allUVs.push(...planeUVs);
    allIndices.push(g+0, g+2, g+1);   perTriangleMaterialIndices.push(0);
    allIndices.push(g+0, g+3, g+2);   perTriangleMaterialIndices.push(0);
    globalVertexOffset += 4;

    //================================//
    // ============== SPHERES ============== //
    const latBands = 32;
    const lonBands = 32;
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
    };
}
