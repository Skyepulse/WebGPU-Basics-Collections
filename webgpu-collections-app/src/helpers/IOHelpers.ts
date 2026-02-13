import { load } from '@loaders.gl/core';
import { GLTFLoader, postProcessGLTF, type GLTFMeshPostprocessed, type GLTFPostprocessed } from '@loaders.gl/gltf';
import { Mesh } from './GeometryUtils';
import { createDefaultMaterial } from './MaterialUtils';
import * as glm from 'gl-matrix';

//================================//
export async function loadMesh(url: string) : Promise<Mesh> 
{
    if (!url.endsWith('.gltf') &&  !url.endsWith('.glb'))
    {
        console.error('Unsupported file format. Only .gltf and .glb are supported.');
        return new Mesh("EmptyMesh", createDefaultMaterial({}));
    }

    try
    {
        const gltf = await load(url, GLTFLoader);
        const processedGLTF: GLTFPostprocessed = postProcessGLTF(gltf);

        const meshes: GLTFMeshPostprocessed[] = processedGLTF.meshes;
        const numMeshes = meshes.length;

        if (numMeshes === 0)
        {
            console.warn('No meshes found in the GLTF file.');
            return new Mesh("EmptyMesh", createDefaultMaterial({}));
        }

        const firstMesh = meshes[0];
        const meshName = firstMesh.name || "UnnamedMesh";
        const material = createDefaultMaterial({});

        const mesh = new Mesh(meshName, material);
        
        // Get triangles (make sure we can triangulate the mesh in case the 
        // primitives are not triangles)
        for (const primitive of firstMesh.primitives)
        {
            if (primitive.mode !== undefined && primitive.mode !== 4)
            {
                console.warn(`Skipping non-triangle primitive (mode: ${primitive.mode})`);
                continue;
            }

            const attributes = primitive.attributes;
            const positions = attributes.POSITION?.value as Float32Array | undefined;
            const normals = attributes.NORMAL?.value as Float32Array | undefined;
            const uvs = attributes.TEXCOORD_0?.value as Float32Array | undefined;

            if (!positions)
            {
                console.warn('Primitive has no POSITION attribute, skipping.');
                continue;
            }

            const vertexCount = positions.length / 3;
            const baseVertex = mesh.getNumVertices();

            for (let i = 0; i < vertexCount; ++i)
            {
                const pos = glm.vec3.fromValues(
                    positions[i * 3],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2]
                );

                const normal = normals
                    ? glm.vec3.fromValues(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2])
                    : glm.vec3.fromValues(0, 0, 1);

                const uv = uvs
                    ? glm.vec2.fromValues(uvs[i * 2], uvs[i * 2 + 1])
                    : glm.vec2.fromValues(0, 0);

                mesh.addVertex({ pos, normal, uv });
            }

            const indices = primitive.indices?.value as Uint16Array | Uint32Array | undefined;

            if (indices)
            {
                for (let i = 0; i < indices.length; i += 3)
                {
                    mesh.addTriangle([
                        baseVertex + indices[i],
                        baseVertex + indices[i + 1],
                        baseVertex + indices[i + 2]
                    ]);
                }
            }
            else // NON INDEXED
            {
                for (let i = 0; i < vertexCount; i += 3)
                {
                    mesh.addTriangle([
                        baseVertex + i,
                        baseVertex + i + 1,
                        baseVertex + i + 2
                    ]);
                }
            }
        }

        return mesh;
    }
    catch (error)
    {
        console.error('Error loading mesh:', error);
        return new Mesh("EmptyMesh", createDefaultMaterial({}));
    }
}