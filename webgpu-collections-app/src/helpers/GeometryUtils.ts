import { computeNormal, rotationMatrix3 } from "./MathUtils"
import { type Material, createDefaultMaterial, flattenMaterial } from "./MaterialUtils";
import * as glm from 'gl-matrix';
import { BVH } from "./BVHHelpers";
import { loadMesh } from "./IOHelpers";

//================================//
interface VertexInformation
{
    x: number,
    y: number,
    z?: number,
    r?: number,
    g?: number,
    b?: number
}

//================================//
export interface Transform
{
    translation: glm.vec3,
    rotation: glm.quat,
    scale: glm.vec3
}

//================================//
export interface Ray
{
    origin: glm.vec3,
    direction: glm.vec3
    invDir: glm.vec3
}

//================================//
export interface TopologyInformation
{
    vertexData: Float32Array,
    indexData: Uint16Array,
    numVertices: number

    normalData?: Float32Array
    colorData?: Float32Array
    uvData?: Float32Array
    reflectanceData?: Float32Array

    additionalInfo?: any
    transform?: Transform
}

//================================//
export interface Triangle
{
    vA: vertex,
    vB: vertex,
    vC: vertex,
}

//================================//
export interface vertex
{
    pos: glm.vec3,
    normal: glm.vec3,
    uv: glm.vec2
}

//================================//
export class Mesh
{
    public triangles: Triangle[];
    public vertices: vertex[];
    public indices: number[];

    public Material: Material;

    public name: string;

    private transform: Transform;
    private BVH: BVH;
    private WorldMatrix: glm.mat4;
    private inverseWorldMatrix: glm.mat4;

    //================================//
    constructor(name: string, material: Material)
    {
        this.name = name;
        this.Material = material;

        this.triangles = [];
        this.indices = [];
        this.vertices = [];

        this.BVH = new BVH();

        this.transform = { translation: glm.vec3.create(), rotation: glm.quat.create(), scale: glm.vec3.fromValues(1, 1, 1) };

        this.WorldMatrix = glm.mat4.create();
        this.inverseWorldMatrix = glm.mat4.create();
    }

    //================================//
    public TransformMesh(transform: Transform): void
    {
        this.transform = transform;
        this.computeMatrices();
    }

    //================================//
    public RotateMesh(rotation: glm.quat): void
    {
        glm.quat.multiply(this.transform.rotation, this.transform.rotation, rotation);
        this.computeMatrices();
    }

    //================================//
    private computeMatrices(): void
    {
        this.WorldMatrix = glm.mat4.create();
        glm.mat4.fromRotationTranslationScale(
            this.WorldMatrix,
            this.transform.rotation,      // quat, not euler
            this.transform.translation,
            this.transform.scale
        );

        this.inverseWorldMatrix = glm.mat4.create();
        glm.mat4.invert(this.inverseWorldMatrix, this.WorldMatrix);
    }

    //================================//
    public GetWorldMatrix(): glm.mat4
    {
        return this.WorldMatrix;
    }

    //================================//
    public GetInverseWorldMatrix(): glm.mat4
    {
        return this.inverseWorldMatrix;
    }

    //================================//
    public GetFlatWorldMatrix(): Float32Array
    {
        return new Float32Array(this.WorldMatrix);
    }

    //================================//
    public GetFlatNormalMatrix(): Float32Array
    {
        const normalMatrix = glm.mat3.create();
        glm.mat3.normalFromMat4(normalMatrix, this.WorldMatrix);

        const padded = new Float32Array(12);
        padded[0]  = normalMatrix[0]; padded[1]  = normalMatrix[1]; padded[2]  = normalMatrix[2];
        padded[4]  = normalMatrix[3]; padded[5]  = normalMatrix[4]; padded[6]  = normalMatrix[5];
        padded[8]  = normalMatrix[6]; padded[9]  = normalMatrix[7]; padded[10] = normalMatrix[8];

        return padded;
    }

    //================================//
    public GetFlatInverseWorldMatrix(): Float32Array
    {
        return new Float32Array(this.inverseWorldMatrix);
    }

    //================================//
    public GetTransform(): Transform
    {
        return this.transform;
    }

    //================================//
    public GetMaterial(): Material
    {
        return this.Material;
    }

    //================================//
    public GetFlattenedMaterial(): Float32Array
    {
        return flattenMaterial(this.Material);
    }

    //================================//
    public addVertex(vertex: vertex): number
    {
        this.vertices.push(vertex);
        return this.vertices.length - 1;
    }

    //================================//
    public addTriangle(indices: number[]): void
    {
        if (indices.length !== 3) return;

        const tri: Triangle = 
        {
            vA: this.vertices[indices[0]],
            vB: this.vertices[indices[1]],
            vC: this.vertices[indices[2]]
        }

        this.triangles.push(tri);
        this.indices.push(...indices);
    }

    //================================//
    public getVertexData(): Float32Array
    {
        const data = Array(this.vertices.length * 3);
        const float32View = new Float32Array(data);

        for (let i = 0; i < this.vertices.length; ++i)
        {
            const pos = this.vertices[i].pos;
            float32View.set(pos, i * 3);
        }

        return float32View;
    }

    //================================//
    public getTransformedVertexData(): Float32Array
    {
        const float32View = new Float32Array(this.vertices.length * 3);
        const temp = glm.vec3.create();

        for (let i = 0; i < this.vertices.length; ++i)
        {
            glm.vec3.transformMat4(temp, this.vertices[i].pos, this.WorldMatrix);
            float32View.set(temp, i * 3);
        }

        return float32View;
    }
    //================================//
    public getNormalData(): Float32Array
    {
        const data = Array(this.vertices.length * 3);
        const float32View = new Float32Array(data);

        for (let i = 0; i < this.vertices.length; ++i)
        {
            float32View.set(this.vertices[i].normal, i * 3);
        }

        return float32View;
    }

    //================================//
    public getTransformedNormalData(): Float32Array
    {
        const float32View = new Float32Array(this.vertices.length * 3);
        const normalMat = glm.mat3.create();
        glm.mat3.normalFromMat4(normalMat, this.WorldMatrix);
        const temp = glm.vec3.create();

        for (let i = 0; i < this.vertices.length; ++i)
        {
            glm.vec3.transformMat3(temp, this.vertices[i].normal, normalMat);
            glm.vec3.normalize(temp, temp);
            float32View.set(temp, i * 3);
        }

        return float32View;
    }

    //================================//
    public getUVData(): Float32Array
    {
        const data = Array(this.vertices.length * 2);
        const float32View = new Float32Array(data);

        for (let i = 0; i < this.vertices.length; ++i)
            float32View.set(this.vertices[i].uv, i * 2);

        return float32View;
    }

    //================================//
    public getIndexData16(): Uint16Array
    {
        return new Uint16Array(this.indices);
    }

    //================================//
    public getIndexData32(): Uint32Array
    {
    return new Uint32Array(this.indices);
    }

    //================================//
    public getNumVertices(): number
    {
        return this.vertices.length;
    }

    //================================//
    public getNumTriangles(): number
    {
        return this.triangles.length;
    }

    //================================//
    public getTriangles(): Triangle[]
    {
        return this.triangles;
    }

    //================================//
    public ComputeBVH(): void
    {
        this.BVH.buildBVH(this);
    }

    //================================//
    public GetBVHGeometry(maxDepth: number = Infinity): { vertexData: Float32Array, count: number }
    {
        return this.BVH.generateWireframeGeometry(maxDepth);
    }

    //================================//
    public getFlattenedBVHData(nodeOffset: number = 0): { data: ArrayBuffer, numNodes: number }
    {
        return this.BVH.getFlattenedBVHData(nodeOffset);
    }

    //================================//
    intersectMeshWithRay(ray: Ray, depth: number): number
    {
        const localRayOrigin = glm.vec3.create();
        glm.vec3.transformMat4(localRayOrigin, ray.origin, this.GetInverseWorldMatrix());

        const localRayDirection = glm.vec3.create();
        const invMat3 = glm.mat3.create(); 
        glm.mat3.fromMat4(invMat3, this.GetInverseWorldMatrix()); // No translation for direction
        glm.vec3.transformMat3(localRayDirection, ray.direction, invMat3);

        const localRay: Ray = {
            origin: localRayOrigin,
            direction: localRayDirection,
            invDir: glm.vec3.fromValues(1 / localRayDirection[0], 1 / localRayDirection[1], 1 / localRayDirection[2])
        }

        const dist = this.BVH.traverse(localRay, depth);

        return dist;
    }

    //================================//
    public getReorderedIndexData32(): Uint32Array
    {
        return this.BVH.getReorderedIndices(this.indices);
    }
}

//================================//
export interface SceneInformation
{
    meshes: Mesh[],

    additionalInfo?: any
}

//================================//
export function createQuadVertices(): TopologyInformation
{
    // Two triangles for a Quad
    const numVertices = 4;
    const vertexData: Float32Array = new Float32Array(numVertices * 2); //position only
    
    let offset = 0;
    const addVertex = (vertex: VertexInformation) => {
        vertexData[offset++] = vertex.x;
        vertexData[offset++] = vertex.y;
    };

    addVertex({ x: -0.5, y: -0.5 }); // Bottom left
    addVertex({ x:  0.5, y: -0.5 }); // Bottom right
    addVertex({ x: -0.5, y:  0.5 }); // Top left
    addVertex({ x:  0.5, y:  0.5 }); // Top right

    const indexData = new Uint16Array([
        0, 1, 2, // First triangle
        2, 1, 3  // Second triangle
    ]);

    return {
        vertexData,
        indexData,
        numVertices: indexData.length
    };
}

//================================//
// This method contains optimization for color usage and also index buffer optimization
export function createCircleVerticesWithColor(
{
    radius = 1,
    subdivisions = 24,
    innerRadius = 0,
    startAngle = 0,
    endAngle = Math.PI * 2
} = {}): TopologyInformation
{
    const numVertices = (subdivisions + 1) * 2;
    const vertexData: Float32Array = new Float32Array(numVertices * (2 + 1)); // position + color per vertex 8 x (4 here) bit
    const colorData: Uint8Array = new Uint8Array(vertexData.buffer); // This is a 8 bit per channel view of the 32 bit channel float buffer

    let offset = 0;
    let colorOffset = 8;
    const addVertex = (vertex: VertexInformation) => {
        vertexData[offset++] = vertex.x;
        vertexData[offset++] = vertex.y;
        offset+=1; // Skip the color (1 byte, 8 bits)
        colorData[colorOffset++] = (vertex.r ?? 0) * 255;
        colorData[colorOffset++] = (vertex.g ?? 0) * 255;
        colorData[colorOffset++] = (vertex.b ?? 0) * 255;
        colorOffset += 9; // Skip the remaining byte and position (1 byte + 8 bytes)
    };

    const innerColor = [1.0, 1.0, 1.0];
    const outerColor = [0.1, 0.1, 0.1];

    // 2 triangles per subdivision
    //
    // 0--2
    // | /|
    // |/ |
    // 1--3
    // 
    // Up to down per angle

    for (let i = 0; i <= subdivisions; i++) 
    {
        const angle = startAngle + (i + 0) * (endAngle - startAngle) / subdivisions;

        const c1 = Math.cos(angle);
        const s1 = Math.sin(angle);

        addVertex({ x: c1 * radius, y: s1 * radius, r: outerColor[0], g: outerColor[1], b: outerColor[2] });
        addVertex({ x: c1 * innerRadius, y: s1 * innerRadius, r: innerColor[0], g: innerColor[1], b: innerColor[2] });
    }

    const indexData = new Uint16Array(subdivisions * 6);
    let index = 0;

    // 1st tri  2nd tri  3rd tri  4th tri
    // 0 1 2    2 1 3    2 3 4    4 3 5
    //
    // 0--2        2     2--4        4  .....
    // | /        /|     | /        /|
    // |/        / |     |/        / |
    // 1        1--3     3        3--5  .....
    for (let i = 0; i < subdivisions; ++i) {
        const offset = i * 2;

        // Triangle One
        indexData[index++] = offset;
        indexData[index++] = offset + 1;
        indexData[index++] = offset + 2;

        // Triangle Two
        indexData[index++] = offset + 2;
        indexData[index++] = offset + 1;
        indexData[index++] = offset + 3;
    }

    return {
        vertexData,
        indexData,
        numVertices: indexData.length
    };
}

//================================//
export function createCircleVerticesTopology(
{
    radius = 1,
    subdivisions = 24,
    innerRadius = 0,
    startAngle = 0,
    endAngle = Math.PI * 2
} = {}): TopologyInformation
{
    const numVertices = (subdivisions + 1) * 2;
    const vertexData: Float32Array = new Float32Array(numVertices * 2);

    let offset = 0;
    const addVertex = (vertex: VertexInformation) => {
        vertexData[offset++] = vertex.x;
        vertexData[offset++] = vertex.y;
    };

    // 2 triangles per subdivision
    //
    // 0--2
    // | /|
    // |/ |
    // 1--3
    // 
    // Up to down per angle

    for (let i = 0; i <= subdivisions; i++) 
    {
        const angle = startAngle + (i + 0) * (endAngle - startAngle) / subdivisions;

        const c1 = Math.cos(angle);
        const s1 = Math.sin(angle);

        addVertex({ x: c1 * radius, y: s1 * radius});
        addVertex({ x: c1 * innerRadius, y: s1 * innerRadius});
    }

    const indexData = new Uint16Array(subdivisions * 6);
    let index = 0;

    // 1st tri  2nd tri  3rd tri  4th tri
    // 0 1 2    2 1 3    2 3 4    4 3 5
    //
    // 0--2        2     2--4        4  .....
    // | /        /|     | /        /|
    // |/        / |     |/        / |
    // 1        1--3     3        3--5  .....
    for (let i = 0; i < subdivisions; ++i) {
        const offset = i * 2;

        // Triangle One
        indexData[index++] = offset;
        indexData[index++] = offset + 1;
        indexData[index++] = offset + 2;

        // Triangle Two
        indexData[index++] = offset + 2;
        indexData[index++] = offset + 1;
        indexData[index++] = offset + 3;
    }

    return {
        vertexData,
        indexData,
        numVertices: indexData.length
    };
}


//================================//
export function createCircleVertices(
{
    radius = 1,
    subdivisions = 24,
    innerRadius = 0,
    startAngle = 0,
    endAngle = Math.PI * 2
} = {}): Float32Array
{
    const numVertices = subdivisions * 3 * 2;
    const vertexData: Float32Array = new Float32Array(numVertices * 2);

    let offset = 0;
    const addVertex = (x: number, y: number) => {
        vertexData[offset++] = x;
        vertexData[offset++] = y;
    };

    // 2 triangles per subdivision
    //
    // 0--1 4
    // | / /|
    // |/ / |
    // 2 3--5

    for (let i = 0; i < subdivisions; i++) 
    {
        const angle1 = startAngle + (i + 0) * (endAngle - startAngle) / subdivisions;
        const angle2 = startAngle + (i + 1) * (endAngle - startAngle) / subdivisions;

        const c1 = Math.cos(angle1);
        const s1 = Math.sin(angle1);
        const c2 = Math.cos(angle2);
        const s2 = Math.sin(angle2);

        // Outer circle vertices
        addVertex(c1 * radius, s1 * radius);
        addVertex(c2 * radius, s2 * radius);
        addVertex(c1 * innerRadius, s1 * innerRadius);

        // Inner circle vertices
        addVertex(c1 * innerRadius, s1 * innerRadius);
        addVertex(c2 * radius, s2 * radius);
        addVertex(c2 * innerRadius, s2 * innerRadius);
    }

    return vertexData;
}

//================================//
export function createCornellBox(): TopologyInformation
{    
    // Colors
    const white: [number, number, number] = [0.73, 0.73, 0.73];
    const red: [number, number, number] = [0.65, 0.05, 0.05];
    const green: [number, number, number] = [0.12, 0.45, 0.15];
    const light: [number, number, number] = [1.0, 1.0, 1.0];
    
    // Temporary arrays to collect data
    const vertices: number[] = [];
    const normals: number[] = [];
    const colors: number[] = [];
    const reflectances: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];
    
    let vertexCount = 0;
    
    //================================//
    function addVertex(
        position: [number, number, number],
        normal: [number, number, number],
        color: [number, number, number],
        uv: [number, number],
        reflectance: number = 0.0
    ): number 
    {
        vertices.push(
            position[0],
            position[1],
            position[2]
        );
        normals.push(normal[0], normal[1], normal[2]);
        colors.push(color[0], color[1], color[2]);
        uvs.push(uv[0], uv[1]);
        reflectances.push(reflectance);
        return vertexCount++;
    }
    
    //================================//
    function addQuad(
        v0: [number, number, number],
        v1: [number, number, number],
        v2: [number, number, number],
        v3: [number, number, number],
        color: [number, number, number],
        flipNormal: boolean = false,
        reflectance: number = 0.0
    ): void 
    {
        let normal = computeNormal(v0, v1, v2);
        if (flipNormal) {
            normal = glm.vec3.fromValues(-normal[0], -normal[1], -normal[2]);
        }
        
        const i0 = addVertex(v0, [normal[0], normal[1], normal[2]], color, [0, 0], reflectance);
        const i1 = addVertex(v1, [normal[0], normal[1], normal[2]], color, [1, 0], reflectance);
        const i2 = addVertex(v2, [normal[0], normal[1], normal[2]], color, [1, 1], reflectance);
        const i3 = addVertex(v3, [normal[0], normal[1], normal[2]], color, [0, 1], reflectance);

        indices.push(i0, i1, i2);
        indices.push(i0, i2, i3);
    }

    //================================//
    function addCube(
        center: [number, number, number],
        size: [number, number, number],
        color: [number, number, number],
        rotation: [number, number, number] = [0, 0, 0], // in radians
        reflectance: number = 0.0
    ): void
    {
        const hx = size[0] / 2;
        const hy = size[1] / 2;
        const hz = size[2] / 2;

        
        let v0: [number, number, number] = [center[0] - hx, center[1] - hy, center[2] - hz];
        let v1: [number, number, number] = [center[0] + hx, center[1] - hy, center[2] - hz];
        let v2: [number, number, number] = [center[0] + hx, center[1] + hy, center[2] - hz];
        let v3: [number, number, number] = [center[0] - hx, center[1] + hy, center[2] - hz];
        let v4: [number, number, number] = [center[0] - hx, center[1] - hy, center[2] + hz];
        let v5: [number, number, number] = [center[0] + hx, center[1] - hy, center[2] + hz];
        let v6: [number, number, number] = [center[0] + hx, center[1] + hy, center[2] + hz];
        let v7: [number, number, number] = [center[0] - hx, center[1] + hy, center[2] + hz];

        // Apply rotation to the vertices
        const rotMat: Float32Array = new Float32Array(9);
        const cx = Math.cos(rotation[0]);
        const sx = Math.sin(rotation[0]);
        const cy = Math.cos(rotation[1]);
        const sy = Math.sin(rotation[1]);
        const cz = Math.cos(rotation[2]);
        const sz = Math.sin(rotation[2]);

        rotMat[0] = cy * cz;
        rotMat[1] = -cy * sz;
        rotMat[2] = sy;
        rotMat[3] = sx * sy * cz + cx * sz;
        rotMat[4] = -sx * sy * sz + cx * cz;
        rotMat[5] = -sx * cy;
        rotMat[6] = -cx * sy * cz + sx * sz;
        rotMat[7] = cx * sy * sz + sx * cz;
        rotMat[8] = cx * cy;

        const rotateVertex = (v: [number, number, number]): [number, number, number] => 
        {
            const x = v[0] - center[0];
            const y = v[1] - center[1];
            const z = v[2] - center[2];

            return [
                rotMat[0] * x + rotMat[1] * y + rotMat[2] * z + center[0],
                rotMat[3] * x + rotMat[4] * y + rotMat[5] * z + center[1],
                rotMat[6] * x + rotMat[7] * y + rotMat[8] * z + center[2]
            ];
        }

        v0 = rotateVertex(v0);
        v1 = rotateVertex(v1);
        v2 = rotateVertex(v2);
        v3 = rotateVertex(v3);
        v4 = rotateVertex(v4);
        v5 = rotateVertex(v5);
        v6 = rotateVertex(v6);
        v7 = rotateVertex(v7);
        
        // front
        addQuad(v4, v5, v6, v7, color, false, reflectance);
        // back
        addQuad(v1, v0, v3, v2, color, false, reflectance);
        // left
        addQuad(v0, v4, v7, v3, color, false, reflectance);
        // right
        addQuad(v5, v1, v2, v6, color, false, reflectance);
        // top
        addQuad(v3, v7, v6, v2, color, false, reflectance);
        // bottom
        addQuad(v0, v1, v5, v4, color, false, reflectance);
    }

    /*
    //================================//
    // Simple sphere
    function addSphere(
        center: [number, number, number],
        radius: number,
        color: [number, number, number],
        latitudeBands: number = 12,
        longitudeBands: number = 12,
        reflectance: number = 0.0
    ): void 
    {
        const startIndex = vertexCount;

        for (let latNumber = 0; latNumber <= latitudeBands; latNumber++)
        {
            const theta = latNumber * Math.PI / latitudeBands;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            for (let longNumber = 0; longNumber <= longitudeBands; longNumber++)
            {
                const phi = longNumber * 2 * Math.PI / longitudeBands;
                const sinPhi = Math.sin(phi);
                const cosPhi = Math.cos(phi);

                const x = cosPhi * sinTheta;
                const y = cosTheta;
                const z = sinPhi * sinTheta;
                const u = 1 - (longNumber / longitudeBands);
                const v = 1 - (latNumber / latitudeBands);

                const vertexPosition: [number, number, number] = [
                    center[0] + radius * x,
                    center[1] + radius * y,
                    center[2] + radius * z
                ];

                addVertex(vertexPosition, [x, y, z], color, [u, v], reflectance);
            }
        }

        for (let latNumber = 0; latNumber < latitudeBands; latNumber++)
        {
            for (let longNumber = 0; longNumber < longitudeBands; longNumber++)
            {
                const first = startIndex + (latNumber * (longitudeBands + 1)) + longNumber;
                const second = first + longitudeBands + 1;

                indices.push(first, first + 1, second); // tri 1
                indices.push(second, first + 1, second + 1); // tri 2
            }
        }
    }  
    */

    // ============== FLOOR (white) ============== //
    addQuad(
        [552.8, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 559.2],
        [549.6, 0.0, 559.2],
        white,
        false,
        0.98
    );
    
    // ============== CEILING (white) ============== //
    addQuad(
        [556.0, 548.8, 0.0],
        [556.0, 548.8, 559.2],
        [0.0, 548.8, 559.2],
        [0.0, 548.8, 0.0],
        white,
        false,
        0.98
    );
    
    // ============== LIGHT (slightly below ceiling) ============== //
    const lightEpsilon = 1.0;
    const lightY = 548.8 - lightEpsilon;
    addQuad(
        [343.0, lightY, 227.0],
        [343.0, lightY, 332.0],
        [213.0, lightY, 332.0],
        [213.0, lightY, 227.0],
        light
    );
    
    // ============== BACK WALL (white) ============== //
    addQuad(
        [549.6, 0.0, 559.2],
        [0.0, 0.0, 559.2],
        [0.0, 548.8, 559.2],
        [556.0, 548.8, 559.2],
        white,
    );
    
    // ============== RIGHT WALL (green) ============== //
    addQuad(
        [0.0, 0.0, 559.2],
        [0.0, 0.0, 0.0],
        [0.0, 548.8, 0.0],
        [0.0, 548.8, 559.2],
        green
    );
    
    // ============== LEFT WALL (red) ============== //
    addQuad(
        [552.8, 0.0, 0.0],
        [549.6, 0.0, 559.2],
        [556.0, 548.8, 559.2],
        [556.0, 548.8, 0.0],
        red
    );

    let startCubeVertex = vertexCount;
    addCube(
        [278.0, 224.4, 279.5],
        [120.0, 120.0, 120.0],
        white,
        [4, Math.PI / 9, 7],
        1.0
    );
    let numVerticesCube = vertexCount - startCubeVertex;

    /*
    // Add Sphere in middle of room
    addSphere(
        [278.0, 224.4, 279.5],
        120.0,
        white,
        64,
        64,
        1.0
    );
    */
    
    /*
    // ============== SHORT BLOCK (white) ============== //
    // Top
    addQuad(
        [130.0, 165.0, 65.0],
        [82.0, 165.0, 225.0],
        [240.0, 165.0, 272.0],
        [290.0, 165.0, 114.0],
        white,
    );
    
    // Front face
    addQuad(
        [290.0, 0.0, 114.0],
        [290.0, 165.0, 114.0],
        [240.0, 165.0, 272.0],
        [240.0, 0.0, 272.0],
        white
    );
    
    // Right face
    addQuad(
        [130.0, 0.0, 65.0],
        [130.0, 165.0, 65.0],
        [290.0, 165.0, 114.0],
        [290.0, 0.0, 114.0],
        white
    );
    
    // Back face
    addQuad(
        [82.0, 0.0, 225.0],
        [82.0, 165.0, 225.0],
        [130.0, 165.0, 65.0],
        [130.0, 0.0, 65.0],
        white,
    );
    
    // Left face
    addQuad(
        [240.0, 0.0, 272.0],
        [240.0, 165.0, 272.0],
        [82.0, 165.0, 225.0],
        [82.0, 0.0, 225.0],
        white
    );
    
    
    // ============== TALL BLOCK (white) ============== //
    // Top
    addQuad(
        [423.0, 330.0, 247.0],
        [265.0, 330.0, 296.0],
        [314.0, 330.0, 456.0],
        [472.0, 330.0, 406.0],
        white,
    );
    
    // Front face
    addQuad(
        [423.0, 0.0, 247.0],
        [423.0, 330.0, 247.0],
        [472.0, 330.0, 406.0],
        [472.0, 0.0, 406.0],
        white,
    );
    
    // Right face
    addQuad(
        [472.0, 0.0, 406.0],
        [472.0, 330.0, 406.0],
        [314.0, 330.0, 456.0],
        [314.0, 0.0, 456.0],
        white,
    );
    
    // Back face
    addQuad(
        [314.0, 0.0, 456.0],
        [314.0, 330.0, 456.0],
        [265.0, 330.0, 296.0],
        [265.0, 0.0, 296.0],
        white,
    );
    
    // Left face
    addQuad(
        [265.0, 0.0, 296.0],
        [265.0, 330.0, 296.0],
        [423.0, 330.0, 247.0],
        [423.0, 0.0, 247.0],
        white,
    );
    */
    
    return {
        vertexData: new Float32Array(vertices),
        indexData: new Uint16Array(indices),
        numVertices: indices.length,
        normalData: new Float32Array(normals),
        colorData: new Float32Array(colors),
        reflectanceData: new Float32Array(reflectances),
        uvData: new Float32Array(uvs),
        additionalInfo: {
            cubeVertexStart: startCubeVertex,
            cubeVertexCount: numVerticesCube,
            cubeCenter: [278.0, 224.4, 279.5],
            cubeVertexInfo: new Float32Array(vertices.slice(startCubeVertex * 3, (startCubeVertex + numVerticesCube) * 3)),
            cubeNormalsInfo: new Float32Array(normals.slice(startCubeVertex * 3, (startCubeVertex + numVerticesCube) * 3))
        }
    };
}

//================================//
export function createQuad(transform: Transform, color: [number, number, number]): TopologyInformation
{
    let numVertices = 4;
    const vertexData: Float32Array = new Float32Array(numVertices * 3);
    const colorData = new Float32Array(numVertices * 3);
    const normalData = new Float32Array(numVertices * 3);
    const uvData = new Float32Array(numVertices * 2);
    const indexData = new Uint16Array([0, 1, 2, 0, 2, 3]);

    const center = transform.translation;
    const hx = transform.scale[0] / 2;
    const hy = transform.scale[1] / 2;
    const rotation = transform.rotation;

    const corners: glm.vec3[] = [
        glm.vec3.fromValues(-hx, -hy, 0),
        glm.vec3.fromValues( hx, -hy, 0),
        glm.vec3.fromValues( hx,  hy, 0),
        glm.vec3.fromValues(-hx,  hy, 0)
    ];

    const rotMat: glm.mat3 = rotationMatrix3(rotation[0], rotation[1], rotation[2]);
    for (let i = 0; i < corners.length; ++i) {
        // Rotate
        glm.vec3.transformMat3(corners[i], corners[i], rotMat);
        // Translate
        glm.vec3.add(corners[i], corners[i], center);
    }

    let offset = 0;
    const addVertex = (vertex: glm.vec3, color: [number, number, number]) => {
        vertexData[offset]     = vertex[0];
        vertexData[offset + 1] = vertex[1];
        vertexData[offset + 2] = vertex[2];

        colorData[offset]      = color[0];
        colorData[offset + 1]  = color[1];
        colorData[offset + 2]  = color[2];
        offset += 3;
    };

    addVertex(corners[0], color);
    addVertex(corners[1], color);
    addVertex(corners[2], color);
    addVertex(corners[3], color);

    // normals
    const normal = glm.vec3.fromValues(0, 0, 1);
    // Rotate normal
    glm.vec3.transformMat3(normal, normal, rotMat);
    for (let i = 0; i < numVertices; ++i) {
        normalData[i * 3 + 0] = normal[0];
        normalData[i * 3 + 1] = normal[1];
        normalData[i * 3 + 2] = normal[2];
    }

    // uvs
    uvData[0] = 0; uvData[1] = 0;
    uvData[2] = 1; uvData[3] = 0;
    uvData[4] = 1; uvData[5] = 1;
    uvData[6] = 0; uvData[7] = 1;

    return {
        vertexData: vertexData,
        indexData: indexData,
        colorData: colorData,
        normalData: normalData,
        uvData: uvData,
        numVertices: indexData.length,
        transform: transform
    };
}

//================================//
export function createSphere(
        center: [number, number, number],
        radius: number,
        color: [number, number, number],
        latitudeBands: number = 12,
        longitudeBands: number = 12
    ): TopologyInformation
{
    const vertices: number[] = [];
    const normals: number[] = [];
    const colors: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];

    const addVertex = (
        position: [number, number, number],
        normal: [number, number, number],
        color: [number, number, number],
        uv: [number, number]) =>
    {
        vertices.push(
            position[0],
            position[1],
            position[2]
        );
        normals.push(normal[0], normal[1], normal[2]);
        colors.push(color[0], color[1], color[2]);
        uvs.push(uv[0], uv[1]);
    }

    for (let latNumber = 0; latNumber <= latitudeBands; latNumber++)
    {
        const theta = latNumber * Math.PI / latitudeBands;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);

        for (let longNumber = 0; longNumber <= longitudeBands; longNumber++)
        {
            const phi = longNumber * 2 * Math.PI / longitudeBands;
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            const x = cosPhi * sinTheta;
            const y = cosTheta;
            const z = sinPhi * sinTheta;
            const u = 1 - (longNumber / longitudeBands);
            const v = 1 - (latNumber / latitudeBands);

            const vertexPosition: [number, number, number] = [
                center[0] + radius * x,
                center[1] + radius * y,
                center[2] + radius * z
            ];

            addVertex(vertexPosition, [x, y, z], color, [u, v]);
        }
    }

    for (let latNumber = 0; latNumber < latitudeBands; latNumber++)
    {
        for (let longNumber = 0; longNumber < longitudeBands; longNumber++)
        {
            const first = (latNumber * (longitudeBands + 1)) + longNumber;
            const second = first + longitudeBands + 1;

            indices.push(first, first + 1, second); // tri 1
            indices.push(second, first + 1, second + 1); // tri 2
        }
    }

    return {
        vertexData: new Float32Array(vertices),
        indexData: new Uint16Array(indices),
        numVertices: indices.length,
        normalData: new Float32Array(normals),
        colorData: new Float32Array(colors),
        uvData: new Float32Array(uvs),
        transform: {
            translation: glm.vec3.fromValues(center[0], center[1], center[2]),
            rotation: glm.vec3.fromValues(0, 0, 0),
            scale: glm.vec3.fromValues(radius, radius, radius)
        }
    };
}

//================================//
export function createCornellBox2(sphereMaterials: Material[], sphereResolution: number = 8): SceneInformation
{   
    const Meshes: Mesh[] = [];

    Meshes.push(new Mesh("white wall", createDefaultMaterial({ albedo: [0.73, 0.73, 0.73], name: "whiteWall" })));
    Meshes.push(new Mesh("red wall", createDefaultMaterial({ albedo: [0.65, 0.05, 0.05], name: "redWall" })));
    Meshes.push(new Mesh("green wall", createDefaultMaterial({ albedo: [0.12, 0.45, 0.15], name: "greenWall" })));
    Meshes.push(new Mesh("light", createDefaultMaterial({ albedo: [1.0, 1.0, 1.0], roughness: 0.0, name: "light" })));
    Meshes.push(new Mesh("sphereOne", 
        sphereMaterials.find(mat => mat.name === "sphereOne") || createDefaultMaterial({
        albedo: [0.12, 0.45, 0.15],
        name: "sphereOne",
        textureIndex: 0
    })));
    Meshes.push(new Mesh("sphereTwo",
        sphereMaterials.find(mat => mat.name === "sphereTwo") || createDefaultMaterial({
        albedo: [0.05, 0.05, 0.65],
        roughness: 0.5,
        metalness: 0.5,
        name: "sphereTwo",
        textureIndex: 1
    })));
    Meshes.push(new Mesh("sphereThree",
        sphereMaterials.find(mat => mat.name === "sphereThree") || createDefaultMaterial({
        albedo: [0.65, 0.05, 0.05],
        roughness: 0.01,
        metalness: 0.98,
        name: "sphereThree",
        textureIndex: 2
    })));

    //================================//
    function addVertex(
        mesh: Mesh,
        position: glm.vec3,
        normal: glm.vec3,
        uv: glm.vec2,
    ): void 
    {
        const vertex: vertex = {pos: position, normal: normal, uv: uv };
        mesh.addVertex(vertex);
    }
    
    //================================//
    function addQuad(
        Mesh: Mesh,
        v0: glm.vec3,
        v1: glm.vec3,
        v2: glm.vec3,
        v3: glm.vec3,
        flipNormal: boolean = false
    ): void 
    {
        let normal = computeNormal(v0, v1, v2);
        if (flipNormal)
            normal = glm.vec3.fromValues(-normal[0], -normal[1], -normal[2]);
        
        const i0 = Mesh.addVertex({pos: v0, normal: normal, uv: glm.vec2.fromValues(0, 0)});
        const i1 = Mesh.addVertex({pos: v1, normal: normal, uv: glm.vec2.fromValues(1, 0)});
        const i2 = Mesh.addVertex({pos: v2, normal: normal, uv: glm.vec2.fromValues(1, 1)});
        const i3 = Mesh.addVertex({pos: v3, normal: normal, uv: glm.vec2.fromValues(0, 1)});

        Mesh.addTriangle([i0, i1, i2]);
        Mesh.addTriangle([i0, i2, i3]);
    }

    //================================//
    function addSphere(
        Mesh: Mesh,
        center: glm.vec3,
        radius: number,
        latitudeBands: number = 12,
        longitudeBands: number = 12
    ): void 
    {
        const startIndex = Mesh.getNumVertices();
        for (let latNumber = 0; latNumber <= latitudeBands; latNumber++)
        {
            const theta = latNumber * Math.PI / latitudeBands;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            for (let longNumber = 0; longNumber <= longitudeBands; longNumber++)
            {
                const phi = longNumber * 2 * Math.PI / longitudeBands;
                const sinPhi = Math.sin(phi);
                const cosPhi = Math.cos(phi);

                const x = cosPhi * sinTheta;
                const y = cosTheta;
                const z = sinPhi * sinTheta;
                const u = 1 - (longNumber / longitudeBands);
                const v = 1 - (latNumber / latitudeBands);

                const vertexPosition: glm.vec3 = glm.vec3.fromValues(
                    center[0] + radius * x,
                    center[1] + radius * y,
                    center[2] + radius * z
                );

                addVertex(Mesh, vertexPosition, glm.vec3.fromValues(x, y, z), glm.vec2.fromValues(u, v));
            }
        }

        for (let latNumber = 0; latNumber < latitudeBands; latNumber++)
        {
            for (let longNumber = 0; longNumber < longitudeBands; longNumber++)
            {
                const first = startIndex + (latNumber * (longitudeBands + 1)) + longNumber;
                const second = first + longitudeBands + 1;

                Mesh.addTriangle([first, first + 1, second]); // tri 1
                Mesh.addTriangle([second, first + 1, second + 1]); // tri 2
            }
        }
    } 

    // ============== FLOOR (white) ============== //
    addQuad(
        Meshes[0],
        glm.vec3.fromValues(552.8, 0.0, 0.0),
        glm.vec3.fromValues(0.0, 0.0, 0.0),
        glm.vec3.fromValues(0.0, 0.0, 559.2),
        glm.vec3.fromValues(549.6, 0.0, 559.2),
        false,
    );
    
    // ============== CEILING (white) ============== //
    addQuad(
        Meshes[0],
        glm.vec3.fromValues(556.0, 548.8, 0.0),
        glm.vec3.fromValues(556.0, 548.8, 559.2),
        glm.vec3.fromValues(0.0, 548.8, 559.2),
        glm.vec3.fromValues(0.0, 548.8, 0.0),
        false,
    );
    
    // ============== LIGHT (slightly below ceiling) ============== //
    const lightEpsilon = 1.0;
    const lightY = 548.8 - lightEpsilon;
    addQuad(
        Meshes[3],
        glm.vec3.fromValues(343.0, lightY, 227.0),
        glm.vec3.fromValues(343.0, lightY, 332.0),
        glm.vec3.fromValues(213.0, lightY, 332.0),
        glm.vec3.fromValues(213.0, lightY, 227.0),
        false,
    );
    
    // ============== BACK WALL (white) ============== //
    addQuad(
        Meshes[0],
        glm.vec3.fromValues(549.6, 0.0, 559.2),
        glm.vec3.fromValues(0.0, 0.0, 559.2),
        glm.vec3.fromValues(0.0, 548.8, 559.2),
        glm.vec3.fromValues(556.0, 548.8, 559.2),
        false,
    );
    
    // ============== RIGHT WALL (green) ============== //
    addQuad(
        Meshes[2],
        glm.vec3.fromValues(0.0, 0.0, 559.2),
        glm.vec3.fromValues(0.0, 0.0, 0.0),
        glm.vec3.fromValues(0.0, 548.8, 0.0),
        glm.vec3.fromValues(0.0, 548.8, 559.2),
        false
    );
    
    // ============== LEFT WALL (red) ============== //
    addQuad(
        Meshes[1],
        glm.vec3.fromValues(552.8, 0.0, 0.0),
        glm.vec3.fromValues(549.6, 0.0, 559.2),
        glm.vec3.fromValues(556.0, 548.8, 559.2),
        glm.vec3.fromValues(556.0, 548.8, 0.0),
        false,
    );

    // Middle
    let middleCubeCenter: [number, number, number] = [278.0, 224.4, 279.5];
    let sphereRadius = 90.0;
    let distanceToCenter = 120.0;

    let directions: glm.vec3[] = [
        glm.vec3.fromValues(0, 1, 0),                      // 0°
        glm.vec3.fromValues(Math.sqrt(3)/2, -0.5, 0),      // 120°
        glm.vec3.fromValues(-Math.sqrt(3)/2, -0.5, 0)      // 240°
    ];

    for (let i = 0; i < 3; ++i) 
    {
        addSphere(
            Meshes[i + 4],
            [0, 0, 0],
            1.0,
            sphereResolution,
            sphereResolution
        );
    }

    Meshes[4].TransformMesh(
    {
        translation: glm.vec3.fromValues(
            middleCubeCenter[0] + directions[0][0] * distanceToCenter,
            middleCubeCenter[1] + directions[0][1] * distanceToCenter,
            middleCubeCenter[2] + directions[0][2] * distanceToCenter
        ),
        rotation: glm.quat.fromValues(0, 0, 0, 1),
        scale: glm.vec3.fromValues(sphereRadius, sphereRadius, sphereRadius)
    });
    Meshes[5].TransformMesh(
    {
        translation: glm.vec3.fromValues(
            middleCubeCenter[0] + directions[1][0] * distanceToCenter,
            middleCubeCenter[1] + directions[1][1] * distanceToCenter,
            middleCubeCenter[2] + directions[1][2] * distanceToCenter
        ),
        rotation: glm.quat.fromValues(0, 0, 0, 1),
        scale: glm.vec3.fromValues(sphereRadius, sphereRadius, sphereRadius)
    });
    Meshes[6].TransformMesh(
    {
        translation: glm.vec3.fromValues(
            middleCubeCenter[0] + directions[2][0] * distanceToCenter,
            middleCubeCenter[1] + directions[2][1] * distanceToCenter,
            middleCubeCenter[2] + directions[2][2] * distanceToCenter
        ),
        rotation: glm.quat.fromValues(0, 0, 0, 1),
        scale: glm.vec3.fromValues(sphereRadius, sphereRadius, sphereRadius)
    });

    return {
        meshes: Meshes,
        additionalInfo: {
            sphereMaterialIndices: [4, 5, 6],
            sphereTransforms: [Meshes[4].GetTransform(), Meshes[5].GetTransform(), Meshes[6].GetTransform()],
            sphereMaterials: [
                Meshes[4].GetMaterial(), // sphereOne
                Meshes[5].GetMaterial(), // sphereTwo
                Meshes[6].GetMaterial()  // sphereThree
            ]
        }
    };
}

//================================//
export async function createCornellBox3(meshMaterials: Material[]): Promise<SceneInformation>
{   
    const Meshes: Mesh[] = [];

    Meshes.push(new Mesh("white wall", createDefaultMaterial({ albedo: [0.73, 0.73, 0.73], name: "whiteWall", metalness: 1.0, roughness: 0.0 })));
    Meshes.push(new Mesh("Back Wall", createDefaultMaterial({ albedo: [0.73, 0.73, 0.73], name: "backWall", metalness: 0.3, roughness: 0.6 })));
    Meshes.push(new Mesh("red wall", createDefaultMaterial({ albedo: [0.65, 0.05, 0.05], name: "redWall" })));
    Meshes.push(new Mesh("green wall", createDefaultMaterial({ albedo: [0.12, 0.45, 0.15], name: "greenWall" })));
    Meshes.push(new Mesh("light", createDefaultMaterial({ albedo: [1.0, 1.0, 1.0], roughness: 0.0, name: "light" })));
    
    const dragonMaterial = meshMaterials.find(mat => mat.name === "dragon") || createDefaultMaterial({
        albedo: [0.12, 0.45, 0.15],
        name: "dragon",
        textureIndex: 0,
        useAlbedoTexture: true,
        useRoughnessTexture: true,
        useMetalnessTexture: true
    });
    
    //================================//
    function addQuad(
        Mesh: Mesh,
        v0: glm.vec3,
        v1: glm.vec3,
        v2: glm.vec3,
        v3: glm.vec3,
        flipNormal: boolean = false
    ): void 
    {
        let normal = computeNormal(v0, v1, v2);
        if (flipNormal)
            normal = glm.vec3.fromValues(-normal[0], -normal[1], -normal[2]);
        
        const i0 = Mesh.addVertex({pos: v0, normal: normal, uv: glm.vec2.fromValues(0, 0)});
        const i1 = Mesh.addVertex({pos: v1, normal: normal, uv: glm.vec2.fromValues(1, 0)});
        const i2 = Mesh.addVertex({pos: v2, normal: normal, uv: glm.vec2.fromValues(1, 1)});
        const i3 = Mesh.addVertex({pos: v3, normal: normal, uv: glm.vec2.fromValues(0, 1)});

        Mesh.addTriangle([i0, i1, i2]);
        Mesh.addTriangle([i0, i2, i3]);
    }

    // ============== FLOOR (white) ============== //
    addQuad(
        Meshes[0],
        glm.vec3.fromValues(552.8, 0.0, 0.0),
        glm.vec3.fromValues(0.0, 0.0, 0.0),
        glm.vec3.fromValues(0.0, 0.0, 559.2),
        glm.vec3.fromValues(549.6, 0.0, 559.2),
        false,
    );
    
    // ============== CEILING (white) ============== //
    addQuad(
        Meshes[0],
        glm.vec3.fromValues(556.0, 548.8, 0.0),
        glm.vec3.fromValues(556.0, 548.8, 559.2),
        glm.vec3.fromValues(0.0, 548.8, 559.2),
        glm.vec3.fromValues(0.0, 548.8, 0.0),
        false,
    );
    
    // ============== LIGHT (slightly below ceiling) ============== //
    const lightEpsilon = 1.0;
    const lightY = 548.8 - lightEpsilon;
    addQuad(
        Meshes[4],
        glm.vec3.fromValues(343.0, lightY, 227.0),
        glm.vec3.fromValues(343.0, lightY, 332.0),
        glm.vec3.fromValues(213.0, lightY, 332.0),
        glm.vec3.fromValues(213.0, lightY, 227.0),
        false,
    );
    
    // ============== BACK WALL (white) ============== //
    addQuad(
        Meshes[1],
        glm.vec3.fromValues(549.6, 0.0, 559.2),
        glm.vec3.fromValues(0.0, 0.0, 559.2),
        glm.vec3.fromValues(0.0, 548.8, 559.2),
        glm.vec3.fromValues(556.0, 548.8, 559.2),
        false,
    );
    
    // ============== RIGHT WALL (green) ============== //
    addQuad(
        Meshes[3],
        glm.vec3.fromValues(0.0, 0.0, 559.2),
        glm.vec3.fromValues(0.0, 0.0, 0.0),
        glm.vec3.fromValues(0.0, 548.8, 0.0),
        glm.vec3.fromValues(0.0, 548.8, 559.2),
        false
    );
    
    // ============== LEFT WALL (red) ============== //
    addQuad(
        Meshes[2],
        glm.vec3.fromValues(552.8, 0.0, 0.0),
        glm.vec3.fromValues(549.6, 0.0, 559.2),
        glm.vec3.fromValues(556.0, 548.8, 559.2),
        glm.vec3.fromValues(556.0, 548.8, 0.0),
        false,
    );

    // Middle
    let middleCubeCenter: [number, number, number] = [278.0, 224.4, 279.5];

    const loadedDragonMesh: Mesh = await loadMesh('/meshes/dragon/scene.gltf');
    loadedDragonMesh.Material = dragonMaterial;
    loadedDragonMesh.TransformMesh({
        translation: glm.vec3.fromValues(middleCubeCenter[0], middleCubeCenter[1], middleCubeCenter[2]),
        rotation: glm.quat.fromEuler(glm.quat.create(), 0, 0, 0),
        scale: glm.vec3.fromValues(2, 2, 2)
    });
    Meshes.push(loadedDragonMesh);

    for (const mesh of Meshes)
    {
        mesh.ComputeBVH();
    }

    return {
        meshes: Meshes,
        additionalInfo: {
            meshIndices: [5],
            meshTransforms: [Meshes[5].GetTransform()],
            meshMaterials: [
                Meshes[5].GetMaterial(), // dragon
            ]
        }
    };
}