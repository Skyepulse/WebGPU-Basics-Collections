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
export interface TopologyInformation
{
    vertexData: Float32Array,
    indexData: Uint16Array,
    numVertices: number

    normalData?: Float32Array
    colorData?: Float32Array
    uvData?: Float32Array
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
    const scale = 1.0;
    
    // Colors
    const white: [number, number, number] = [0.73, 0.73, 0.73];
    const red: [number, number, number] = [0.65, 0.05, 0.05];
    const green: [number, number, number] = [0.12, 0.45, 0.15];
    const light: [number, number, number] = [1.0, 1.0, 1.0];
    
    // Temporary arrays to collect data
    const vertices: number[] = [];
    const normals: number[] = [];
    const colors: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];
    
    let vertexCount = 0;
    
    //================================//
    function vec3Subtract(a: [number, number, number], b: [number, number, number]): [number, number, number] 
    {
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    }
    
    //================================//
    function vec3Cross(a: [number, number, number], b: [number, number, number]): [number, number, number] 
    {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }
    
    //================================//
    function vec3Normalize(v: [number, number, number]): [number, number, number] 
    {
        const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len > 0.00001) {
            return [v[0] / len, v[1] / len, v[2] / len];
        }
        return [0, 0, 0];
    }
    
    //================================//
    function computeNormal(
        v0: [number, number, number],
        v1: [number, number, number],
        v2: [number, number, number]
    ): [number, number, number] 
    {
        const edge1 = vec3Subtract(v1, v0);
        const edge2 = vec3Subtract(v2, v0);
        return vec3Normalize(vec3Cross(edge1, edge2));
    }
    
    //================================//
    function addVertex(
        position: [number, number, number],
        normal: [number, number, number],
        color: [number, number, number],
        uv: [number, number]
    ): number 
    {
        vertices.push(
            position[0] * scale - 0.5,
            position[1] * scale,
            position[2] * scale - 0.5
        );
        normals.push(normal[0], normal[1], normal[2]);
        colors.push(color[0], color[1], color[2]);
        uvs.push(uv[0], uv[1]);
        return vertexCount++;
    }
    
    //================================//
    function addQuad(
        v0: [number, number, number],
        v1: [number, number, number],
        v2: [number, number, number],
        v3: [number, number, number],
        color: [number, number, number],
        flipNormal: boolean = false
    ): void 
    {
        let normal = computeNormal(v0, v1, v2);
        if (flipNormal) {
            normal = [-normal[0], -normal[1], -normal[2]];
        }
        
        const i0 = addVertex(v0, normal, color, [0, 0]);
        const i1 = addVertex(v1, normal, color, [1, 0]);
        const i2 = addVertex(v2, normal, color, [1, 1]);
        const i3 = addVertex(v3, normal, color, [0, 1]);
        
        indices.push(i0, i1, i2);
        indices.push(i0, i2, i3);
    }
    
    // ============== FLOOR (white) ============== //
    addQuad(
        [552.8, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 559.2],
        [549.6, 0.0, 559.2],
        white
    );
    
    // ============== CEILING (white) ============== //
    addQuad(
        [556.0, 548.8, 0.0],
        [556.0, 548.8, 559.2],
        [0.0, 548.8, 559.2],
        [0.0, 548.8, 0.0],
        white
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
        white
    );
    
    // Right face
    addQuad(
        [472.0, 0.0, 406.0],
        [472.0, 330.0, 406.0],
        [314.0, 330.0, 456.0],
        [314.0, 0.0, 456.0],
        white
    );
    
    // Back face
    addQuad(
        [314.0, 0.0, 456.0],
        [314.0, 330.0, 456.0],
        [265.0, 330.0, 296.0],
        [265.0, 0.0, 296.0],
        white
    );
    
    // Left face
    addQuad(
        [265.0, 0.0, 296.0],
        [265.0, 330.0, 296.0],
        [423.0, 330.0, 247.0],
        [423.0, 0.0, 247.0],
        white
    );
    
    return {
        vertexData: new Float32Array(vertices),
        indexData: new Uint16Array(indices),
        numVertices: indices.length,
        normalData: new Float32Array(normals),
        colorData: new Float32Array(colors),
        uvData: new Float32Array(uvs)
    };
}