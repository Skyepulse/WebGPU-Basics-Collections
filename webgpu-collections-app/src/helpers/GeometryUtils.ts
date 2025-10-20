//================================//
interface VertexInformation
{
    x: number,
    y: number,
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