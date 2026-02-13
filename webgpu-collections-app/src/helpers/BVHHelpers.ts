import type { Mesh, Triangle } from "./GeometryUtils";

const floatMax = Number.MAX_VALUE;
const floatMin = -Number.MAX_VALUE;

const MAX_DEPTH = 32;

//================================//
interface splitInfo
{
    axis: number; // 0 for x, 1 for y, 2 for z
    position: number;
    cost: number;
}

//================================//
interface bin
{
    count: number;
    minBounds: [number, number, number];
    maxBounds: [number, number, number];
}

//================================//
export class BVHTriangle
{
    public center: [number, number, number];
    public MinValues: [number, number, number];
    public MaxValues: [number, number, number];

    constructor(public v0: [number, number, number], public v1: [number, number, number], public v2: [number, number, number])
    {
        const x0 = v0[0], y0 = v0[1], z0 = v0[2];
        const x1 = v1[0], y1 = v1[1], z1 = v1[2];
        const x2 = v2[0], y2 = v2[1], z2 = v2[2];

        this.center = [(x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3, (z0 + z1 + z2) / 3];

        var minX = Math.min(x0, x1, x2);
        var minY = Math.min(y0, y1, y2);
        var minZ = Math.min(z0, z1, z2);

        this.MinValues = [minX, minY, minZ];

        var maxX = Math.max(x0, x1, x2);
        var maxY = Math.max(y0, y1, y2);
        var maxZ = Math.max(z0, z1, z2);

        this.MaxValues = [maxX, maxY, maxZ];
    }
}

//================================//
export class BVHNode
{
    public minBounds: [number, number, number];
    public maxBounds: [number, number, number];
    public triangleCount: number;
    public startIndex: number;

    constructor(minBounds: [number, number, number], maxBounds: [number, number, number], triangleCount: number, startIndex: number)
    {
        this.minBounds = minBounds;
        this.maxBounds = maxBounds;
        this.triangleCount = triangleCount;
        this.startIndex = startIndex;
    }
}

//================================//
export class BVH
{
    public Triangles: Triangle[] = [];
    public builtTriangles: BVHTriangle[] = [];
    public Nodes: BVHNode[] = [];

    //================================//
    public buildBVH(Mesh: Mesh)
    {
        this.Triangles = [];
        this.builtTriangles = [];
        this.Nodes = [];

        const numTriangles = Mesh.getNumTriangles();
        this.Triangles = Mesh.getTriangles();

        let minX = floatMax, minY = floatMax, minZ = floatMax;
        let maxX = floatMin, maxY = floatMin, maxZ = floatMin;

        for (let i = 0; i < numTriangles; i++)
        {
            const posA: [number, number, number] = [this.Triangles[i].vA.pos[0], this.Triangles[i].vA.pos[1], this.Triangles[i].vA.pos[2]];
            const posB: [number, number, number] = [this.Triangles[i].vB.pos[0], this.Triangles[i].vB.pos[1], this.Triangles[i].vB.pos[2]];
            const posC: [number, number, number] = [this.Triangles[i].vC.pos[0], this.Triangles[i].vC.pos[1], this.Triangles[i].vC.pos[2]];
            const t = new BVHTriangle(posA, posB, posC);
            this.builtTriangles.push(t);

            const min = t.MinValues;
            const max = t.MaxValues;

            if (min[0] < minX) minX = min[0];
            if (min[1] < minY) minY = min[1];
            if (min[2] < minZ) minZ = min[2];
            if (max[0] > maxX) maxX = max[0];
            if (max[1] > maxY) maxY = max[1];
            if (max[2] > maxZ) maxZ = max[2];
        }
        
        // Add root node
        this.Nodes.push(new BVHNode([minX, minY, minZ], [maxX, maxY, maxZ], -1, -1));

        this.buildTree(0, 0, numTriangles);
    }

    //================================//
    public buildTree(parent: number, start: number, numTriangles: number, depth: number = 0): void
    {
        const parentNode: BVHNode = this.Nodes[parent];

        const parentSize: [number, number, number] = [
            parentNode.maxBounds[0] - parentNode.minBounds[0],
            parentNode.maxBounds[1] - parentNode.minBounds[1],
            parentNode.maxBounds[2] - parentNode.minBounds[2]
        ];
        const costParent = this.computeCost(parentSize, numTriangles);

        // Choose the split
        const split: splitInfo = this.chooseSplit(parentNode, start, numTriangles);

        if (split.cost < costParent && depth < MAX_DEPTH)
        {
            let minValuesL: [number, number, number] = [floatMax, floatMax, floatMax];
            let maxValuesL: [number, number, number] = [floatMin, floatMin, floatMin];
            let minValuesR: [number, number, number] = [floatMax, floatMax, floatMax];
            let maxValuesR: [number, number, number] = [floatMin, floatMin, floatMin];

            let numLeft = 0;

            for (let triIndex = start; triIndex < start + numTriangles; triIndex++)
            {
                const tri: BVHTriangle = this.builtTriangles[triIndex];

                let c: number;
                switch (split.axis)
                {
                    case 0: c = tri.center[0]; break;
                    case 1: c = tri.center[1]; break;
                    case 2: c = tri.center[2]; break;
                    default: c = tri.center[0]; break;
                }

                // Decide on left or right
                if (c < split.position)
                {
                    if (tri.MinValues[0] < minValuesL[0]) minValuesL[0] = tri.MinValues[0];
                    if (tri.MinValues[1] < minValuesL[1]) minValuesL[1] = tri.MinValues[1];
                    if (tri.MinValues[2] < minValuesL[2]) minValuesL[2] = tri.MinValues[2];
                    if (tri.MaxValues[0] > maxValuesL[0]) maxValuesL[0] = tri.MaxValues[0];
                    if (tri.MaxValues[1] > maxValuesL[1]) maxValuesL[1] = tri.MaxValues[1];
                    if (tri.MaxValues[2] > maxValuesL[2]) maxValuesL[2] = tri.MaxValues[2];

                    const potentialSwap: BVHTriangle = this.builtTriangles[start + numLeft];
                    this.builtTriangles[start + numLeft] = tri;
                    this.builtTriangles[triIndex] = potentialSwap; // We need to swap to have contiguous triangles for left and right child
                    numLeft++;
                }
                else
                {
                    if (tri.MinValues[0] < minValuesR[0]) minValuesR[0] = tri.MinValues[0];
                    if (tri.MinValues[1] < minValuesR[1]) minValuesR[1] = tri.MinValues[1];
                    if (tri.MinValues[2] < minValuesR[2]) minValuesR[2] = tri.MinValues[2];
                    if (tri.MaxValues[0] > maxValuesR[0]) maxValuesR[0] = tri.MaxValues[0];
                    if (tri.MaxValues[1] > maxValuesR[1]) maxValuesR[1] = tri.MaxValues[1];
                    if (tri.MaxValues[2] > maxValuesR[2]) maxValuesR[2] = tri.MaxValues[2];
                }
            }

            if (numLeft === 0 || numLeft === numTriangles)
            {
                parentNode.startIndex = start;
                parentNode.triangleCount = numTriangles;
                this.Nodes[parent] = parentNode;

                return;
            }

            const startLeftSide = start;
            const startRightSide = start + numLeft;

            const nodeLeft: BVHNode = new BVHNode(minValuesL, maxValuesL, -1, startLeftSide);
            const nodeRight: BVHNode = new BVHNode(minValuesR, maxValuesR, -1, startRightSide);

            const nodeLeftIndex = this.Nodes.length;
            this.Nodes.push(nodeLeft);
            const nodeRightIndex = this.Nodes.length;
            this.Nodes.push(nodeRight);

            parentNode.startIndex = nodeLeftIndex;
            this.Nodes[parent] = parentNode;

            this.buildTree(nodeLeftIndex, startLeftSide, numLeft, depth + 1);
            this.buildTree(nodeRightIndex, startRightSide, numTriangles - numLeft, depth + 1);
        }
        else // The parent node is actually a leaf
        {
            parentNode.startIndex = start;
            parentNode.triangleCount = numTriangles;
            this.Nodes[parent] = parentNode;
        }
    }

    //================================//
    private computeCost(size: [number, number, number], numTriangles: number): number
    {
        if (numTriangles === 0) return 0;

        return (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]) * numTriangles;
    }

    //================================//
    private expandBin(bin: bin, triangle: BVHTriangle): void
    {
        bin.count++;
        for (let i = 0; i < 3; i++)
        {
            if (triangle.MinValues[i] < bin.minBounds[i]) bin.minBounds[i] = triangle.MinValues[i];
            if (triangle.MaxValues[i] > bin.maxBounds[i]) bin.maxBounds[i] = triangle.MaxValues[i];
        }
    }

    //================================//
    private mergeBins(binA: bin, binB: bin): bin
    {
        return {
            count: binA.count + binB.count,
            minBounds: [
                Math.min(binA.minBounds[0], binB.minBounds[0]),
                Math.min(binA.minBounds[1], binB.minBounds[1]),
                Math.min(binA.minBounds[2], binB.minBounds[2]),
            ],
            maxBounds: [
                Math.max(binA.maxBounds[0], binB.maxBounds[0]),
                Math.max(binA.maxBounds[1], binB.maxBounds[1]),
                Math.max(binA.maxBounds[2], binB.maxBounds[2]),
            ]
        };
    }

    //================================//
    // Implementation of binned SAH splitting technique
    private chooseSplit(node: BVHNode, start: number, numTriangles: number): splitInfo
    {
        const numBins = 12;
        let bestCost = Number.MAX_VALUE;
        let bestAxis = -1;
        let bestPosition = 0;

        for (let axis = 0; axis < 3; axis++)
        {
            const axisMin = node.minBounds[axis];
            const axisMax = node.maxBounds[axis];
            const range = axisMax - axisMin;
            if (range < 1e-5) continue;

            const bins: bin[] = [];
            for (let i = 0; i < numBins; i++)
            {
                bins.push({count: 0, minBounds: [floatMax, floatMax, floatMax], maxBounds: [floatMin, floatMin, floatMin] });
            }

            // 1. Bin assignment O(n) sweep
            for (let i = 0; i < numTriangles; i++)
            {
                const tri: BVHTriangle = this.builtTriangles[start + i];
                const t = (tri.center[axis] - axisMin) / range;

                // once we have t it is easy to determine which bin it is in
                let binIndex = Math.floor(t * numBins);
                if (binIndex >= numBins) binIndex = numBins - 1;
                if (binIndex < 0) binIndex = 0;

                this.expandBin(bins[binIndex], tri);
            }

            // 2. Sweep to find best split O(numBins)

            // L -> R
            const LeftBins: bin[] = [];
            LeftBins[0] = bins[0];
            for (let i = 1; i < numBins - 1; i++)
            {
                LeftBins[i] = this.mergeBins(LeftBins[i - 1], bins[i]);
            }

            // R -> L
            const RightBins: bin[] = [];
            RightBins[numBins - 2] = bins[numBins - 1];
            for (let i = numBins - 3; i >= 0; i--)
            {
                RightBins[i] = this.mergeBins(RightBins[i + 1], bins[i + 1]);
            }

            // Evaluate
            for (let i = 0; i < numBins - 1; i++)
            {
                const sizeLeft: [number, number, number] = [
                    LeftBins[i].maxBounds[0] - LeftBins[i].minBounds[0],
                    LeftBins[i].maxBounds[1] - LeftBins[i].minBounds[1],
                    LeftBins[i].maxBounds[2] - LeftBins[i].minBounds[2]
                ];
                const sizeRight: [number, number, number] = [
                    RightBins[i].maxBounds[0] - RightBins[i].minBounds[0],
                    RightBins[i].maxBounds[1] - RightBins[i].minBounds[1],
                    RightBins[i].maxBounds[2] - RightBins[i].minBounds[2]
                ];

                const totalCost = this.computeCost(sizeLeft, LeftBins[i].count) + this.computeCost(sizeRight, RightBins[i].count);
                if (totalCost < bestCost)
                {
                    bestCost = totalCost;
                    bestAxis = axis;
                    bestPosition = axisMin + range * (i + 1) / numBins;
                }
            }
        }

        return { axis: bestAxis, position: bestPosition, cost: bestCost };
    }

    //================================//
    public generateWireframeGeometry(maxDepth: number = Infinity): { vertexData: Float32Array, count: number }
    {
        const lines: number[] = [];

        const addEdge = (ax: number, ay: number, az: number, bx: number, by: number, bz: number) => 
        {
            lines.push(ax, ay, az, bx, by, bz);
        };

        const addBox = (min: [number, number, number], max: [number, number, number]) =>
        {
            addEdge(min[0], min[1], min[2], max[0], min[1], min[2]);
            addEdge(min[0], max[1], min[2], max[0], max[1], min[2]);
            addEdge(min[0], min[1], max[2], max[0], min[1], max[2]);
            addEdge(min[0], max[1], max[2], max[0], max[1], max[2]);

            addEdge(min[0], min[1], min[2], min[0], max[1], min[2]);
            addEdge(max[0], min[1], min[2], max[0], max[1], min[2]);
            addEdge(min[0], min[1], max[2], min[0], max[1], max[2]);
            addEdge(max[0], min[1], max[2], max[0], max[1], max[2]);

            addEdge(min[0], min[1], min[2], min[0], min[1], max[2]);
            addEdge(max[0], min[1], min[2], max[0], min[1], max[2]);
            addEdge(min[0], max[1], min[2], min[0], max[1], max[2]);
            addEdge(max[0], max[1], min[2], max[0], max[1], max[2]);
        };

        const stack: {index: number, depth: number}[] = [{index: 0, depth: 0}]; // At least root

        while (stack.length > 0)
        {
            const { index, depth } = stack.pop()!;
            const node = this.Nodes[index];

            if (depth >= maxDepth)
                continue;

            if (node.triangleCount === -1) // Not a leaf
            {
                stack.push({index: node.startIndex, depth: depth + 1});
                stack.push({index: node.startIndex + 1, depth: depth + 1});

                if (depth == maxDepth - 1)
                {
                    addBox(node.minBounds, node.maxBounds);
                }
            }
            else
            {
                addBox(node.minBounds, node.maxBounds); // Push all leaf nodes
            }
        }

        const vertexData = new Float32Array(lines);
        return { vertexData, count: vertexData.length / 3 };
    }
}