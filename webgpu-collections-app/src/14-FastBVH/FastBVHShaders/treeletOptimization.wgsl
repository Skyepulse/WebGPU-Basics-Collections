// This is shader 2 / 2 of the treelet optimization step described in 
// https://www.highperformancegraphics.org/wp-content/uploads/2013/Karras-BVH.pdf

// This shader is dispatched indirectly with the number of treelets found in the previous shader.
// Each treelet has a workgroup size of 32 assigned all to work on the DP problem.

// ALGORITHM TO FIND THE LEAVES OF THE TREELET (FORMATION) (4.1 of Karras 2013):
// 1. "To form a treelet, we start
// with the designated treelet root and its two children. We then grow
// the treelet iteratively by choosing the treelet leaf with the largest
// surface area and turning it into an internal node. This is accomplished
// by removing the chosen node from the set of leaves and
// using its children as new leaves instead. Repeating this, we need 5
// iterations in total to reach n = 7."

// ALGORITHM ONCE WE HAVE THE LEAVES OF THE TEELET:
// Size of treelet is fixed to 7 leaves. A, B, C, D, E, F, G.
// L is the ordered set of leaves.
// with s and p being the bitmasks representing the subsets of the 7 leaves:
// Total number of subsets is 2^7 - 1 = 127
// 
// 2. Calculate surface area for each subset
// for s = 1 to 127:
//     a[s] = surfaceArea(unionOfAABBS(L, s))

// 3. Initialize cost of individual leaves:
// for i = 0 to 6:
//     copt[2^i] = C(Li)
//
// 4. Optimize every subset of leaves
// for k = 2 to 7 do:   // Here k is the number of bits set to 1 therefore if k = 2 we visit AB, BC, AC, AD, ...
//   for s = 1 to 127 do:
//     cs <- inf
//     ps <- 0
//     for each p in partitioning(s) do:
//        c <- copt[p] + copt[s XOR p]
//        if c < cs then (cs, ps) <- (c, p)
//     end for
//     t <- TotalNumTriangles(L, s)
//     copt[s] <- min((1.2 * a[s] + cs), (1.0 * a[s] * t))
//     popt[s] <- ps
//   end for
// end for

// Details on the inner loop in partitioning(s):
// delta <- (s - 1) AND s
// p <- (-delta) AND s
// repeat
//   c <- copt[p] + copt[s XOR p]
//   if c < cs then (cs, ps) <- (c, p)
//   p <- (p - delta) AND s
// until p == 0

// The paper uses 32 threads which comes from CUDA I think,
// we could up this in WGPU but it would require more synchronization
// which I do not want to write now... sticking with 32 is ok I guess (?)

//================================//
override INTERNAL_NODE_COUNT: u32;
override LEAF_NODE_COUNT: u32;
const LEAF_BIT: u32 = 0x80000000u;

//================================//
const N_LEAVES: u32 = 7u;
const N_NODES: u32 = 13u;
const N_SUBSETS: u32 = 127u;

//================================//
// Hardcoded tables for scheduling
// on step 4 of subset management
const S2_COUNT: u32 = 21u;
const S2 = array<u32, 21>(
    3,5,6,9,10,12,17,18,20,24,
    33,34,36,40,48,65,66,68,72,80,96
);
const S3_COUNT: u32 = 35u;
const S3 = array<u32, 35>(
    7,11,13,14,19,21,22,25,26,28,
    35,37,38,41,42,44,49,50,52,56,
    67,69,70,73,74,76,81,82,84,88,
    97,98,100,104,112
);
const S4_COUNT: u32 = 35u;
const S4 = array<u32, 35>(
    15,23,27,29,30,39,43,45,46,51,
    53,54,57,58,60,71,75,77,78,83,
    85,86,89,90,92,99,101,102,105,106,
    108,113,114,116,120
);
const S5_COUNT: u32 = 21u;
const S5 = array<u32, 21>(
    31,47,55,59,61,62,79,87,91,93,
    94,103,107,109,110,115,117,118,121,122,124
);
const S6_COUNT: u32 = 7u;
const S6 = array<u32, 7>(
    63,95,111,119,123,125,126
);

//================================//
struct BVHNode 
{
    aabbMin: vec3f,
    parent:  u32,
    aabbMax: vec3f,
    triangleCount: u32,

    left:    u32,
    right:   u32,
    sahCost: f32,
    _pad1: u32,
};

struct LeafAABB 
{
    aabbMin: vec3f,
    _pad0: u32,
    aabbMax: vec3f,
    _pad1: u32,
};

//================================//
@group(0) @binding(0) var<storage, read_write> internalNodes: array<BVHNode>;
@group(0) @binding(1) var<storage, read_write> leafParents: array<u32>;
@group(0) @binding(2) var<storage, read> leafAABBs: array<LeafAABB>;
@group(0) @binding(3) var<storage, read>  treeletRoots:   array<u32>;

// I could separate a and copt into two different arrays,
// but normally a subset that has not yet been processed stores the area
// and copt after.
// This is what happens in:
// a_copt[s - 1u] = min(1.2 * a_copt[s - 1u] + bc, 1.0 * a_copt[s - 1u] * f32(totalTriangles));
// We change the area with the subset cost, the area does not need to be accessed again.
var<workgroup> a_copt:      array<f32, 127>; 
var<workgroup> popt:        array<u32, 127>;
var<workgroup> triTotal:    array<u32, 127>; 

// Just simple arrays to store the results of local minima per thread
// to reduce them to global minima
// when number of partitions is bigger than 32 (see steps 5 and 6)
var<workgroup> reduceC:     array<f32, 32>;
var<workgroup> reduceP:     array<u32, 32>;

// The 6 first slots are internal nodes, 7 after are leaves IN ORDER
var<workgroup> nodeIndex:       array<u32, 13>; // Whoch BVH node
var<workgroup> nodeAABBMin:     array<vec3f, 13>;
var<workgroup> nodeAABBMax:     array<vec3f, 13>;
var<workgroup> nodeCost:        array<f32, 13>; // C(n)
var<workgroup> nodeTriCount:    array<u32, 13>;
var<workgroup> nodeLeft:        array<u32, 13>;
var<workgroup> nodeRight:       array<u32, 13>;
var<workgroup> nodeSA:          array<f32, 13>;

// I had an error on not all threads hitting all workGroupBarriers.
// So this shared variable is set as a flag.
var<workgroup> shouldWriteBack: u32;

//================================//
@compute
@workgroup_size(32)
fn cs(
    @builtin(workgroup_id) w_id: vec3u, 
    @builtin(local_invocation_id) l_id: vec3u)
{
    let isFirstThread = l_id.x == 0u;
    let treeletRootIndex = treeletRoots[w_id.x];

    // [0] Load root and two first children
    if (isFirstThread)
    {
        nodeIndex[0]    = treeletRootIndex;
        nodeAABBMin[0]  = internalNodes[treeletRootIndex].aabbMin;
        nodeAABBMax[0]  = internalNodes[treeletRootIndex].aabbMax;
        nodeCost[0]     = internalNodes[treeletRootIndex].sahCost;
        nodeTriCount[0] = internalNodes[treeletRootIndex].triangleCount;
        nodeLeft[0]     = internalNodes[treeletRootIndex].left;
        nodeRight[0]    = internalNodes[treeletRootIndex].right;
        nodeSA[0]       = surfaceArea(nodeAABBMin[0], nodeAABBMax[0]);
    }
    workgroupBarrier();

    if (l_id.x < 2u)
    {
        var childIndex: u32;
        if (isFirstThread){ childIndex = nodeLeft[0]; }
        else { childIndex = nodeRight[0]; }
        loadNodeFromBVH(l_id.x + 1u, childIndex);
    }
    workgroupBarrier();

    // [1] Expand to find 5 more leaves, 5 sweeps
    // Number of threads: 1 purely sequential
    var currentLeafCount: u32 = 2u;
    var currentInternalNodesCount: u32 = 1u;
    for (var step = 0u; step < 5u; step++)
    {
        var bestSlot = 0u;
        var bestSA: f32 = -1.0;

        // Find the max
        if (isFirstThread)
        {
            for (var i = currentInternalNodesCount; i < currentInternalNodesCount + currentLeafCount; i++)
            {
                let idx = nodeIndex[i];
                if ((idx & LEAF_BIT) == 0u && nodeSA[i] > bestSA) // We can only expands BVH internal nodes
                {
                    bestSA = nodeSA[i];
                    bestSlot = i;
                }
            }
        }
        workgroupBarrier();

        if (isFirstThread)
        {
            let expandIndex = bestSlot;
            let swapIndex = currentInternalNodesCount;
            if (expandIndex != swapIndex)
            {
                swapNodes(expandIndex, swapIndex);
            }

            let parentBVHIndex = nodeIndex[swapIndex];
            let leftChildBVHIndex = internalNodes[parentBVHIndex].left;
            let rightChildBVHIndex = internalNodes[parentBVHIndex].right;

            let newSlot1 = currentInternalNodesCount + currentLeafCount;
            let newSlot2 = newSlot1 + 1u;
            loadNodeFromBVH(newSlot1, leftChildBVHIndex);
            loadNodeFromBVH(newSlot2, rightChildBVHIndex);
        }
        workgroupBarrier();

        currentInternalNodesCount+= 1u;
        currentLeafCount+= 1u;
    }

    // [2] Init surface areas for all subsets
    // Number of threads: all 32.
    for (var subset = l_id.x + 1u; subset <= N_SUBSETS; subset += 32u)
    {
        var mn = vec3f(1e30);
        var mx = vec3f(-1e30);
        var totalTriangles: u32 = 0u;

        for (var i = 0u; i < N_LEAVES; i++)
        {
            // Know if a leaf i is in the subset: subset & (1 << i) != 0 
            if ((subset & (1u << i)) != 0u)
            {
                let slot = 6u + i;
                mn = min(mn, nodeAABBMin[slot]);
                mx = max(mx, nodeAABBMax[slot]);
                totalTriangles += nodeTriCount[slot];
            }
        }

        a_copt[subset - 1u] = surfaceArea(mn, mx);
        triTotal[subset - 1u] = totalTriangles;
    }
    workgroupBarrier();

    // [3] Initialize all 7 single leaf subsets costs
    // Number of threads: 7 threads, one for each leaf subset.
    if (l_id.x < N_LEAVES)
    {
        let leafSlot = 6u + l_id.x;
        let mask = 1u << l_id.x;
        a_copt[mask - 1u] = nodeCost[leafSlot]; // We can overwrite this since
        // a_copt entries for single-leaf masks hold copt values, while multi-leaf entries still hold surface areas
    }
    workgroupBarrier();

    // [4] DP for k = 2 to 5
    // Number of threads: one thread per subset, workgroupBarrier between each k.
    // We need the previous k value to end before jumping on next value of k
    // Since we need (this is the DP part) all sub jobs to be done before we move k+=1
    if (l_id.x < S2_COUNT) { processSubset(S2[l_id.x]); }
    workgroupBarrier();

    if (l_id.x < 32u) { processSubset(S3[l_id.x]); } // Two rounds since we have 35 subsets, and only 32 threads.
    workgroupBarrier();
    if (l_id.x < 3u) { processSubset(S3[32u + l_id.x]); }
    workgroupBarrier();

    if (l_id.x < 32u) { processSubset(S4[l_id.x]); }
    workgroupBarrier();
    if (l_id.x < 3u) { processSubset(S4[32u + l_id.x]); }
    workgroupBarrier();

    if (l_id.x < S5_COUNT){ processSubset(S5[l_id.x]); }
    workgroupBarrier();

    // [5] DP for k = 6
    // 7 subsets here,  31 partitions each, so 4 threads per subset (4 * 7 = 28 threads used)

    // local best
    let subsetIndex = l_id.x / 4u;
    let chunk = l_id.x % 4u;

    if (subsetIndex < S6_COUNT)
    {
        let s = S6[subsetIndex];

        var cs: f32 = 1e30;
        var ps: u32 = 0u;
        var count: u32 = 0u;

        let delta = (s - 1u) & s;
        var p = (~delta + 1u) & s;

        while (p != 0u && p != s)
        {
            if (count % 4u == chunk)
            {
                let c = a_copt[p - 1u] + a_copt[(s^p) - 1u];
                if (c < cs)
                {
                    cs = c;
                    ps = p;
                }
            }
            p = (p - delta) & s;
            count++;
        }

        reduceC[l_id.x] = cs;
        reduceP[l_id.x] = ps;
    }
    workgroupBarrier();

    // Now reduce over local bests to find global bests for each subset
    if (chunk == 0u && subsetIndex < S6_COUNT)
    {
        let base = subsetIndex * 4u;
        var bc: f32 = reduceC[base];
        var bp: u32 = reduceP[base];

        for (var i = 1u; i < 4u; i++)
        {
            if (reduceC[base + i] < bc)
            {
                bc = reduceC[base + i];
                bp = reduceP[base + i];
            }
        }

        let s = S6[subsetIndex];
        let totalTriangles = triTotal[s - 1u];
        a_copt[s - 1u] = min(1.2 * a_copt[s - 1u] + bc, 1.0 * a_copt[s - 1u] * f32(totalTriangles));
        popt[s - 1u] = bp;
    }
    workgroupBarrier();

    // [6] DP for k = 7, only one subset and 63 partitions. We use all threads
    {
        let s = 127u;
        var threadBestC: f32 = 1e30;
        var threadBestP: u32 = 0u;
        var threadCount: u32 = 0u; // Each thread can at most do 2 partitions, some will do only 1

        let delta = (s - 1u) & s;
        var p = (~delta + 1u) & s;

        while (p != 0u && p != s)
        {
            if (threadCount % 32u == l_id.x)
            {
                let c = a_copt[p - 1u] + a_copt[(s^p) - 1u];
                if (c < threadBestC)
                {
                    threadBestC = c;
                    threadBestP = p;
                }
            }
            p = (p - delta) & s;
            threadCount++;
        }

        reduceC[l_id.x] = threadBestC;
        reduceP[l_id.x] = threadBestP;
    }
    workgroupBarrier();

    // Now reduce 16 -> 8 -> 4 -> 2 -> 1
    if (l_id.x < 16u) { if (reduceC[l_id.x + 16u] < reduceC[l_id.x]) { reduceC[l_id.x] = reduceC[l_id.x + 16u]; reduceP[l_id.x] = reduceP[l_id.x + 16u]; } }
    workgroupBarrier();
    if (l_id.x < 8u) { if (reduceC[l_id.x + 8u] < reduceC[l_id.x]) { reduceC[l_id.x] = reduceC[l_id.x + 8u]; reduceP[l_id.x] = reduceP[l_id.x + 8u]; } }
    workgroupBarrier();
    if (l_id.x < 4u) { if (reduceC[l_id.x + 4u] < reduceC[l_id.x]) { reduceC[l_id.x] = reduceC[l_id.x + 4u]; reduceP[l_id.x] = reduceP[l_id.x + 4u]; } }
    workgroupBarrier();
    if (l_id.x < 2u) { if (reduceC[l_id.x + 2u] < reduceC[l_id.x]) { reduceC[l_id.x] = reduceC[l_id.x + 2u]; reduceP[l_id.x] = reduceP[l_id.x + 2u]; } }
    workgroupBarrier();
    if (isFirstThread)
    {
        if (reduceC[1u] < reduceC[0u])
        {
            reduceC[0u] = reduceC[1u];
            reduceP[0u] = reduceP[1u];
        }

        let s = 127u;
        let totalTriangles = triTotal[s - 1u];
        a_copt[s - 1u] = min(1.2 * a_copt[s - 1u] + reduceC[0u], 1.0 * a_copt[s - 1u] * f32(totalTriangles));
        popt[s - 1u] = reduceP[0u];
    }
    workgroupBarrier();

    // [7] Now that we have the optimal partitioning
    // We just need to reconstruct it

    if (isFirstThread) // Replaced early return with flag.
    {
        if (a_copt[126u] < nodeCost[0])
        {
            shouldWriteBack = 1u;
        }
        else // No improvement to the treelet, let it as is.
        {
            shouldWriteBack = 0u;
        }
    }
    workgroupBarrier();

    let shouldWeWriteBack = shouldWriteBack == 1u;

    if (isFirstThread && shouldWeWriteBack)
    {
        // sequential reconstruction
        var originalInternalNodes: array<u32, 6>;
        for (var i = 0u; i < 6u; i++)
        {
            originalInternalNodes[i] = nodeIndex[i];
        }

        var stack: array<u32, 6>;
        var stackSlot: array<u32, 6>;
        var pointer: i32 = 0;
        var nextSlot: u32 = 0u;

        stack[0] = 127u;
        stackSlot[0] = 0u;
        pointer = 1;

        while(pointer > 0)
        {
            pointer -= 1;
            let subset = stack[pointer];
            let slot = stackSlot[pointer];
            let leftSubset = popt[subset - 1u];
            let rightSubset = subset ^ leftSubset;

            // Left child
            var leftBVHIndex: u32;
            if (countOneBits(leftSubset) == 1u) // i.e. leaf
            {
                let leafBit = u32(firstLeadingBit(leftSubset));
                leftBVHIndex = nodeIndex[6u + leafBit];
            }
            else
            {
                nextSlot += 1u;
                stack[pointer] = leftSubset;
                stackSlot[pointer] = nextSlot;
                pointer += 1;
                leftBVHIndex = originalInternalNodes[nextSlot];
            }

            // Right child
            var rightBVHIndex: u32;
            if (countOneBits(rightSubset) == 1u)
            {
                let leafBit = u32(firstLeadingBit(rightSubset));
                rightBVHIndex = nodeIndex[6u + leafBit];
            }
            else
            {
                nextSlot += 1u;
                stack[pointer] = rightSubset;
                stackSlot[pointer] = nextSlot;
                pointer += 1;
                rightBVHIndex = originalInternalNodes[nextSlot];
            }

            nodeIndex[slot] = originalInternalNodes[slot];
            nodeLeft[slot] = leftBVHIndex;
            nodeRight[slot] = rightBVHIndex;

            var mn = vec3f(1e30);
            var mx = vec3f(-1e30);
            for (var i = 0u; i < 7u; i++)
            {
                if ((subset & (1u << i)) != 0u)
                {
                    mn = min(mn, nodeAABBMin[6u + i]);
                    mx = max(mx, nodeAABBMax[6u + i]);
                }
            }
            nodeAABBMin[slot] = mn;
            nodeAABBMax[slot] = mx;
            nodeTriCount[slot] = triTotal[subset - 1u];
            nodeCost[slot] = a_copt[subset - 1u];
        }
    }
    workgroupBarrier();

    // [8] Write back to global memory
    // 6 internal nodes == 6 threads.
    if (l_id.x < 6u && shouldWeWriteBack)
    {
        let slot = l_id.x;
        let bvhIndex = nodeIndex[slot];
        internalNodes[bvhIndex].left = nodeLeft[slot];
        internalNodes[bvhIndex].right = nodeRight[slot];
        internalNodes[bvhIndex].aabbMin = nodeAABBMin[slot];
        internalNodes[bvhIndex].aabbMax = nodeAABBMax[slot];
        internalNodes[bvhIndex].triangleCount = nodeTriCount[slot];
        internalNodes[bvhIndex].sahCost = nodeCost[slot];
    }
    workgroupBarrier();

    if (l_id.x < 6u && shouldWeWriteBack)
    {
        let slot = l_id.x;
        let parentBVHIndex = nodeIndex[slot];

        let leftChild = nodeLeft[slot];
        if ((leftChild & LEAF_BIT) != 0u) // if is a real BVH leaf, treat differently
        {
            leafParents[leftChild & 0x7FFFFFFFu] = parentBVHIndex;
        }
        else
        {
            internalNodes[leftChild].parent = parentBVHIndex;
        }

        let rightChild = nodeRight[slot];
        if ((rightChild & LEAF_BIT) != 0u)
        {
            leafParents[rightChild & 0x7FFFFFFFu] = parentBVHIndex;
        }
        else
        {
            internalNodes[rightChild].parent = parentBVHIndex;
        }
    }
}

//================================//
fn surfaceArea(aabbMin: vec3f, aabbMax: vec3f) -> f32
{
    let d = aabbMax - aabbMin;
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

//================================//
fn loadNodeFromBVH(slot: u32, bvhIndex: u32)
{
    nodeIndex[slot] = bvhIndex;
    if ((bvhIndex & LEAF_BIT) != 0u)
    {
        let li = bvhIndex & 0x7FFFFFFFu;
        nodeAABBMin[slot]   = leafAABBs[li].aabbMin;
        nodeAABBMax[slot]   = leafAABBs[li].aabbMax;
        nodeCost[slot]      = surfaceArea(leafAABBs[li].aabbMin, leafAABBs[li].aabbMax);
        nodeTriCount[slot]  = 1u;
        nodeSA[slot]        = nodeCost[slot];
        nodeLeft[slot]      = 0u;
        nodeRight[slot]     = 0u;
    }
    else
    {
        nodeAABBMin[slot]   = internalNodes[bvhIndex].aabbMin;
        nodeAABBMax[slot]   = internalNodes[bvhIndex].aabbMax;
        nodeCost[slot]      = internalNodes[bvhIndex].sahCost;
        nodeTriCount[slot]  = internalNodes[bvhIndex].triangleCount;
        nodeSA[slot]        = surfaceArea(nodeAABBMin[slot], nodeAABBMax[slot]);
        nodeLeft[slot]      = internalNodes[bvhIndex].left;
        nodeRight[slot]     = internalNodes[bvhIndex].right;
    }
}

//================================//
fn swapNodes(expandIdx: u32, swapIdx: u32)
{
    let tempIndex = nodeIndex[expandIdx];
    nodeIndex[expandIdx] = nodeIndex[swapIdx];
    nodeIndex[swapIdx] = tempIndex;

    let tempAABBMin = nodeAABBMin[expandIdx];
    nodeAABBMin[expandIdx] = nodeAABBMin[swapIdx];
    nodeAABBMin[swapIdx] = tempAABBMin;

    let tempAABBMax = nodeAABBMax[expandIdx];
    nodeAABBMax[expandIdx] = nodeAABBMax[swapIdx];
    nodeAABBMax[swapIdx] = tempAABBMax;

    let tempCost = nodeCost[expandIdx];
    nodeCost[expandIdx] = nodeCost[swapIdx];
    nodeCost[swapIdx] = tempCost;

    let tempTriCount = nodeTriCount[expandIdx];
    nodeTriCount[expandIdx] = nodeTriCount[swapIdx];
    nodeTriCount[swapIdx] = tempTriCount;

    let tempSA = nodeSA[expandIdx];
    nodeSA[expandIdx] = nodeSA[swapIdx];
    nodeSA[swapIdx] = tempSA;

    let tempLeft = nodeLeft[expandIdx];
    nodeLeft[expandIdx] = nodeLeft[swapIdx];
    nodeLeft[swapIdx] = tempLeft;

    let tempRight = nodeRight[expandIdx];
    nodeRight[expandIdx] = nodeRight[swapIdx];
    nodeRight[swapIdx] = tempRight;
}

//================================//
fn processSubset(s: u32)
{
    var cs: f32 = 1e30;
    var ps: u32 = 0u;

    let delta = (s - 1u) & s;
    var p = (~delta + 1u) & s;

    while (p != 0u && p != s)
    {
        let c = a_copt[p - 1u] + a_copt[(s^p) - 1u];
        if (c < cs)
        {
            cs = c;
            ps = p;
        }
        p = (p - delta) & s;
    }

    let tri = triTotal[s - 1u];
    a_copt[s - 1u] = min(1.2 * a_copt[s - 1u] + cs, 1.0 * a_copt[s - 1u] * f32(tri));
    popt[s - 1u] = ps;
}