// This is shader of the treelet optimization step described in 
// https://www.highperformancegraphics.org/wp-content/uploads/2013/Karras-BVH.pdf

// This shader is a bottom-up traversal and finds possible treelet roots,
// unionizes threads at subgroup levels to work on its optimization, that way we are sure
// we do not concurrently change pointers for two different treelets at the same time.

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

//================================//
// Using subgroups is important since it gives us access to
// ballot and shuffle operations, which are used in the Karras 2013 paper
// on the CUDA implementation. If subgroups is not available on the hardware,
// this piece of shader will not fire (if I did my job correctly jeje).
enable subgroups;

//================================//
override LEAF_NODE_COUNT: u32;
const LEAF_BIT: u32 = 0x80000000u;
const INVALID_NODE: u32 = 0xFFFFFFFFu;
const COLLAPSE_FLAG: u32 = 1u; // this flag will mark whether we collapse this node in the next collapse pass

//================================//
const N_LEAVES: u32 = 7u;
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
    flags: u32,
};

struct LeafAABB 
{
    aabbMin: vec3f,
    _pad0: u32,
    aabbMax: vec3f,
    _pad1: u32,
};

struct GammaUniform
{
    gamma: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

//================================//
@group(0) @binding(0) var<storage, read_write> internalNodes: array<BVHNode>;
@group(0) @binding(1) var<storage, read_write> leafParents: array<u32>;
@group(0) @binding(2) var<storage, read> leafAABBs: array<LeafAABB>;
@group(0) @binding(3) var<storage, read_write>  atomicCounters: array<atomic<u32>>;

@group(1) @binding(0) var<uniform> gammaUniform: GammaUniform;

// I could separate a and copt into two different arrays,
// but normally a subset that has not yet been processed stores the area
// and copt after, so no conflict there.
// This is what happens in:
// a_copt[s - 1u] = min(1.2 * a_copt[s - 1u] + bc, 1.0 * a_copt[s - 1u] * f32(totalTriangles));
// We change the area with the subset cost, the area does not need to be accessed again.
var<workgroup> a_copt:          array<f32, 127>; 
var<workgroup> popt:            array<u32, 127>;
var<workgroup> triTotal:        array<u32, 127>; 
var<workgroup> subsetCollapse:  array<u32, 127>; // 0 if split won, 1 if collapse won

// Just simple arrays to store the results of local minima per thread
// to reduce them to global minima
// when number of partitions is bigger than 32 (see steps 5 and 6)
var<workgroup> reduceC:         array<f32, 32>;
var<workgroup> reduceP:         array<u32, 32>;

// The 6 first slots are internal nodes, 7 after are leaves IN ORDER
var<workgroup> nodeIndex:       array<u32, 13>; // Whoch BVH node
var<workgroup> nodeAABBMin:     array<vec3f, 13>;
var<workgroup> nodeAABBMax:     array<vec3f, 13>;
var<workgroup> nodeCost:        array<f32, 13>; // C(n)
var<workgroup> nodeTriCount:    array<u32, 13>;
var<workgroup> nodeLeft:        array<u32, 13>;
var<workgroup> nodeRight:       array<u32, 13>;
var<workgroup> nodeSA:          array<f32, 13>;
var<workgroup> nodeCollapse:    array<u32, 13>;

// Flag to decide if we write the optimized treelet back or not
var<workgroup> shouldWriteBack: u32;

//================================//
@compute
@workgroup_size(32)
fn cs(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) l_id: vec3u)
{
    let lane = l_id.x; // ID in the subgroup

    if (gid.x >= LEAF_NODE_COUNT) 
    {
        return;
    }

    var currentNode = leafParents[gid.x];

    loop {

        let activeLane = currentNode != INVALID_NODE;

        // MEMO FOR ME -> how subgroupBallot works:
        // It does a warp/wave-wide vote (apparently warp was in CUDA?)
        // across the current subgroup and returns a vec4<u32> bitmask for the active lanes where
        // the predicate is true.
        // Bit i corresponds to subgroup_invocation_id == i;
        // x -> 0-31, y -> 32-63, z -> 64-95, w -> 96-127
        var activeMask = subgroupBallot(activeLane); 

        if (ballotIsZero(activeMask))
        {
            break;
        }

        var candidateRoot = INVALID_NODE;
        var nextNode = INVALID_NODE;

        if (activeLane)
        {
            let isFirstToReach = atomicAdd(&atomicCounters[currentNode], 1u);

            if (isFirstToReach == 0u)
            {
                currentNode = INVALID_NODE; // First thread stops, same logic as the AABB pass.
                // Just instead of returning, in a subgroup apparently it is better
                // practice to decide the valid threads with a ballot.
            }
            else
            {
                if (internalNodes[currentNode].triangleCount >= gammaUniform.gamma)
                {
                    candidateRoot = currentNode;
                }
                nextNode = internalNodes[currentNode].parent;
            }
        }

        // For example a ballot allows us here to gather all the valid
        // candidate roots found in this iteration step!! Super cool.
        var rootMask = subgroupBallot(candidateRoot != INVALID_NODE); 

        loop {
            if (ballotIsZero(rootMask))
            {
                break;
            }

            let ownerLane = firstLaneInBallot(rootMask);

            // MEMO FOR ME -> how subgroupShuffle works:
            // reads the first arg from the lane in the current subgroup
            // whose subgroup invocation ID is specified by the second argument, and returns the 
            // value to the caller.

            // it is a lane-to-lane register exchange without workgroup shared memory.
            // only inside same subgroup though.
            let root = subgroupShuffle(candidateRoot, ownerLane);

            OptimizeOneTreelet(root, lane);

            rootMask = clearLaneInBallot(rootMask, ownerLane);
        }

        if (activeLane && currentNode != INVALID_NODE)
        {
            currentNode = nextNode;
        }
    }
}

//================================//
fn OptimizeOneTreelet(root: u32, lane: u32) 
{
    let isFirstLane = lane == 0u;

    // [0] Load Root (only one thread needed)
    if (isFirstLane) 
    {
        loadNodeFromBVH(0u, root);
    }
    workgroupBarrier();

    // [1] Root children loading (two threads needed)
    if (lane < 2u) 
    {
        let child = select(nodeRight[0], nodeLeft[0], isFirstLane);
        loadNodeFromBVH(1u + lane, child);
    }
    workgroupBarrier();

    var currentLeafCount: u32 = 2u;
    var currentInternalCount: u32 = 1u;

    // [2] Expand 5 times to get 7 leaves.
    // We can use parallel reduction as the paper suggests
    // to find which leaf to expand with the SAH heuristic cost.
    for (var step = 0u; step < 5u; step++) 
    {
        var mySA = -1.0;
        var mySlot = 0xFFFFFFFFu;

        let begin = currentInternalCount;
        let end   = currentInternalCount + currentLeafCount;

        if (lane >= begin && lane < end) 
        {
            let idx = nodeIndex[lane];
            if ((idx & LEAF_BIT) == 0u) 
            {
                mySA = nodeSA[lane];
                mySlot = lane;
            }
        }

        reduceC[lane] = mySA;
        reduceP[lane] = mySlot;
        workgroupBarrier();

        // max-reduction over 32 lanes
        // Reminder: parallel reduction 16 -> 8 -> 4 -> 2 -> 1
        if (lane < 16u) { if (reduceC[lane + 16u] > reduceC[lane]) { reduceC[lane] = reduceC[lane + 16u]; reduceP[lane] = reduceP[lane + 16u]; } }
        workgroupBarrier();
        if (lane < 8u) { if (reduceC[lane + 8u] > reduceC[lane]) { reduceC[lane] = reduceC[lane + 8u]; reduceP[lane] = reduceP[lane + 8u]; } }
        workgroupBarrier();
        if (lane < 4u) { if (reduceC[lane + 4u] > reduceC[lane]) { reduceC[lane] = reduceC[lane + 4u]; reduceP[lane] = reduceP[lane + 4u]; } }
        workgroupBarrier();
        if (lane < 2u) { if (reduceC[lane + 2u] > reduceC[lane]) { reduceC[lane] = reduceC[lane + 2u]; reduceP[lane] = reduceP[lane + 2u]; } }
        workgroupBarrier();
        if (isFirstLane) 
        {
            if (reduceC[1u] > reduceC[0u]) 
            {
                reduceC[0u] = reduceC[1u];
                reduceP[0u] = reduceP[1u];
            }

            let expandSlot = reduceP[0u];
            let swapSlot   = currentInternalCount;

            if (expandSlot != swapSlot) 
            {
                swapNodes(expandSlot, swapSlot);
            }

            let parentBVHIndex = nodeIndex[swapSlot];
            let leftChild  = internalNodes[parentBVHIndex].left;
            let rightChild = internalNodes[parentBVHIndex].right;

            let newSlot1 = currentInternalCount + currentLeafCount;
            let newSlot2 = newSlot1 + 1u;

            loadNodeFromBVH(newSlot1, leftChild);
            loadNodeFromBVH(newSlot2, rightChild);
        }
        workgroupBarrier();

        currentInternalCount += 1u;
        currentLeafCount += 1u;
    }

    // [3] Initialize the SAH cost of all 127 subsets
    // We can do this in parallel using all threads.
    // We have 32 in a subgroup meaning some will do at most 4 subsets, some 3
    for (var subset = lane + 1u; subset <= N_SUBSETS; subset += 32u) 
    {
        var mn = vec3f(1e30);
        var mx = vec3f(-1e30);
        var tri: u32 = 0u;

        for (var i = 0u; i < N_LEAVES; i++) 
        {
            if ((subset & (1u << i)) != 0u) 
            {
                let slot = 6u + i;
                mn = min(mn, nodeAABBMin[slot]);
                mx = max(mx, nodeAABBMax[slot]);
                tri += nodeTriCount[slot];
            }
        }

        a_copt[subset - 1u] = surfaceArea(mn, mx);
        triTotal[subset - 1u] = tri;
    }
    workgroupBarrier();

    // [4] Initialze the cost of individual leaves
    // 7 leaves -> 7 threads
    if (lane < 7u) 
    {
        let mask = 1u << lane;
        let slot = 6u + lane; // We know we have 6 internal nodes before the leaves...
        a_copt[mask - 1u] = nodeCost[slot];
        popt[mask - 1u] = 0u;
        subsetCollapse[mask - 1u] = COLLAPSE_FLAG;
    }
    workgroupBarrier();

    // [5] We process all the DP rounds
    // k = 2...7 -> the number of threads per processing round changes with the fixed scheduler 
    // hardcoded partly in the top part of the shader
    processDP(lane);
    workgroupBarrier();

    // [6] If we find no improvement, well we skip the rest...
    // I cannot return here THERE IS A CONDITION THAT ALL THREADS MUST REACH ALL WORKGROUP BARRIERS, 
    // so instead I use a flag that only the first lane will set.
    if (isFirstLane) 
    {
        shouldWriteBack = select(0u, 1u, a_copt[126u] < nodeCost[0]);
    }
    workgroupBarrier();


    // [7] Now that we have the optimal partitioning
    // We just need to reconstruct it
    if (isFirstLane && shouldWriteBack == 1u) 
    {
        reconstructOptimalTreelet();
    }
    workgroupBarrier();

    // [8] Write Back.
    // We can use one thread per internal node
    if (lane < 6u && shouldWriteBack == 1u) 
    {
        let slot = lane;
        let bvhIndex = nodeIndex[slot];

        internalNodes[bvhIndex].left = nodeLeft[slot];
        internalNodes[bvhIndex].right = nodeRight[slot];
        internalNodes[bvhIndex].aabbMin = nodeAABBMin[slot];
        internalNodes[bvhIndex].aabbMax = nodeAABBMax[slot];
        internalNodes[bvhIndex].triangleCount = nodeTriCount[slot];
        internalNodes[bvhIndex].sahCost = nodeCost[slot];
        internalNodes[bvhIndex].flags = nodeCollapse[slot];
    }
    workgroupBarrier();

    // [9] fix child parent pointers
    if (lane < 6u && shouldWriteBack == 1u) 
    {
        let slot = lane;
        let parentBVHIndex = nodeIndex[slot];

        let leftChild = nodeLeft[slot];
        if ((leftChild & LEAF_BIT) != 0u) // special case since leafParents store only a single u32: parent pointer
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
fn processDP(lane: u32)
{
    // DP for k = 2...5
    if (lane < S2_COUNT) { processSubset(S2[lane]); }
    workgroupBarrier();

    if (lane < 32u) { processSubset(S3[lane]); }
    workgroupBarrier();
    if (lane < 3u) { processSubset(S3[32u + lane]); }
    workgroupBarrier();

    if (lane < 32u) { processSubset(S4[lane]); }
    workgroupBarrier();
    if (lane < 3u) { processSubset(S4[32u + lane]); }
    workgroupBarrier();

    if (lane < S5_COUNT) { processSubset(S5[lane]); }
    workgroupBarrier();

    // DP for k = 6
    // 7 subsets here,  31 partitions each, so 4 threads per subset (4 * 7 = 28 threads used)
    // Since we cant process all in parallel, we compute local minima per thread
    // then reduce to find global minima.

    let subsetIndex = lane / 4u; // 0...6
    let partitionIndex = lane % 4u; // 0...3

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
            if (count % 4u == partitionIndex) 
            {
                let c = a_copt[p - 1u] + a_copt[(s^p) - 1u];
                if (c < cs)
                {
                    cs = c;
                    ps = p;
                }
            }
            p = (p - delta) & s;
            count += 1u;
        }

        reduceC[lane] = cs;
        reduceP[lane] = ps;
    }
    workgroupBarrier();

    // reduction over local minima
    if (partitionIndex == 0u && subsetIndex < S6_COUNT)
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
        let t = triTotal[s - 1u];
        let area = a_copt[s - 1u];

        let splitCost = 1.2 * area + bc;
        let collapseCost = 1.0 * area * f32(t);
        let useCollapse = collapseCost <= splitCost;

        a_copt[s - 1u] = select(splitCost, collapseCost, useCollapse);
        subsetCollapse[s - 1u] = select(0u, COLLAPSE_FLAG, useCollapse);
        popt[s - 1u] = bp;
    }
    workgroupBarrier();

    // DP for k = 7
    // Only one subset and 63 partitions. One thread will do at most 2 partitions
    {
        let s = 127u;
        var threadBestC: f32 = 1e30;
        var threadBestP: u32 = 0u;
        var threadCount: u32 = 0u;

        let delta = (s - 1u) & s;
        var p = (~delta + 1u) & s;

        while (p != 0u && p != s)
        {
            if (threadCount % 32u == lane) 
            {
                let c = a_copt[p - 1u] + a_copt[(s^p) - 1u];
                if (c < threadBestC)
                {
                    threadBestC = c;
                    threadBestP = p;
                }
            }
            p = (p - delta) & s;
            threadCount += 1u;
        }

        reduceC[lane] = threadBestC;
        reduceP[lane] = threadBestP;
    }
    workgroupBarrier();

    // reduction 16 -> 8 -> 4 -> 2 -> 1
    if (lane < 16u) { if (reduceC[lane + 16u] < reduceC[lane]) { reduceC[lane] = reduceC[lane + 16u]; reduceP[lane] = reduceP[lane + 16u]; } }
    workgroupBarrier();
    if (lane < 8u) { if (reduceC[lane + 8u] < reduceC[lane]) { reduceC[lane] = reduceC[lane + 8u]; reduceP[lane] = reduceP[lane + 8u]; } }
    workgroupBarrier();
    if (lane < 4u) { if (reduceC[lane + 4u] < reduceC[lane]) { reduceC[lane] = reduceC[lane + 4u]; reduceP[lane] = reduceP[lane + 4u]; } }
    workgroupBarrier();
    if (lane < 2u) { if (reduceC[lane + 2u] < reduceC[lane]) { reduceC[lane] = reduceC[lane + 2u]; reduceP[lane] = reduceP[lane + 2u]; } }
    workgroupBarrier();
    if (lane == 0u) 
    {
        if (reduceC[1u] < reduceC[0u]) 
        {
            reduceC[0u] = reduceC[1u];
            reduceP[0u] = reduceP[1u];
        }

        let s = 127u;
        let t = triTotal[s - 1u];
        let area = a_copt[s - 1u];

        let splitCost = 1.2 * area + reduceC[0u];
        let collapseCost = 1.0 * area * f32(t);
        let useCollapse = collapseCost <= splitCost;

        a_copt[s - 1u] = select(splitCost, collapseCost, useCollapse);
        subsetCollapse[s - 1u] = select(0u, COLLAPSE_FLAG, useCollapse);
        popt[s - 1u] = reduceP[0u];
    }
    workgroupBarrier();
}

//================================//
// Sequential small stack reconstruction of the tree.
// Keeping original internal nodes in seperate array
// as per the paper recommendation.
// We take the DP solution stored in popt 
fn reconstructOptimalTreelet() // Should be called on one thread only
{
    var originalInternalNodes: array<u32, 6>;
    for (var i = 0u; i < 6u; i++) 
    {
        originalInternalNodes[i] = nodeIndex[i];
    }

    var smallStack: array<u32, 6>;
    var stackSlot: array<u32, 6>;
    var stackPointer: u32 = 0u;
    var nextSlot: u32 = 0u;

    smallStack[0u] = 127u; // Always start from top subset with whole set of leaves
    stackSlot[0u] = 0u;
    stackPointer = 1u;
    
    while (stackPointer > 0u)
    {
        stackPointer -= 1u;
        let currentSubset = smallStack[stackPointer]; // ex: if is ABCDEFG
        let currentSlot = stackSlot[stackPointer];  
        let leftSubset = popt[currentSubset - 1u]; // Might be ABCD
        let rightSubset = currentSubset ^ leftSubset; // and then EFG -> propagating the DP solution

        // left
        var leftBVHIndex: u32;
        if (countOneBits(leftSubset) == 1u) // leaf
        {
            let leafBit: u32 = u32(firstTrailingBit(leftSubset));
            leftBVHIndex = nodeIndex[6u + leafBit];
        }
        else
        {
            nextSlot += 1u;
            smallStack[stackPointer] = leftSubset;
            stackSlot[stackPointer] = nextSlot;
            stackPointer += 1u;
            leftBVHIndex = originalInternalNodes[nextSlot];
        }

        // right
        var rightBVHIndex: u32;
        if (countOneBits(rightSubset) == 1u) // leaf
        {
            let leafBit: u32 = u32(firstTrailingBit(rightSubset));
            rightBVHIndex = nodeIndex[6u + leafBit];
        }
        else
        {
            nextSlot += 1u;
            smallStack[stackPointer] = rightSubset;
            stackSlot[stackPointer] = nextSlot;
            stackPointer += 1u;
            rightBVHIndex = originalInternalNodes[nextSlot];
        }

        nodeIndex[currentSlot] = originalInternalNodes[currentSlot];
        nodeLeft[currentSlot] = leftBVHIndex;
        nodeRight[currentSlot] = rightBVHIndex;

        var mn = vec3f(1e30);
        var mx = vec3f(-1e30);
        for (var i = 0u; i < N_LEAVES; i++) 
        {
            if ((currentSubset & (1u << i)) != 0u) 
            {
                mn = min(mn, nodeAABBMin[6u + i]);
                mx = max(mx, nodeAABBMax[6u + i]);
            }
        }
        nodeAABBMin[currentSlot] = mn;
        nodeAABBMax[currentSlot] = mx;
        nodeCost[currentSlot] = a_copt[currentSubset - 1u];
        nodeTriCount[currentSlot] = triTotal[currentSubset - 1u];
        nodeCollapse[currentSlot] = subsetCollapse[currentSubset - 1u]; // propagate collapse decision for next pass
    }
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
    let area = a_copt[s - 1u];

    let splitCost = 1.2 * area + cs;
    let collapseCost = 1.0 * area * f32(tri);

    let useCollapse = collapseCost <= splitCost;
    a_copt[s - 1u] = select(splitCost, collapseCost, useCollapse);
    subsetCollapse[s - 1u] = select(0u, COLLAPSE_FLAG, useCollapse);
    popt[s - 1u] = ps;
}

//================================//
fn surfaceArea(aabbMin: vec3f, aabbMax: vec3f) -> f32
{
    let d = aabbMax - aabbMin;
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

//================================//
fn ballotIsZero(mask: vec4<u32>) -> bool
{
    return all(mask == vec4<u32>(0u));
}

//================================//
fn firstLaneInBallot(m: vec4<u32>) -> u32 
{
    if (m.x != 0u) { return firstTrailingBit(m.x); }
    if (m.y != 0u) { return 32u + firstTrailingBit(m.y); }
    if (m.z != 0u) { return 64u + firstTrailingBit(m.z); }
    return 96u + firstTrailingBit(m.w);
}

//================================//
fn clearLaneInBallot(m: vec4<u32>, lane: u32) -> vec4<u32> 
{
    var out = m;
    let word = lane >> 5u;
    let bit  = lane & 31u;
    let mask = ~(1u << bit);

    switch word {
        case 0u: { out.x = out.x & mask; }
        case 1u: { out.y = out.y & mask; }
        case 2u: { out.z = out.z & mask; }
        default: { out.w = out.w & mask; }
    }
    return out;
}

//================================//
fn loadNodeFromBVH(slot: u32, bvhIndex: u32) 
{
    nodeIndex[slot] = bvhIndex;

    if ((bvhIndex & LEAF_BIT) != 0u) 
    {
        let li = bvhIndex & 0x7FFFFFFFu;
        nodeAABBMin[slot] = leafAABBs[li].aabbMin;
        nodeAABBMax[slot] = leafAABBs[li].aabbMax;
        nodeCost[slot] = surfaceArea(leafAABBs[li].aabbMin, leafAABBs[li].aabbMax);
        nodeTriCount[slot] = 1u;
        nodeSA[slot] = nodeCost[slot];
        nodeLeft[slot] = 0u;
        nodeRight[slot] = 0u;
        nodeCollapse[slot] = COLLAPSE_FLAG; // By default being a leaf means collapse wins
    } 
    else 
    {
        nodeAABBMin[slot] = internalNodes[bvhIndex].aabbMin;
        nodeAABBMax[slot] = internalNodes[bvhIndex].aabbMax;
        nodeCost[slot] = internalNodes[bvhIndex].sahCost;
        nodeTriCount[slot] = internalNodes[bvhIndex].triangleCount;
        nodeSA[slot] = surfaceArea(nodeAABBMin[slot], nodeAABBMax[slot]);
        nodeLeft[slot] = internalNodes[bvhIndex].left;
        nodeRight[slot] = internalNodes[bvhIndex].right;
        nodeCollapse[slot] = internalNodes[bvhIndex].flags & COLLAPSE_FLAG;
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