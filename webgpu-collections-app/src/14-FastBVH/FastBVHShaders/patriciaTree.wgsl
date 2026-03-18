// A Patricia tree (or radix tree) is a trie like tree structure, in our case with N morton codes,
// The nodes and their children (2 exactly per internal node) are decided by common prefix of these codes.
// https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf
// Karras 2012 says that, we are sure there are N - 1 internal nodes with N codes.
// This way, we assign one thread per internal node, 
// and each thread independently figures out its range and children using only the δ function

// The δ function is defined as:
// - δ(i, j) will be zero for a certain number of keys starting from ki and one for the remaining until kj.
// - We call the index of the last key where the bit is 0 a split position, γ ∈ [i, j−1].
// - δ(i, j) = δ(γ, γ+1)
// " The resulting subranges are given by [i, γ] and [γ + 1, j], and are further partitioned by the left
//  and right child node, respectively.".

// Special case for duplicate keys: they augment each key with a bit representation of their index.
// Ergo, k'i = ki ⊕ i with string concatenation. ". In practice, there is no need to actually store the
// augmented keys—it is enough to simply use i and j as a fallback if ki = k j when evaluating δ(i, j). "

//================================//
// PARALLEL CONSTRUCTION ALGORITHM OF THE TREE:
// We suppose Internal nodes and Leaf nodes are in two separate buffers.
// "We define our node layout so that the root is located at I0, and the indices of its children—as well 
// as the children of any internal node—are assigned according to its respective split position".
// - The left child is at Iγ if it covers more than one key, or Lγ if its a leaf.
// - The right child is at Iγ + 1 if it covers more than one key, or Lγ + 1 if its a leaf.
// "An important property of this particular layout is that the index of every internal node coincides 
// with either its first or its last key".
// 
// For each internal node [0, N - 2] in parallel:
//    d <- sign(δ(i, i + 1) − δ(i, i − 1))  // Direction of the range
//    δmin <- δ(i, i − d)                   // upper bound for the length of the range
//    lmax <- 2
//    while δ(i, i + lmax * d) > δmin do
//       lmax <- lmax * 2
// 
//    l <- 0                                // Find the other end with binary search
//    for t = lmax / 2, lmax / 4, ..., 1 do
//       if δ(i, i + (l + t) * d) > δmin then
//          l <- l + t
//    j <- i + l * d

//    δnode <- δ(i, j)                      // Find split position with binary search
//    s <- 0
//    for t = ceil(l / 2), ceil(l / 4), ..., 1 do
//       if δ(i, i + (s + t) * d) > δnode then
//          s <- s + t
//    γ <- i + s * d + min(d, 0)
//
//    if min(i, j) == γ then left child <- Lγ else left child <- Iγ
//    if max(i, j) == γ + 1 then right child <- Lγ + 1 else right child <- Iγ + 1
//    Ii <- (left child, right child)
// end for

//  δ(i, j) = −1 when j not in [0,n−1].

//================================//
override THREADS_PER_WORKGROUP: u32;
override INTERNAL_NODE_COUNT: u32;
override LEAF_NODE_COUNT: u32;

//================================//
struct BVHNode 
{
    aabbMin: vec3f,
    parent:  u32,
    aabbMax: vec3f,
    padding: u32,

    left:    u32,
    right:   u32,
    _pad0: u32,
    _pad1: u32,
}; // Size = 3 * 16 = 48 bytes.

//================================//
@group(0) @binding(0) var<storage, read> mortonCodes: array<u32>;
@group(0) @binding(1) var<storage, read_write> internalNodes: array<BVHNode>;
@group(0) @binding(2) var<storage, read_write> leafNodes: array<u32>;

//================================//
@compute
@workgroup_size(THREADS_PER_WORKGROUP, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3u)
{
    if (gid.x >= INTERNAL_NODE_COUNT) 
    {
        return;
    }

    let i = i32(gid.x);
    let d = sign(delta(i, i + 1) - delta(i, i - 1));
    let deltaMin = delta(i, i - d);
    
    var lmax = 2u;
    while (delta(i, i + lmax * d) > deltaMin)
    {
        lmax = lmax * 2u;
    }

    var l = 0u;
    var t = lmax / 2u;
    while (t >= 1u)
    {
        if (delta(i, i + (l + t) * d) > deltaMin)
        {
            l = l + t;
        }
        t = t >>= 1u;
    }
    let j = i + l * d;

    let deltaNode = delta(i, j);
    var s = 0;
    var step = i32((l + 1u) >> 1u); // ceil(l/2)
    while (step >= 1) 
    {
        if (delta(i, i + (s + step) * d) > deltaNode) {
            s = s + step;
        }
        if (step == 1) { break; }
        step = (step + 1) >> 1; // ceil(step/2)
    }
    let gamma = i + s * d + min(d, 0);

    const LEAF_BIT: u32 = 0x80000000u;
    let lo = min(i, j);
    let hi = max(i, j);

    if (lo == gamma)
    {
        internalNodes[u32(i)].left = u32(gamma) | LEAF_BIT; // Flag
        leafNodes[u32(gamma)] = u32(i); // store parent
    }   
    else
    {
        internalNodes[u32(i)].left = gamma;
        internalNodes[u32(gamma)].parent = u32(i);
    }

    if (max(i, j) == gamma + 1u)
    {
        internalNodes[u32(i)].right = u32(gamma + 1u) | LEAF_BIT;
        leafNodes[u32(gamma + 1u)] = u32(i);
    }
    else
    {
        internalNodes[u32(i)].right = u32(gamma + 1u);
        internalNodes[u32(gamma + 1u)].parent = u32(i);
    }
}

//================================//
// Small explanation on countLeadingZeros:
// countLeadingZeros(x) returns the number of leading zeros in the binary representation of x.
//================================//
fn delta(i: i32, j: i32) -> i32
{
    if (j < 0 || j >= i32(LEAF_NODE_COUNT))
    {
        return -1;
    }

    let codeI = mortonCodes[u32(i)];
    let codeJ = mortonCodes[u32(j)];

    let xorResult = codeI ^ codeJ;

    if (xorResult != 0u)
    {
        return i32(countLeadingZeros(xorResult));
    }

    // else, duplicate keys
    return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
}
