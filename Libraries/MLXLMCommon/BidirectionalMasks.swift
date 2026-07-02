// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Bidirectional attention mask: every query position attends to every kv
/// position equally (no causal restriction).
///
/// Used by the MTP drafter, whose queries sit at a single constant position
/// outside the target's KV cache and need full visibility into the target's
/// shared K/V pool. Returned as an additive mask (`0` for attend, `-inf` for
/// mask) compatible with `MLXFast.ScaledDotProductAttentionMaskMode.array`.
///
/// - Parameters:
///   - queryLen: number of query tokens (typically `1` for MTP drafting)
///   - kvLen: total kv positions in the shared pool
///   - dtype: array dtype (must match the queries' dtype)
/// - Returns: `[queryLen, kvLen]` array of zeros.
public func createBidirectionalMask(
    queryLen: Int,
    kvLen: Int,
    dtype: DType
) -> MLXArray {
    MLXArray.zeros([queryLen, kvLen], dtype: dtype)
}

/// Bidirectional sliding-window attention mask: when `kvLen` exceeds
/// `windowSize`, each query attends to a contiguous window of the first
/// `windowSize` kv positions; the remaining `kvLen - windowSize` positions
/// are masked with `-inf`. When `windowSize >= kvLen` (the degenerate case
/// — and the production MTP path, where the sliding-attention KV cache is
/// capped at `windowSize` by `RotatingKVCache`), the helper early-exits to
/// an all-zeros mask (every query attends to every kv position).
///
/// Matches the fixture convention in `tools/fixtures/masks/` and HF
/// Transformers' `create_bidirectional_sliding_window_mask` (after the
/// kv-axis flip).
///
/// Contract for the non-degenerate path (`kvLen > windowSize`): the mask
/// is built from an absolute-position predicate (`kIdx < windowSize`), NOT
/// a distance-from-`queryOffset` predicate. This variant is kept for the
/// pinned fixture tests; production MTP drafting uses the offset-aware
/// `createBidirectionalSlidingWindowMask(queryLen:kvLen:windowSize:queryOffset:dtype:)`
/// below, which windows by distance from the query position and therefore
/// stays correct when the (temporally ordered) KV pool grows beyond
/// `windowSize`.
///
/// - Parameters:
///   - queryLen: number of query tokens
///   - kvLen: total kv positions
///   - windowSize: sliding window size
///   - dtype: array dtype (must match the queries' dtype)
/// - Returns: `[queryLen, kvLen]` additive mask.
public func createBidirectionalSlidingWindowMask(
    queryLen: Int,
    kvLen: Int,
    windowSize: Int,
    dtype: DType
) -> MLXArray {
    if windowSize >= kvLen {
        return MLXArray.zeros([queryLen, kvLen], dtype: dtype)
    }
    let kIdx = MLXArray(Int32(0) ..< Int32(kvLen))
    let attend = kIdx .< Int32(windowSize)
    let row = MLX.where(
        attend,
        MLXArray(0, dtype: dtype),
        MLXArray(-Float.infinity, dtype: dtype)
    )
    // Broadcast the row across queryLen rows.
    return broadcast(row[.newAxis, 0...], to: [queryLen, kvLen])
}

/// Offset-aware bidirectional sliding-window mask for MTP drafting over a
/// temporally ordered KV pool (kv index `j` == absolute position `j`).
///
/// All draft queries sit at the single constant position `queryOffset`
/// (`draftBlock` reuses the same offset for every drafted token), so the
/// window is a distance predicate from that one position: attend iff
/// `j >= queryOffset - windowSize`. The inclusive bound replicates exactly
/// the content of the former `RotatingKVCache` pool (the last `windowSize`
/// tokens before the query) and degenerates to the all-zeros mask whenever
/// the pool is shorter than the window — including every fixture case of the
/// absolute-position variant above.
///
/// - Parameters:
///   - queryLen: number of query tokens
///   - kvLen: total kv positions in the (temporally ordered) pool
///   - windowSize: sliding window size
///   - queryOffset: absolute position of the draft queries
///   - dtype: array dtype (must match the queries' dtype)
/// - Returns: `[queryLen, kvLen]` additive mask.
public func createBidirectionalSlidingWindowMask(
    queryLen: Int,
    kvLen: Int,
    windowSize: Int,
    queryOffset: Int,
    dtype: DType
) -> MLXArray {
    let lower = queryOffset - windowSize
    if lower <= 0 {
        return MLXArray.zeros([queryLen, kvLen], dtype: dtype)
    }
    let kIdx = MLXArray(Int32(0) ..< Int32(kvLen))
    let row = MLX.where(
        kIdx .>= Int32(lower),
        MLXArray(0, dtype: dtype),
        MLXArray(-Float.infinity, dtype: dtype)
    )
    return broadcast(row[.newAxis, 0...], to: [queryLen, kvLen])
}
