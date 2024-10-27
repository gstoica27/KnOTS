import torch
import pdb


def topk_values_mask(M, K=0.7):
    if K > 1:
        K /= 100

    if K >= 1:
        return M, torch.ones_like(M), torch.ones_like(M).mean(dim=-1)
    
    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements
    # Find the k-th smallest element by magnitude for each row
    if M.flatten().shape[-1] == 1:
        kth_values = M.abs()
    else:
        kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    if original_shape == M.squeeze().shape:
        final_mask = mask.squeeze()
        M = M.squeeze()
    else:
        final_mask = mask
    
    return M * final_mask, final_mask, final_mask.float().mean(dim=-1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor, mode='sum_of_values'):
    if mode == "sum_of_signs":
        sign_to_mult = torch.sign(torch.sum(torch.sign(Tensor), dim=0))
    elif mode == "sum_of_values":
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
        sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    else:
        raise ValueError(f'Unknown mode: {mode}. Pick from sum_of_signs or sum_of_values')
    return sign_to_mult


def ties_masking(vectors, topK=100, sign_resolve_mode='sum_of_values', **kwargs):
    stacked_vectors = torch.vstack(vectors).clone()
    pruned_vectors, prune_mask, _ = topk_values_mask(
        stacked_vectors, K=topK
    )
    vector_signs = resolve_sign(pruned_vectors, mode=sign_resolve_mode)
    assert vector_signs is not None
    sign_mask = torch.where(
            vector_signs.unsqueeze(0) > 0, pruned_vectors > 0, pruned_vectors < 0
        )
    ties_mask = sign_mask * prune_mask
    return ties_mask


def tv_masking(vectors, topK=100, **kwargs):
    stacked_vectors = torch.vstack(vectors).clone()
    pruned_vectors, prune_mask, _ = topk_values_mask(
        stacked_vectors, K=topK
    )
    return prune_mask


def chunked_disjoint_mean(vectors, chunk_size=10000):
    num_chunks = vectors.size(0) // chunk_size + (1 if vectors.size(0) % chunk_size != 0 else 0)
    total_sum = torch.zeros_like(vectors[0])
    non_zero_counts = torch.zeros_like(vectors[0])

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, vectors.size(0))
        chunk = vectors[start_idx:end_idx]

        # Calculate sum and non-zero counts for the chunk
        total_sum += torch.sum(chunk, dim=0)
        non_zero_counts += (chunk != 0).sum(dim=0)

    # Compute the disjoint mean
    disjoint_aggs = total_sum / torch.clamp(non_zero_counts.float(), min=1)
    disjoint_aggs[non_zero_counts == 0] = 0

    return disjoint_aggs

def chunked_sum(tensor, chunk_size=10000):
    num_chunks = tensor.size(0) // chunk_size + (1 if tensor.size(0) % chunk_size != 0 else 0)
    total_sum = torch.zeros_like(tensor[0])

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, tensor.size(0))
        chunk = tensor[start_idx:end_idx]

        # Add the sum of the current chunk to the total sum
        total_sum += torch.sum(chunk, dim=0)

    return total_sum

def masked_merge(vectors, merge_func, weights=None):
    if weights is not None:
        for vector in vectors:
            vector *= weights[0]
    if merge_func == "mean":
        disjoint_aggs = chunked_disjoint_mean(vectors, chunk_size=10000)
    elif merge_func == "sum":
        disjoint_aggs = chunked_sum(vectors) 
    elif merge_func == "max":
        disjoint_aggs = vectors.abs().max(dim=0)[0]
    elif merge_func == 'unmerged':
        disjoint_aggs = vectors
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs

