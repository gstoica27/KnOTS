import torch


def tv_merging(vectors, weights=None, merging_type='unmerged', **kwargs):

    rows_to_keep = [torch.ones_like(vectors[i]) for i in range(len(vectors))]
    vectors_ = torch.vstack(vectors).clone()
    if merging_type == 'unmerged':
        return vectors_, rows_to_keep
    
    if weights is not None:
        for vector in vectors_:
            vector *= weights[0]
    if merging_type == 'mean':
        result = torch.mean(vectors_, dim=0)
    else:
        result = torch.sum(vectors_, dim=0)
    return result, rows_to_keep, rows_to_keep


## TIES MERGING UTILS
def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100
    
    if K >= 1 and return_mask:
        return M, torch.ones_like(M).mean(dim=-1), torch.ones_like(M)
    elif K >= 1:
        return M, torch.ones_like(M).mean(dim=-1)

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
        
    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=-1), final_mask
    return M * final_mask, final_mask.float().mean(dim=-1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


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


def disjoint_merge(Tensor, merge_func, reference_sign_to_mult, weights=None):
    # If sign is provided then we select the corresponding entries and aggregate.
    if reference_sign_to_mult is not None:
        rows_to_keep = torch.where(
            reference_sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        
    selected_entries = Tensor * rows_to_keep
    if weights is not None:
        for selected_entrie in selected_entries:
            selected_entrie *= weights[0]
    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = chunked_sum(selected_entries)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= reference_sign_to_mult
    elif merge_func == 'unmerged':
        disjoint_aggs = selected_entries
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs, rows_to_keep

def resolve_sign(Tensor, mode=None):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult

def ties_merging(vectors, topK=10, merging_type='mean', weights=None, **kwargs):
    # Add functionality that allows some layers to not be pruned or lets them be skipped
    print(f'TopK is: {topK}')
    print(f'weights is: {weights}')
    stacked_vectors = torch.vstack(vectors).clone()
    pruned_vectors, _, mask = topk_values_mask(
        stacked_vectors, K=topK, return_mask=True
    )
    vector_signs = resolve_sign(pruned_vectors)
    assert vector_signs is not None
    merged_tv, rows_to_keep = disjoint_merge(pruned_vectors, merging_type, vector_signs, weights)
    return merged_tv, rows_to_keep, mask

