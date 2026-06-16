from utils import *
from itertools import combinations
import xml.etree.ElementTree as ET

AA = list("ARNDCQEGHILKMFPSTWYVBJZX*")
blosum_matrix = [
 [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1,-1,-1,-4],
 [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1,-2, 0,-1,-4],
 [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 4,-3, 0,-1,-4],
 [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4,-3, 1,-1,-4],
 [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-1,-3,-1,-4],
 [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0,-2, 4,-1,-4],
 [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1,-3, 4,-1,-4],
 [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-4,-2,-1,-4],
 [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0,-3, 0,-1,-4],
 [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3, 3,-3,-1,-4],
 [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4, 3,-3,-1,-4],
 [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0,-3, 1,-1,-4],
 [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3, 2,-1,-1,-4],
 [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3, 0,-3,-1,-4],
 [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-3,-1,-1,-4],
 [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0,-2, 0,-1,-4],
 [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1,-1,-1,-4],
 [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-2,-2,-1,-4],
 [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-1,-2,-1,-4],
 [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3, 2,-2,-1,-4],
 [-2,-1, 4, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4,-3, 0,-1,-4],
 [-1,-2,-3,-3,-1,-2,-3,-4,-3, 3, 3,-3, 2, 0,-3,-2,-1,-2,-1, 2,-3, 3,-3,-1,-4],
 [-1, 0, 0, 1,-3, 4, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-2,-2,-2, 0,-3, 4,-1,-4],
 [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-4],
 [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1],
]

BLOSUM62 = {}
for i, a in enumerate(AA):
    for j, b in enumerate(AA):
        BLOSUM62[(a, b)] = blosum_matrix[i][j]

def blosum(a, b):
    return BLOSUM62.get((a.upper(), b.upper()), 0.0)

def pairwise_blosum(sa, sb):
    score = np.zeros((len(sa), len(sb)))
    for i in range(len(sa)):
        for j in range(len(sb)):
            score[i, j] = blosum(sa[i], sb[j])
    return score

def l2_similarity(e0, e1, w, reciprocal=200.0, blur=3.0, device='cuda'): # best value so far is 50! 
    # compute pairwise L2 distances
    e0, e1 = e0.to(device), e1.to(device)
    e0_sq = (e0 ** 2).sum(dim=1, keepdim=True)
    e1_sq = (e1 ** 2).sum(dim=1, keepdim=True).T
    dot = e0 @ e1.T
    sim = -torch.sqrt(torch.clamp(e0_sq + e1_sq - 2 * dot, min=1e-8))
    if w > 0:
        # convert to window form 
        L1, L2 = sim.shape
        window_size = 2 * w + 1
        i_idx = torch.arange(L1 - window_size + 1, device=sim.device).view(-1, 1, 1)  # [out_h, 1, 1]
        j_idx = torch.arange(L2 - window_size + 1, device=sim.device).view(1, -1, 1)  # [1, out_w, 1]
        t_idx = torch.arange(window_size, device=sim.device).view(1, 1, -1)      # [1, 1, w]
        sim = sim[i_idx + t_idx, j_idx + t_idx]
        # apply Gaussian-weighted blur
        mask1, mask2 = torch.ones(L1, device=sim.device), torch.ones(L2, device=sim.device)
        mask1[:w], mask1[-w:], mask2[:w], mask2[-w:] = 0., 0., 0., 0.
        mask1_win = mask1.unfold(0, window_size, 1)[:L1]
        mask2_win = mask2.unfold(0, window_size, 1)[:L2]
        valid_mask = mask1_win[:, None, :] * mask2_win[None, :, :]
        weights = torch.exp(-0.5 * ((torch.arange(window_size, device=sim.device) - w) / blur)**2)
        numerator = (sim * weights * valid_mask).sum(dim=-1)
        denominator = (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
        S = numerator / denominator
        del e0, e1, e0_sq, e1_sq, dot, sim, mask1, mask2, mask1_win, mask2_win, valid_mask, weights, numerator, denominator
    else:
        S = sim
        del e0, e1, e0_sq, e1_sq, dot, sim
    S = reciprocal * torch.log(.5 * (F.softmax(S, dim=-1) + F.softmax(S, dim=-2)) + 1e-8) + S
    torch.cuda.empty_cache()
    return S

def l2_similarity_squared_dist(e0, e1, w, reciprocal=200.0, blur=3.0, device='cuda'): # best value so far is 50! 
    # compute pairwise L2 distances
    e0, e1 = e0.to(device), e1.to(device)
    e0_sq = (e0 ** 2).sum(dim=1, keepdim=True)
    e1_sq = (e1 ** 2).sum(dim=1, keepdim=True).T
    dot = e0 @ e1.T
    sim = -torch.clamp(e0_sq + e1_sq - 2 * dot, min=1e-8)
    if w > 0:
        # convert to window form 
        L1, L2 = sim.shape
        window_size = 2 * w + 1
        i_idx = torch.arange(L1 - window_size + 1, device=sim.device).view(-1, 1, 1)  # [out_h, 1, 1]
        j_idx = torch.arange(L2 - window_size + 1, device=sim.device).view(1, -1, 1)  # [1, out_w, 1]
        t_idx = torch.arange(window_size, device=sim.device).view(1, 1, -1)      # [1, 1, w]
        sim = sim[i_idx + t_idx, j_idx + t_idx]
        # apply Gaussian-weighted blur
        mask1, mask2 = torch.ones(L1, device=sim.device), torch.ones(L2, device=sim.device)
        mask1[:w], mask1[-w:], mask2[:w], mask2[-w:] = 0., 0., 0., 0.
        mask1_win = mask1.unfold(0, window_size, 1)[:L1]
        mask2_win = mask2.unfold(0, window_size, 1)[:L2]
        valid_mask = mask1_win[:, None, :] * mask2_win[None, :, :]
        weights = torch.exp(-0.5 * ((torch.arange(window_size, device=sim.device) - w) / blur)**2)
        numerator = (sim * weights * valid_mask).sum(dim=-1)
        denominator = (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
        S = numerator / denominator
        del e0, e1, e0_sq, e1_sq, dot, sim, mask1, mask2, mask1_win, mask2_win, valid_mask, weights, numerator, denominator
    else:
        S = sim
        del e0, e1, e0_sq, e1_sq, dot, sim
    S = reciprocal * torch.log(.5 * (F.softmax(S, dim=-1) + F.softmax(S, dim=-2)) + 1e-8) + S
    torch.cuda.empty_cache()
    return S

def l2_similarity_only_reciprocal(e0, e1, w, reciprocal=200.0, blur=3.0, device='cuda'): # best value so far is 50! 
    # compute pairwise L2 distances
    e0, e1 = e0.to(device), e1.to(device)
    e0_sq = (e0 ** 2).sum(dim=1, keepdim=True)
    e1_sq = (e1 ** 2).sum(dim=1, keepdim=True).T
    dot = e0 @ e1.T
    sim = -torch.sqrt(torch.clamp(e0_sq + e1_sq - 2 * dot, min=1e-8))
    if w > 0:
        # convert to window form 
        L1, L2 = sim.shape
        window_size = 2 * w + 1
        i_idx = torch.arange(L1 - window_size + 1, device=sim.device).view(-1, 1, 1)  # [out_h, 1, 1]
        j_idx = torch.arange(L2 - window_size + 1, device=sim.device).view(1, -1, 1)  # [1, out_w, 1]
        t_idx = torch.arange(window_size, device=sim.device).view(1, 1, -1)      # [1, 1, w]
        sim = sim[i_idx + t_idx, j_idx + t_idx]
        # apply Gaussian-weighted blur
        mask1, mask2 = torch.ones(L1, device=sim.device), torch.ones(L2, device=sim.device)
        mask1[:w], mask1[-w:], mask2[:w], mask2[-w:] = 0., 0., 0., 0.
        mask1_win = mask1.unfold(0, window_size, 1)[:L1]
        mask2_win = mask2.unfold(0, window_size, 1)[:L2]
        valid_mask = mask1_win[:, None, :] * mask2_win[None, :, :]
        weights = torch.exp(-0.5 * ((torch.arange(window_size, device=sim.device) - w) / blur)**2)
        numerator = (sim * weights * valid_mask).sum(dim=-1)
        denominator = (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
        S = numerator / denominator
        del e0, e1, e0_sq, e1_sq, dot, sim, mask1, mask2, mask1_win, mask2_win, valid_mask, weights, numerator, denominator
    else:
        S = sim
        del e0, e1, e0_sq, e1_sq, dot, sim
    S = reciprocal * torch.log(.5 * (F.softmax(S, dim=-1) + F.softmax(S, dim=-2)) + 1e-8)
    torch.cuda.empty_cache()
    return S

def l2_similarity_geometric_mean(e0, e1, w, reciprocal=200.0, blur=3.0, device='cuda'): # best value so far is 50! 
    # compute pairwise L2 distances
    e0, e1 = e0.to(device), e1.to(device)
    e0_sq = (e0 ** 2).sum(dim=1, keepdim=True)
    e1_sq = (e1 ** 2).sum(dim=1, keepdim=True).T
    dot = e0 @ e1.T
    sim = -torch.sqrt(torch.clamp(e0_sq + e1_sq - 2 * dot, min=1e-8))
    if w > 0:
        # convert to window form 
        L1, L2 = sim.shape
        window_size = 2 * w + 1
        i_idx = torch.arange(L1 - window_size + 1, device=sim.device).view(-1, 1, 1)  # [out_h, 1, 1]
        j_idx = torch.arange(L2 - window_size + 1, device=sim.device).view(1, -1, 1)  # [1, out_w, 1]
        t_idx = torch.arange(window_size, device=sim.device).view(1, 1, -1)      # [1, 1, w]
        sim = sim[i_idx + t_idx, j_idx + t_idx]
        # apply Gaussian-weighted blur
        mask1, mask2 = torch.ones(L1, device=sim.device), torch.ones(L2, device=sim.device)
        mask1[:w], mask1[-w:], mask2[:w], mask2[-w:] = 0., 0., 0., 0.
        mask1_win = mask1.unfold(0, window_size, 1)[:L1]
        mask2_win = mask2.unfold(0, window_size, 1)[:L2]
        valid_mask = mask1_win[:, None, :] * mask2_win[None, :, :]
        weights = torch.exp(-0.5 * ((torch.arange(window_size, device=sim.device) - w) / blur)**2)
        numerator = (sim * weights * valid_mask).sum(dim=-1)
        denominator = (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
        S = numerator / denominator
        del e0, e1, e0_sq, e1_sq, dot, sim, mask1, mask2, mask1_win, mask2_win, valid_mask, weights, numerator, denominator
    else:
        S = sim
        del e0, e1, e0_sq, e1_sq, dot, sim
    S = reciprocal * torch.log((F.softmax(S, dim=-1) * F.softmax(S, dim=-2)) ** 0.5 + 1e-8) + S
    torch.cuda.empty_cache()
    return S

def l2_similarity_geometric_mean_squared_dist(e0, e1, w, reciprocal=200.0, blur=3.0, device='cuda'): # best value so far is 50! 
    # compute pairwise L2 distances
    e0, e1 = e0.to(device), e1.to(device)
    e0_sq = (e0 ** 2).sum(dim=1, keepdim=True)
    e1_sq = (e1 ** 2).sum(dim=1, keepdim=True).T
    dot = e0 @ e1.T
    sim = -torch.clamp(e0_sq + e1_sq - 2 * dot, min=1e-8)
    if w > 0:
        # convert to window form 
        L1, L2 = sim.shape
        window_size = 2 * w + 1
        i_idx = torch.arange(L1 - window_size + 1, device=sim.device).view(-1, 1, 1)  # [out_h, 1, 1]
        j_idx = torch.arange(L2 - window_size + 1, device=sim.device).view(1, -1, 1)  # [1, out_w, 1]
        t_idx = torch.arange(window_size, device=sim.device).view(1, 1, -1)      # [1, 1, w]
        sim = sim[i_idx + t_idx, j_idx + t_idx]
        # apply Gaussian-weighted blur
        mask1, mask2 = torch.ones(L1, device=sim.device), torch.ones(L2, device=sim.device)
        mask1[:w], mask1[-w:], mask2[:w], mask2[-w:] = 0., 0., 0., 0.
        mask1_win = mask1.unfold(0, window_size, 1)[:L1]
        mask2_win = mask2.unfold(0, window_size, 1)[:L2]
        valid_mask = mask1_win[:, None, :] * mask2_win[None, :, :]
        weights = torch.exp(-0.5 * ((torch.arange(window_size, device=sim.device) - w) / blur)**2)
        numerator = (sim * weights * valid_mask).sum(dim=-1)
        denominator = (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
        S = numerator / denominator
        del e0, e1, e0_sq, e1_sq, dot, sim, mask1, mask2, mask1_win, mask2_win, valid_mask, weights, numerator, denominator
    else:
        S = sim
        del e0, e1, e0_sq, e1_sq, dot, sim
    S = reciprocal * torch.log((F.softmax(S, dim=-1) * F.softmax(S, dim=-2)) ** 0.5 + 1e-8) + S
    torch.cuda.empty_cache()
    return S

def l2_similarity_geometric_mean_only_reciprocal(e0, e1, w, reciprocal=200.0, blur=3.0, device='cuda'): # best value so far is 50! 
    # compute pairwise L2 distances
    e0, e1 = e0.to(device), e1.to(device)
    e0_sq = (e0 ** 2).sum(dim=1, keepdim=True)
    e1_sq = (e1 ** 2).sum(dim=1, keepdim=True).T
    dot = e0 @ e1.T
    sim = -torch.sqrt(torch.clamp(e0_sq + e1_sq - 2 * dot, min=1e-8))
    if w > 0:
        # convert to window form 
        L1, L2 = sim.shape
        window_size = 2 * w + 1
        i_idx = torch.arange(L1 - window_size + 1, device=sim.device).view(-1, 1, 1)  # [out_h, 1, 1]
        j_idx = torch.arange(L2 - window_size + 1, device=sim.device).view(1, -1, 1)  # [1, out_w, 1]
        t_idx = torch.arange(window_size, device=sim.device).view(1, 1, -1)      # [1, 1, w]
        sim = sim[i_idx + t_idx, j_idx + t_idx]
        # apply Gaussian-weighted blur
        mask1, mask2 = torch.ones(L1, device=sim.device), torch.ones(L2, device=sim.device)
        mask1[:w], mask1[-w:], mask2[:w], mask2[-w:] = 0., 0., 0., 0.
        mask1_win = mask1.unfold(0, window_size, 1)[:L1]
        mask2_win = mask2.unfold(0, window_size, 1)[:L2]
        valid_mask = mask1_win[:, None, :] * mask2_win[None, :, :]
        weights = torch.exp(-0.5 * ((torch.arange(window_size, device=sim.device) - w) / blur)**2)
        numerator = (sim * weights * valid_mask).sum(dim=-1)
        denominator = (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
        S = numerator / denominator
        del e0, e1, e0_sq, e1_sq, dot, sim, mask1, mask2, mask1_win, mask2_win, valid_mask, weights, numerator, denominator
    else:
        S = sim
        del e0, e1, e0_sq, e1_sq, dot, sim
    S = reciprocal * torch.log((F.softmax(S, dim=-1) * F.softmax(S, dim=-2)) ** 0.5 + 1e-8)
    torch.cuda.empty_cache()
    return S

def cosine_similarity(e0, e1, w, reciprocal=200.0, blur=3.0, device='cuda'):
    # compute pairwise cosine similarity
    e0, e1 = e0.to(device), e1.to(device)
    e0 = F.normalize(e0, dim=1)
    e1 = F.normalize(e1, dim=1)
    sim = e0 @ e1.T  # cosine similarity

    if w > 0:
        # convert to window form
        L1, L2 = sim.shape
        window_size = 2 * w + 1
        i_idx = torch.arange(L1 - window_size + 1, device=sim.device).view(-1, 1, 1)
        j_idx = torch.arange(L2 - window_size + 1, device=sim.device).view(1, -1, 1)
        t_idx = torch.arange(window_size, device=sim.device).view(1, 1, -1)
        sim = sim[i_idx + t_idx, j_idx + t_idx]

        # apply Gaussian-weighted blur
        mask1, mask2 = torch.ones(L1, device=sim.device), torch.ones(L2, device=sim.device)
        mask1[:w], mask1[-w:], mask2[:w], mask2[-w:] = 0., 0., 0., 0.
        mask1_win = mask1.unfold(0, window_size, 1)[:L1]
        mask2_win = mask2.unfold(0, window_size, 1)[:L2]
        valid_mask = mask1_win[:, None, :] * mask2_win[None, :, :]

        weights = torch.exp(
            -0.5 * ((torch.arange(window_size, device=sim.device) - w) / blur) ** 2
        )
        numerator = (sim * weights * valid_mask).sum(dim=-1)
        denominator = (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
        S = numerator / denominator

        del e0, e1, sim, mask1, mask2, mask1_win, mask2_win, valid_mask, weights, numerator, denominator
    else:
        S = sim
        del e0, e1, sim

    # reciprocal reweighting (same as L2 version)
    S = reciprocal * torch.log(
        0.5 * (F.softmax(S, dim=-1) + F.softmax(S, dim=-2)) + 1e-8
    ) + S

    torch.cuda.empty_cache()
    return S


def load_core_domain(xml_path, msa_order):
    """Parse BAliBASE XML and return (domain_targets, domain_reference, residue_index_sets) for core-block scoring."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    seq_map = {}
    for seq_node in root.findall('.//sequence'):
        name_node, data_node = seq_node.find('seq-name'), seq_node.find('seq-data')
        if name_node is None or data_node is None or not name_node.text or not data_node.text:
            continue
        seq_map[name_node.text.strip()] = data_node.text.replace('\n', '').replace(' ', '').upper().replace('.', '-')
    if not seq_map:
        return None

    norm_map = {''.join(c for c in k.upper() if c.isalnum()): v for k, v in seq_map.items()}
    ordered = []
    for key in msa_order:
        seq = seq_map.get(key) or norm_map.get(''.join(c for c in key.upper() if c.isalnum()))
        if seq is None:
            return None
        ordered.append(seq)

    core_mask = None
    for colscore in root.findall('.//column-score'):
        name_node = colscore.find('colsco-name')
        if name_node is None or (name_node.text or '').strip() != 'coreblock':
            continue
        data_node = colscore.find('colsco-data')
        if data_node is not None and data_node.text:
            try:
                core_mask = [int(tok) == 1 for tok in data_node.text.split()]
            except ValueError:
                pass
        break
    if core_mask is None:
        return None

    aln_len = len(ordered[0])
    if len(core_mask) < aln_len:
        core_mask = core_mask + [False] * (aln_len - len(core_mask))
    else:
        core_mask = core_mask[:aln_len]

    keep_cols = [i for i, keep in enumerate(core_mask) if keep]
    if not keep_cols:
        return None

    domain_reference = [''.join(seq[i] for i in keep_cols) for seq in ordered]
    domain_targets = [seq.replace('-', '') for seq in domain_reference]

    residue_index_sets = []
    for seq in ordered:
        keep, res_idx = set(), 0
        for col, ch in enumerate(seq):
            if ch != '-':
                if core_mask[col]:
                    keep.add(res_idx)
                res_idx += 1
        residue_index_sets.append(keep)

    return domain_targets, domain_reference, residue_index_sets


def project_to_domain(full_aln, idx_sets, targets):
    """Project a full predicted alignment to core-block residues only."""
    if not full_aln or len(full_aln) != len(idx_sets) or len(full_aln) != len(targets):
        return None
    ncols = len(full_aln[0])
    if any(len(s) != ncols for s in full_aln):
        return None

    projected_rows = []
    for seq, keep_indices in zip(full_aln, idx_sets):
        row, res_idx = [], 0
        for ch in seq:
            if ch == '-':
                row.append('-')
            else:
                row.append(ch if res_idx in keep_indices else '-')
                res_idx += 1
        projected_rows.append(row)

    keep_cols = [c for c in range(ncols) if any(projected_rows[r][c] != '-' for r in range(len(projected_rows)))]
    if not keep_cols:
        return None

    domain_aln = [''.join(row[c] for c in keep_cols) for row in projected_rows]
    if any(s.replace('-', '') != t for s, t in zip(domain_aln, targets)):
        return None
    return domain_aln
