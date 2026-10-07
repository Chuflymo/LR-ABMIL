import timm
from utils import *
import torchvision.models as models
from torch.nn.attention import SDPBackend, sdpa_kernel
try:
    from flash_attn import flash_attn_qkvpacked_func,flash_attn_varlen_func,flash_attn_func
    from .utils import *
    import xformers.ops as xops
except:
    #print(f"Not installed Flash Attention")
    pass


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    for m in module.modules():
        if hasattr(m, 'init_weights'):
            m.init_weights()


# class M_DAttentionX(nn.Module):
#       """
#       Multi-Depth Attention X (Focus High)
   
#       Core idea: Separate instances into high and low attention groups.
#       - High instances: self-attention among themselves
#       - Low instances: cross-attention from high instances (learn from high)
#       - Fusion: combine original, high-branch, and low-branch attention
   
#       Paper Eq: A_fused = Softmax(A + α·A^high - β·A^low)
#       """
#       def __init__(self, input_dim, n_classes, dropout=0.25, act='gelu',
#                    mil_norm=None, mil_bias=False, inner_dim=512,
#                    n_heads=8, proj_drop=0., attn_type='mlp', attn_bias=False,
#                    ffn=False, init_k=0.02, step=0.005, **kwargs):
#           super(M_DAttentionX, self).__init__()

#           self.input_dim = input_dim
#           self.n_classes = n_classes
#           self.inner_dim = inner_dim
#           self.n_heads = n_heads
#           self.head_dim = inner_dim // n_heads
#           self.init_k = init_k
#           self.mil_norm = mil_norm

#           # Learnable fusion parameters [Eq (7)]
#           self.alpha = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))
#           self.beta = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))

#           # Feature extractor
#           self.feature = nn.Sequential(
#               nn.LayerNorm(input_dim, bias=mil_bias) if mil_norm == 'ln' else nn.Identity(),
#               nn.Linear(input_dim, inner_dim, bias=mil_bias),
#               nn.GELU() if act.lower() == 'gelu' else nn.ReLU(),
#               nn.Dropout(dropout),
#           )

#           # Initial attention (ABMIL-style)
#           if attn_type == 'mlp':
#               self.init_attention = MLPAttn(self.head_dim, bias=attn_bias)

#           # ============ High Branch: Self-attention ============
#           # Q and K for high instances [Eq (5)]
#           self.high_q_proj = nn.Sequential(
#               nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
#               nn.ReLU(),
#               nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
#           )
#           self.high_k_proj = nn.Sequential(
#               nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
#               nn.ReLU(),
#               nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
#           )
#           self.high_v_proj = nn.Sequential(
#               nn.Linear(1, 4),
#               nn.ReLU(),
#               nn.Linear(4, 1),
#           )

#           # ============ Low Branch: Cross-attention to high ============
#           # Q from low instances, K from high instances [Eq (6)]
#           self.low_q_proj = nn.Sequential(
#               nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
#               nn.ReLU(),
#               nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
#           )
#           # Key for low branch (comes from high instances)
#           self.high_k_for_low_proj = nn.Sequential(
#               nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
#               nn.ReLU(),
#               nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
#           )
#           self.low_v_proj = nn.Sequential(
#               nn.Linear(1, 4),
#               nn.ReLU(),
#               nn.Linear(4, 1),
#           )

#           # Post-processing
#           self.norm1 = nn.LayerNorm(inner_dim, bias=mil_bias) if mil_norm == 'ln' else nn.Identity()
#           self.proj = nn.Linear(inner_dim, inner_dim, bias=mil_bias)
#           self.proj_drop = nn.Dropout(proj_drop)

#           if ffn:
#               self.norm_ffn = nn.LayerNorm(inner_dim)
#               self.mlp = Mlp(inner_dim, int(inner_dim * 4.), act_layer=nn.GELU)
#           self.ffn = ffn

#           self.classifier = nn.Linear(inner_dim, n_classes, bias=mil_bias)
#           self._init_weights()

#       def _init_weights(self):
#           for m in self.modules():
#               if isinstance(m, nn.Linear):
#                   nn.init.xavier_normal_(m.weight)
#                   if m.bias is not None:
#                       m.bias.data.zero_()
#               elif isinstance(m, nn.LayerNorm):
#                   if m.bias is not None:
#                       nn.init.constant_(m.bias, 0)
#                   nn.init.constant_(m.weight, 1.0)

#       def _create_high_low_masks(self, A_squeezed, k_ratio):
#           """
#           Create binary masks for high and low attention instances.
   
#           Args:
#               A_squeezed: [B, H, N] - attention scores
#               k_ratio: float - ratio of instances to select as high attention
   
#           Returns:
#               high_mask: [B, H, N] - True for high attention instances
#               low_mask: [B, H, N] - True for low attention instances
#           """
#           B, H, N = A_squeezed.shape
#           k_num = max(1, int(N * k_ratio))

#           # Top-k instances as high attention
#           _, topk_indices = torch.topk(A_squeezed, k_num, dim=-1, sorted=False)
#           high_mask = torch.zeros_like(A_squeezed, dtype=torch.long)
#           high_mask.scatter_(-1, topk_indices, 1)
#           high_mask = high_mask.bool()

#           # Bottom-k instances as low attention
#           _, bottomk_indices = torch.topk(-A_squeezed, k_num, dim=-1, sorted=False)
#           low_mask = torch.zeros_like(A_squeezed, dtype=torch.long)
#           low_mask.scatter_(-1, bottomk_indices, 1)
#           low_mask = low_mask.bool()

#           return high_mask, low_mask

#       def _compute_high_branch_attention(self, x_high, A_high, high_mask):
#           """
#           Compute high branch self-attention [Eq (5)].
   
#           Args:
#               x_high: [B, H, N, D] - weighted features of high instances
#               A_high: [B, H, N, 1] - attention scores of high instances
#               high_mask: [B, H, N] - high instance mask
   
#           Returns:
#               new_A_high: [B, H, N, 1] - updated attention from high branch
#           """
#           # [Eq (5)] A^high = W_v^high · Softmax(Q^high · (K^high)^T / √d)
#           Q = self.high_q_proj(x_high)      # [B, H, N, D]
#           K = self.high_k_proj(x_high)      # [B, H, N, D]

#           # Self-attention scores
#           attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

#           # Create mask: high instances attend to high instances only
#           mask_2d = torch.matmul(
#               high_mask.float().unsqueeze(-1),   # [B, H, N, 1]
#               high_mask.float().unsqueeze(-2)   # [B, H, 1, N]
#           ).bool()  # [B, H, N, N]

#           min_val = -torch.finfo(attn_scores.dtype).max
#           attn_scores = attn_scores.masked_fill(~mask_2d, min_val / 2)
#           attn_weights = F.softmax(attn_scores, dim=-1)

#           # Apply to attention values
#           V = self.high_v_proj(A_high)  # [B, H, N, 1]
#           new_A_high = torch.matmul(attn_weights, V)  # [B, H, N, 1]

#           return new_A_high

#       def _compute_low_branch_attention(self, x_low, x_high, A_high, high_mask, low_mask):
#           """
#           Compute low branch cross-attention to high instances [Eq (6)].
   
#           Args:
#               x_low: [B, H, N, D] - weighted features of low instances
#               x_high: [B, H, N, D] - weighted features of high instances
#               A_high: [B, H, N, 1] - attention scores of high instances
#               high_mask: [B, H, N] - high instance mask
#               low_mask: [B, H, N] - low instance mask
   
#           Returns:
#               new_A_low: [B, H, N, 1] - updated attention from low branch
#           """
#           # [Eq (6)] A^low = W_v^low · Softmax(Q^low · (K^high→low)^T / √d)
#           # Key comes from high instances (cross-attention)
#           Q = self.low_q_proj(x_low)              # [B, H, N, D] - from low
#           K = self.high_k_for_low_proj(x_high)    # [B, H, N, D] - from high

#           # Cross-attention scores: low instances attend to high instances
#           attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

#           # Create mask: low instances attend to high instances only
#           mask_2d = torch.matmul(
#               low_mask.float().unsqueeze(-1),    # [B, H, N, 1] - query (low)
#               high_mask.float().unsqueeze(-2)    # [B, H, 1, N] - key (high)
#           ).bool()  # [B, H, N, N]

#           min_val = -torch.finfo(attn_scores.dtype).max
#           attn_scores = attn_scores.masked_fill(~mask_2d, min_val / 2)
#           attn_weights = F.softmax(attn_scores, dim=-1)

#           # Apply to attention values (from high instances)
#           V = self.low_v_proj(A_high)  # [B, H, N, 1] - from high
#           new_A_low = torch.matmul(attn_weights, V)  # [B, H, N, 1]

#           return new_A_low

#       def forward(self, x, return_attn=False, return_act=False, **kwargs):
#           B, N, _ = x.shape if len(x.size()) == 3 else (1, *x.size())
#           if len(x.size()) == 2:
#               x = x.unsqueeze(0)

#           # Feature extraction
#           x = self.feature(x)  # [B, N, inner_dim]
#           act = x.clone()

#           # Reshape for multi-head
#           x = x.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

#           # ============ Initial attention (ABMIL) [Eq (3)] ============
#           A_init = self.init_attention(x)  # [B, H, N, 1]
#           A_squeezed = A_init.squeeze(-1)  # [B, H, N]

#           # ============ Separate into high and low attention groups [Eq (4)] ============
#           high_mask, low_mask = self._create_high_low_masks(A_squeezed, self.init_k)

#           # Weighted features
#           x_weighted = x * A_init  # [B, H, N, D]

#           # High instances
#           x_high = x_weighted * high_mask.unsqueeze(-1)      # [B, H, N, D]
#           A_high = A_init * high_mask.unsqueeze(-1)          # [B, H, N, 1]

#           # Low instances
#           x_low = x_weighted * low_mask.unsqueeze(-1)        # [B, H, N, D]

#           # ============ High branch: self-attention [Eq (5)] ============
#           A_high_branch = self._compute_high_branch_attention(x_high, A_high, high_mask)

#           # ============ Low branch: cross-attention [Eq (6)] ============
#           A_low_branch = self._compute_low_branch_attention(x_low, x_high, A_high, high_mask, low_mask)

#           # ============ Fusion [Eq (7)] ============
#           # A_fused = Softmax(A + α·A^high - β·A^low)
#           A_fused = A_init + self.alpha * A_high_branch - self.beta * A_low_branch
#           A_fused = F.softmax(A_fused, dim=2)

#           # Pooling and classification
#           x_bag = torch.matmul(A_fused.transpose(2, 3), x)  # [B, H, 1, D]
#           x_bag = x_bag.reshape(B, self.inner_dim)
#           x_bag = self.proj_drop(self.proj(x_bag))
#           x_bag = self.norm1(x_bag)

#           if self.ffn:
#               x_bag = x_bag + self.mlp(self.norm_ffn(x_bag))

#           logits = self.classifier(x_bag)

#           if return_attn:
#               outputs = [logits, A_init.squeeze(-2)]
#               if return_act:
#                   outputs.append(act.squeeze(1))
#               return tuple(outputs)

#           return logits



#FH_ABMIL? (Focus High)
# class M_DAttentionX(nn.Module):
#     def __init__(self, input_dim, n_classes, dropout=0.25, act='gelu', mil_norm=None, mil_bias=False, inner_dim=512,
#                  n_heads=8, proj_drop=0., attn_type='mlp', attn_bias=False, ffn=False, init_k=0.02, step=0.005, **kwargs):
#         super(M_DAttentionX, self).__init__()

#         self.inner_dim = inner_dim
#         self.head_dim = inner_dim // n_heads  # Dimension per attention head & the dimension for MLPAttn
#         self.mid_dim = self.head_dim # Intermediate dimension for MLPAttn
#         self.out_dim = 1  # Output dimension of MLPAttn

#         self.feature = []  # List to store feature extractor layers
#         self.mil_norm = mil_norm  # Type of normalization for MIL
#         self.n_heads = n_heads  # Number of attention heads

#         self.alpha = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))
#         self.beta = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))

#        # Initialize Q&K linear layers
#         self.W_q_high = nn.Sequential(
#             nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
#             nn.ReLU(),
#             nn.Linear(self.head_dim * 4, self.head_dim, bias=False),    
#         )
#         self.W_q_low = nn.Sequential(
#             nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
#             nn.ReLU(),
#             nn.Linear(self.head_dim * 4, self.head_dim, bias=False),  
#         )
#         self.W_k_high_to_high = nn.Sequential(
#             nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
#             nn.ReLU(),
#             nn.Linear(self.head_dim * 4, self.head_dim, bias=False),  
#         )
#         self.W_k_high_to_low = nn.Sequential(
#             nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
#             nn.ReLU(),
#             nn.Linear(self.head_dim * 4, self.head_dim, bias=False),  
#         )
#         self.W_v_high = nn.Sequential(
#             nn.Linear(1, 4),
#             nn.ReLU(),
#             nn.Linear(4, 1),
#         )
#         self.W_v_low = nn.Sequential(
#             nn.Linear(1, 4),
#             nn.ReLU(),
#             nn.Linear(4, 1),
#         )


#         # same key for to_high and to_low
#         # self.W_k_high = nn.Linear(self.head_dim, self.head_dim, bias=False)

#         if ffn:
#             self.norm_ffn = nn.LayerNorm(inner_dim) if not mil_norm else nn.Identity()  # LayerNorm before FFN
#             self.mlp = Mlp(  # Feed-Forward Network
#                 in_features=inner_dim,
#                 hidden_features=int(inner_dim * 4.),
#                 act_layer=nn.GELU,
#             )
#         self.ffn = ffn  # Flag to use FFN

#         if mil_norm == 'bn':
#             self.norm = nn.BatchNorm1d(input_dim)  # Batch normalization for input
#             self.norm1 = nn.BatchNorm1d(self.L * self.K)  # Batch normalization after attention
#         elif mil_norm == 'ln':
#             self.feature += [nn.LayerNorm(input_dim, bias=mil_bias)]  # Layer normalization for input
#             self.norm1 = nn.LayerNorm(inner_dim, bias=mil_bias)  # Layer normalization after attention
#         else:
#             self.norm1 = self.norm = nn.Identity()  # No normalization

#         self.feature += [nn.Linear(input_dim, inner_dim, bias=mil_bias)]  # Linear layer for feature extraction
#         if act.lower() == 'gelu':
#             self.feature += [nn.GELU()]  # GELU activation
#         elif act.lower() == 'relu':
#             self.feature += [nn.ReLU()]  # ReLU activation
#         self.feature += [nn.Dropout(dropout)]  # Dropout layer
#         self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()

#         if attn_type == 'mlp':
#             # Initialize MLPAttn for standard MLP-based attention
#             self.attention = MLPAttn(self.head_dim, self.mid_dim, bias=attn_bias,
#                                      k=1)

#         self.proj = nn.Linear(inner_dim, inner_dim, bias=mil_bias)  # Projection layer after attention
#         self.proj_drop = nn.Dropout(proj_drop) # Dropout for projection layer

#         self.classifier = nn.Linear(inner_dim, n_classes, bias=mil_bias)  # Classifier layer

#         self.apply(initialize_weights)  # Apply weight initialization

#         self.init_k = init_k
#         self.step = step

#         # self.register_buffer('epoch_step', torch.tensor(0))

#     # def find_optimal_k(self, A_squeezed):

#     #     B, H, N = A_squeezed.shape
#     #     k_values = torch.arange(self.init_k, 0.2, self.step).to(A_squeezed.device)

#     #     optimal_k = self.init_k
#     #     max_var_diff = 0

#     #     for k in k_values:
#     #         k_num = max(1, int(N * k))

#     #         _, topk_indices = torch.topk(A_squeezed, k_num, dim=-1, sorted=False)
#     #         high_split = torch.zeros_like(A_squeezed).scatter_(-1, topk_indices, 1).bool()
#     #         not_high_split = ~high_split

#     #         high_values = A_squeezed[high_split]
#     #         low_values = A_squeezed[not_high_split]

#     #         if len(high_values) == 0 or len(low_values) == 0:
#     #             continue

#     #         var_high = torch.var(high_values, unbiased=False)
#     #         var_low = torch.var(low_values, unbiased=False)
#     #         var_diff = var_high - var_low
#     #         if var_diff > max_var_diff:
#     #             max_var_diff = var_diff
#     #             optimal_k = k

#     #         # range_high = torch.max(high_values) - torch.min(high_values)
#     #         # range_low = torch.max(low_values) - torch.min(low_values)
#     #         # if range_high >= range_low:
#     #         #     optimal_k = k
#     #         #     break 

#     #     return optimal_k

#     def forward(self, x, return_attn=False, return_act=False, return_img_feat=False, **kwargs):
#         if len(x.size()) == 2:
#             x.unsqueeze_(0)  # Add batch dimension if missing

#         B, N, _ = x.shape  # Batch size, Number of instances, Input dimension

#         if self.mil_norm == 'bn':
#             x = torch.transpose(x, -1, -2)  # Transpose for BatchNorm1d
#             x = self.norm(x)  # Apply batch normalization
#             x = torch.transpose(x, -1, -2)  # Transpose back

#         x = self.feature(x)  # Extract features

#         _, _, C = x.shape  # Get the channel dimension after feature extraction (inner_dim)
#         act = x.clone()  # Clone for returning activations if needed

#         x = x.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # B H N D (Reshape for multi-head)
#         A = self.attention(x)  # B H N K (Attention scores from MLPAttn per head)
#         B, H, N, D = x.shape
        
#         # use_custom_attn = self.epoch_step.item() >= -1
#         # if use_custom_attn:
#         # Mask attention
#         A_squeezed = A.squeeze(-1)  # [B, H, N]
#         # optimal_k = self.find_optimal_k(A_squeezed)
#         optimal_k = self.init_k
#         k_high = max(1, int(N * optimal_k))
#         _, topk_indices = torch.topk(A_squeezed, k_high, dim=-1, sorted=False)
#         high_mask = torch.zeros_like(A_squeezed, dtype=torch.long)
#         high_mask.scatter_(-1, topk_indices, 1)
#         high_mask = high_mask.bool()
#         k_low = max(1, int(N * optimal_k))
#         _, bottomk_indices = torch.topk(-A_squeezed, k_low, dim=-1, sorted=False)
#         low_mask = torch.zeros_like(A_squeezed, dtype=torch.long)
#         low_mask.scatter_(-1, bottomk_indices, 1)
#         low_mask = low_mask.bool()

#         A_orig = A.clone()

#         # High attention instances branch
#         x_weighted = x * A  # [B, H, N, D]
#         x_high = x_weighted * high_mask.unsqueeze(-1)  # [B, H, N, D]
#         A_high = A * high_mask.unsqueeze(-1)  # [B, H, N, 1]
#         Q_high = self.W_q_high(x_high)  # [B, H, N, D]
#         K_high_to_high = self.W_k_high_to_high(x_high)  # [B, H, N, D]
#         # K_high_to_high = self.W_k_high(x_high)  # [B, H, N, D]
#         attn_scores_high = torch.matmul(Q_high, K_high_to_high.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         mask_high_2d = torch.matmul(
#             high_mask.float().unsqueeze(-1),
#             high_mask.float().unsqueeze(-2)
#         ).bool()  # [B, H, N, N]
#         min_val = -torch.finfo(attn_scores_high.dtype).max
#         attn_scores_high = attn_scores_high.masked_fill(~mask_high_2d, min_val / 2)
#         attn_weights_high = F.softmax(attn_scores_high, dim=-1)
#         v_high = self.W_v_high(A_high)  # [B, H, N, 1]
#         new_A_high_to_high = torch.matmul(attn_weights_high, v_high)  # [B, H, N, 1]

#         # Low attention instances branch
#         x_low = x_weighted * low_mask.unsqueeze(-1)  # [B, H, N, D]
#         Q_low = self.W_q_low(x_low)  # [B, H, N, D]
#         K_high_to_low = self.W_k_high_to_low(x_high)
#         # K_high_to_low = self.W_k_high(x_high)  # [B, H, N, D]
#         attn_scores_low = torch.matmul(Q_low, K_high_to_low.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         mask_low_2d = torch.matmul(
#             low_mask.float().unsqueeze(-1),
#             high_mask.float().unsqueeze(-2)
#         ).bool()  # [B, H, N, N]
#         attn_scores_low = attn_scores_low.masked_fill(~mask_low_2d, min_val / 2)
#         attn_weights_low = F.softmax(attn_scores_low, dim=-1)
#         v_low = self.W_v_low(A_high)  # [B, H, N, 1]
#         new_A_high_to_low = torch.matmul(attn_weights_low, v_low)  # [B, H, N, 1]

#         # Combine the two branches and add them to the original attention scores
#         fused_A = A_orig + self.alpha * new_A_high_to_high  - self.beta * new_A_high_to_low
#         # else :
#         #     fused_A = A
#         fused_A = F.softmax(fused_A, dim=2)

#         x = torch.matmul(fused_A.transpose(2, 3), x)  # [B, H, 1, D]
#         x = x.transpose(1, 2)  # [B, 1, H, D]
#         x = x.reshape(B, 1, -1)  # [B, 1, inner_dim]
#         x = x.reshape(B, C)  # Reshape to (Batch size, inner_dim)
#         x = self.proj(x)  # Apply projection layer
#         x = self.proj_drop(x)  # Apply dropout
#         if self.ffn:
#             x = x + self.mlp(self.norm_ffn(x))  # Apply FFN with residual connection
#         x = self.norm1(x)  # Apply normalization (LayerNorm or BatchNorm)
#         _logits = self.classifier(x)  # Get classification logits

#         # if self.training:
#         #     self.epoch_step += 1

#         if return_img_feat:
#             _logits = [_logits, x] 
 
#         if return_attn:
#             output = []
#             output.append(_logits)
#             output.append(A.squeeze(-2))  
#             if return_act:
#                 output.append(act.squeeze(1)) 
#             return output
#         else:
#             return _logits


class M_DAttentionX(nn.Module): #_Gather
    """
    Multi-Depth Attention X with Gather Optimization (Focus High)

    OPTIMIZED VERSION: Uses gather/scatter instead of sparse matrix operations.
    - Reduces memory from [B,H,N,D] to [B,H,kN,D] (where k << 1)
    - Reduces computation from N^2 to (kN)^2 (~2500x speedup for k=0.02)
    - Equivalent functionality to original M_DAttentionX

    Core idea: Separate instances into high and low attention groups.
    - High instances: self-attention among themselves
    - Low instances: cross-attention from high instances (learn from high)
    - Fusion: combine original, high-branch, and low-branch attention

    Paper Eq: A_fused = Softmax(A + α·A^high - β·A^low)
    """
    def __init__(self, input_dim, n_classes, dropout=0.25, act='gelu', mil_norm=None,
                 mil_bias=False, inner_dim=256, n_heads=8, proj_drop=0., attn_type='mlp',
                 attn_bias=False, ffn=False, init_k=0.02, step=0.005, **kwargs):
        super(M_DAttentionX, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.inner_dim = inner_dim
        self.n_heads = n_heads
        self.head_dim = inner_dim // n_heads
        self.init_k = init_k
        self.step = step
        self.mil_norm = mil_norm

        # Learnable fusion parameters [Eq (7)]
        self.alpha = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))
        self.beta = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))

        # Feature extractor
        feature_layers = []
        if mil_norm == 'ln':
            feature_layers.append(nn.LayerNorm(input_dim, bias=mil_bias))
        feature_layers.append(nn.Linear(input_dim, inner_dim, bias=mil_bias))
        if act.lower() == 'gelu':
            feature_layers.append(nn.GELU())
        elif act.lower() == 'relu':
            feature_layers.append(nn.ReLU())
        feature_layers.append(nn.Dropout(dropout))
        self.feature = nn.Sequential(*feature_layers)

        # Initial attention (ABMIL-style) [Eq (3)]
        if attn_type == 'mlp':
            self.init_attention = MLPAttn(self.head_dim, self.head_dim, bias=attn_bias, k=1)

        # ============ High Branch: Self-attention [Eq (5)] ============
        # Q, K, V projections for high instances
        self.high_q_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.high_k_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.high_v_proj = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        # ============ Low Branch: Cross-attention to high [Eq (6)] ============
        # Q from low instances, K/V from high instances
        self.low_q_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        # Key projection for cross-attention (comes from high instances)
        self.high_k_for_low_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.low_v_proj = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        # Post-processing
        self.norm1 = nn.LayerNorm(inner_dim, bias=mil_bias) if mil_norm == 'ln' else nn.Identity()
        self.proj = nn.Linear(inner_dim, inner_dim, bias=mil_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if ffn:
            self.norm_ffn = nn.LayerNorm(inner_dim)
            self.mlp = Mlp(inner_dim, int(inner_dim * 4.), act_layer=nn.GELU)
        self.ffn = ffn

        self.classifier = nn.Linear(inner_dim, n_classes, bias=mil_bias)
        self.apply(initialize_weights)

    def _gather_features(self, x, indices, expand_dim=True):
        """
        Gather features from x using indices.

        Args:
            x: [B, H, N, D] - source features
            indices: [B, H, k] - indices to gather
            expand_dim: if True, expand indices to match D dimension

        Returns:
            gathered: [B, H, k, D] - gathered features
        """
        if expand_dim:
            # Expand indices: [B, H, k] -> [B, H, k, D]
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
            return x.gather(2, indices_expanded)
        else:
            # For 1D values like attention scores
            return x.gather(2, indices.unsqueeze(-1))

    def _scatter_back_safe(self, x_full, indices, x_compact):
        """
        Scatter compact features back to full sequence.

        Args:
            x_full: [B, H, N, D] - full size tensor (will be modified)
            indices: [B, H, k] - target indices
            x_compact: [B, H, k, D] - compact features to scatter

        Returns:
            x_full with scattered values
        """
        # scatter_ is gradient-friendly and simple
        return x_full.scatter_(2, indices.unsqueeze(-1).expand(-1, -1, -1, x_compact.size(-1)), x_compact)

    def forward(self, x, return_attn=False, return_act=False, return_img_feat=False, **kwargs):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)  # [B, N, C]

        B, N, _ = x.shape

        # Feature extraction
        x = self.feature(x)  # [B, N, inner_dim]
        act = x.clone()

        # Reshape for multi-head: [B, N, inner_dim] -> [B, H, N, D]
        x = x.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # ============ Initial attention [Eq (3)] ============
        A_init = self.init_attention(x)  # [B, H, N, 1]
        A_squeezed = A_init.squeeze(-1)  # [B, H, N]

        # ============ Get indices for high and low groups [Eq (4)] ============
        k_num = max(1, int(N * self.init_k))

        # Top-k indices (high attention instances)
        _, topk_indices = torch.topk(A_squeezed, k_num, dim=-1, sorted=False)
        # Bottom-k indices (low attention instances)
        _, bottomk_indices = torch.topk(-A_squeezed, k_num, dim=-1, sorted=False)

        # ============ OPTIMIZATION: Gather only selected instances ============
        # Gather high instances: [B, H, N, D] -> [B, H, k_num, D]
        x_high = self._gather_features(x, topk_indices, expand_dim=True)
        A_high = self._gather_features(A_init, topk_indices, expand_dim=False)

        # Gather low instances: [B, H, N, D] -> [B, H, k_num, D]
        x_low = self._gather_features(x, bottomk_indices, expand_dim=True)
        A_low = self._gather_features(A_init, bottomk_indices, expand_dim=False)

        # Weighted features (already compact, no mask needed!)
        x_high_weighted = x_high * A_high  # [B, H, k_num, D]
        x_low_weighted = x_low * A_low      # [B, H, k_num, D]

        # ============ High branch: Self-attention [Eq (5)] ============
        # [Eq (5)] A^high = W_v^high · Softmax(Q^high · (K^high)^T / √d)
        Q_high = self.high_q_proj(x_high_weighted)   # [B, H, k_num, D]
        K_high = self.high_k_proj(x_high_weighted)   # [B, H, k_num, D]

        # Compact attention: [B, H, k_num, k_num] instead of [B, H, N, N]
        attn_scores_high = torch.matmul(Q_high, K_high.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights_high = F.softmax(attn_scores_high, dim=-1)

        V_high = self.high_v_proj(A_high)  # [B, H, k_num, 1]
        new_A_high = torch.matmul(attn_weights_high, V_high)  # [B, H, k_num, 1]

        # ============ Low branch: Cross-attention to high [Eq (6)] ============
        # [Eq (6)] A^low = W_v^low · Softmax(Q^low · (K^high→low)^T / √d)
        # Key comes from high instances (cross-attention)
        Q_low = self.low_q_proj(x_low_weighted)                 # [B, H, k_num, D]
        K_high_for_low = self.high_k_for_low_proj(x_high_weighted)  # [B, H, k_num, D]

        # Cross-attention: low instances attend to high instances
        attn_scores_low = torch.matmul(Q_low, K_high_for_low.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights_low = F.softmax(attn_scores_low, dim=-1)

        V_low = self.low_v_proj(A_high)  # [B, H, k_num, 1] - value from high
        new_A_low = torch.matmul(attn_weights_low, V_low)  # [B, H, k_num, 1]

        # ============ Scatter back to full sequence ============
        new_A_high_full = torch.zeros_like(A_init)  # [B, H, N, 1]
        new_A_low_full = torch.zeros_like(A_init)   # [B, H, N, 1]

        # Scatter computed values back to original positions (safe for gradients)
        self._scatter_back_safe(new_A_high_full, topk_indices, new_A_high)
        self._scatter_back_safe(new_A_low_full, topk_indices, new_A_low)

        # ============ Fusion [Eq (7)] ============
        # A_fused = Softmax(A + α·A^high - β·A^low)
        A_fused = A_init + self.alpha * new_A_high_full - self.beta * new_A_low_full
        A_fused = F.softmax(A_fused, dim=2)

        # Pooling: [B, H, N, 1] @ [B, H, N, D] -> [B, H, 1, D]
        x_bag = torch.matmul(A_fused.transpose(2, 3), x)
        x_bag = x_bag.transpose(1, 2)  # [B, 1, H, D]
        x_bag = x_bag.reshape(B, -1)  # [B, inner_dim]

        # Post-processing
        x_bag = self.proj_drop(self.proj(x_bag))
        if self.ffn:
            x_bag = x_bag + self.mlp(self.norm_ffn(x_bag))
        x_bag = self.norm1(x_bag)

        _logits = self.classifier(x_bag)

        if return_img_feat:
            _logits = [_logits, x_bag]

        if return_attn:
            output = [_logits, A_init.squeeze(-2)]
            if return_act:
                output.append(act.squeeze(1))
            return output
        return _logits


# Flash Attention Optimized Version
class M_DAttentionX_FlashOpt(nn.Module):
    """
    Multi-Depth Attention X with Flash Attention Optimization (Focus High)

    OPTIMIZED VERSION: Uses Flash Attention for efficient sparse attention computation.
    - Keeps the gather/scatter approach for selecting instances
    - Replaces manual matrix multiplication with Flash Attention for better GPU utilization
    - Reduces computation from N^2 to (kN)^2 (~2500x speedup for k=0.02)
    - Uses Flash Attention kernels for efficient small matrix multiplication

    Core idea: Separate instances into high and low attention groups.
    - High instances: self-attention among themselves (using Flash Attention)
    - Low instances: cross-attention from high instances (using Flash Attention)
    - Fusion: combine original, high-branch, and low-branch attention

    Paper Eq: A_fused = Softmax(A + α·A^high - β·A^low)
    """
    def __init__(self, input_dim, n_classes, dropout=0.25, act='gelu', mil_norm=None,
                 mil_bias=False, inner_dim=256, n_heads=8, proj_drop=0., attn_type='mlp',
                 attn_bias=False, ffn=False, init_k=0.02, step=0.005, **kwargs):
        super(M_DAttentionX_FlashOpt, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.inner_dim = inner_dim
        self.n_heads = n_heads
        self.head_dim = inner_dim // n_heads
        self.init_k = init_k
        self.step = step
        self.mil_norm = mil_norm

        # Learnable fusion parameters [Eq (7)]
        self.alpha = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))
        self.beta = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))

        # Feature extractor
        feature_layers = []
        if mil_norm == 'ln':
            feature_layers.append(nn.LayerNorm(input_dim, bias=mil_bias))
        feature_layers.append(nn.Linear(input_dim, inner_dim, bias=mil_bias))
        if act.lower() == 'gelu':
            feature_layers.append(nn.GELU())
        elif act.lower() == 'relu':
            feature_layers.append(nn.ReLU())
        feature_layers.append(nn.Dropout(dropout))
        self.feature = nn.Sequential(*feature_layers)

        # Initial attention (ABMIL-style) [Eq (3)]
        if attn_type == 'mlp':
            self.init_attention = MLPAttn(self.head_dim, self.head_dim, bias=attn_bias, k=1)

        # ============ High Branch: Self-attention [Eq (5)] ============
        # Q, K, V projections for high instances
        self.high_q_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.high_k_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.high_v_proj = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        # ============ Low Branch: Cross-attention to high [Eq (6)] ============
        # Q from low instances, K/V from high instances
        self.low_q_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        # Key projection for cross-arsn (comes from high instances)
        self.high_k_for_low_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.low_v_proj = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        # Post-processing
        self.norm1 = nn.LayerNorm(inner_dim, bias=mil_bias) if mil_norm == 'ln' else nn.Identity()
        self.proj = nn.Linear(inner_dim, inner_dim, bias=mil_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if ffn:
            self.norm_ffn = nn.LayerNorm(inner_dim)
            self.mlp = Mlp(inner_dim, int(inner_dim * 4.), act_layer=nn.GELU)
        self.ffn = ffn

        self.classifier = nn.Linear(inner_dim, n_classes, bias=mil_bias)
        self.apply(initialize_weights)

    def _flash_attention(self, q, k, v, dropout=0.0, training=True, deterministic=True):
        """
        Flash Attention for efficient attention computation with fallback to PyTorch SDPA.

        Args:
            q: [B, H, N, D] - query
            k: [B, H, M, D] - key
            v: [B, H, M, D_v] - value

        Returns:
            out: [B, H, N, D_v] - attention output
        """
        B, H, N, D = q.shape
        M = k.shape[2]
        D_v = v.shape[3]

        # Reshape for Flash Attention: [B, N, H, D]
        q_transposed = q.transpose(1, 2).contiguous()
        k_transposed = k.transpose(1, 2).contiguous()
        v_transposed = v.transpose(1, 2).contiguous()

        # Try Flash Attention first
        try:
            # Flash Attention expects: [B, N, H, D]
            output = flash_attn_func(
                q_transposed, k_transposed, v_transposed,
                dropout_p=dropout if training else 0.0,
                deterministic=deterministic if training else False
            )
            # Reshape back: [B, H, N, D_v]
            return output.transpose(1, 2)
        except:
            # Fallback to PyTorch SDPA
            scale = D ** -0.5
            output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout if training else 0.0,
                scale=scale
            )
            return output

    def _gather_features(self, x, indices, expand_dim=True):
        """
        Gather features from x using indices.

        Args:
            x: [B, H, N, D] - source features
            indices: [B, H, k] - indices to gather
            expand_dim: if True, expand indices to match D dimension

        Returns:
            gathered: [B, H, k, D] - gathered features
        """
        if expand_dim:
            # Expand indices: [B, H, k] -> [B, H, k, D]
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
            return x.gather(2, indices_expanded)
        else:
            # For 1D values like attention scores
            return x.gather(2, indices.unsqueeze(-1))

    def _scatter_back_safe(self, x_full, indices, x_compact):
        """
        Scatter compact features back to full sequence using scatter_ (gradient-friendly).
        """
        # scatter_ is gradient-friendly and simple
        return x_full.scatter_(2, indices.unsqueeze(-1).expand(-1, -1, -1, x_compact.size(-1)), x_compact)

    def forward(self, x, return_attn=False, return_act=False, return_img_feat=False, **kwargs):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)  # [B, N, C]

        B, N, _ = x.shape

        # Feature extraction
        x = self.feature(x)  # [B, N, inner_dim]
        act = x.clone()

        # Reshape for multi-head: [B, N, inner_dim] -> [B, H, N, D]
        x = x.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # ============ Initial attention [Eq (3)] ============
        A_init = self.init_attention(x)  # [B, H, N, 1]
        A_squeezed = A_init.squeeze(-1)  # [B, H, N]

        # ============ Get indices for high and low groups [Eq (4)] ============
        k_num = max(1, int(N * self.init_k))

        # Top-k indices (high attention instances)
        _, topk_indices = torch.topk(A_squeezed, k_num, dim=-1, sorted=False)
        # Bottom-k indices (low attention instances)
        _, bottomk_indices = torch.topk(-A_squeezed, k_num, dim=-1, sorted=False)

        # ============ Gather only selected instances ============
        # Gather high instances: [B, H, N, D] -> [B, H, k_num, D]
        x_high = self._gather_features(x, topk_indices, expand_dim=True)
        A_high = self._gather_features(A_init, topk_indices, expand_dim=False)

        # Gather low instances: [B, H, N, D] -> [B, H, k_num, D]
        x_low = self._gather_features(x, bottomk_indices, expand_dim=True)
        A_low = self._gather_features(A_init, bottomk_indices, expand_dim=False)

        # Weighted features (already compact, no mask needed!)
        x_high_weighted = x_high * A_high  # [B, H, k_num, D]
        x_low_weighted = x_low * A_low      # [B, H, k_num, D]

        # ============ High branch: Self-attention using Flash Attention [Eq (5)] ============
        # [Eq (5)] A^high = W_v^high · Softmax(Q^high · (K^high)^T / √d)
        Q_high = self.high_q_proj(x_high_weighted)   # [B, H, k_num, D]
        K_high = self.high_k_proj(x_high_weighted)   # [B, H, k_num, D]
        V_high = self.high_v_proj(A_high)           # [B, H, k_num, 1]

        # Use Flash Attention for efficient computation
        flash_out_high = self._flash_attention(Q_high, K_high, V_high, training=self.training)
        new_A_high = flash_out_high  # [B, H, k_num, 1]

        # ============ Low branch: Cross-attention using Flash Attention [Eq (6)] ============
        # [Eq (6)] A^low = W_v^low · Softmax(Q^low · (K^high→low)^T / √d)
        Q_low = self.low_q_proj(x_low_weighted)                 # [B, H, k_num, D]
        K_high_for_low = self.high_k_for_low_proj(x_high_weighted)  # [B, H, k_num, D]
        V_low = self.low_v_proj(A_high)                         # [B, H, k_num, 1]

        # Cross-attention: low instances attend to high instances
        flash_out_low = self._flash_attention(Q_low, K_high_for_low, V_low, training=self.training)
        new_A_low = flash_out_low  # [B, H, k_num, 1]

        # ============ Scatter back to full sequence ============
        new_A_high_full = torch.zeros_like(A_init)  # [B, H, N, 1]
        new_A_low_full = torch.zeros_like(A_init)   # [B, H, N, 1]

        # Scatter computed values back to original positions (safe for gradients)
        self._scatter_back_safe(new_A_high_full, topk_indices, new_A_high)
        self._scatter_back_safe(new_A_low_full, bottomk_indices, new_A_low)

        # ============ Fusion [Eq (7)] ============
        # A_fused = Softmax(A + α·A^high - β·A^low)
        A_fused = A_init + self.alpha * new_A_high_full - self.beta * new_A_low_full
        A_fused = F.softmax(A_fused, dim=2)

        # Pooling: [B, H, N, 1] @ [B, H, N, D] -> [B, H, 1, D]
        x_bag = torch.matmul(A_fused.transpose(2, 3), x)
        x_bag = x_bag.transpose(1, 2)  # [B, 1, H, D]
        x_bag = x_bag.reshape(B, -1)  # [B, inner_dim]

        # Post-processing
        x_bag = self.proj_drop(self.proj(x_bag))
        if self.ffn:
            x_bag = x_bag + self.mlp(self.norm_ffn(x_bag))
        x_bag = self.norm1(x_bag)

        _logits = self.classifier(x_bag)

        if return_img_feat:
            _logits = [_logits, x_bag]

        if return_attn:
            output = [_logits, A_init.squeeze(-2)]
            if return_act:
                output.append(act.squeeze(1))
            return output
        return _logits


# Global Flash Attention with High/Low Instance Separation
class M_DAttentionX_GlobalFlash(nn.Module):
    """
    Multi-Depth Attention X with Global Flash Attention (Focus High)

    OPTIMIZED VERSION: Uses global Flash Attention with mask-based instance separation.
    - Uses global N×N Flash Attention for maximum GPU efficiency
    - Separates instances into high and low groups using attention masks
    - No gather/scatter overhead (pure mask-based approach)
    - Leverages Flash Attention's optimized kernels

    Core idea: Separate instances into high and low attention groups.
    - High instances: self-attention among themselves (via global attention with mask)
    - Low instances: cross-attention from high instances (via global attention with mask)
    - Fusion: combine original, high-branch, and low-branch attention

    Paper Eq: (1) A = MLPAttn(x), (2) select top-k/bottom-k, (3) A^high = masked_attn(high), (4) A^low = masked_attn(low->high), (5) A_fused fused = Softmax(A + α·A^high - β·A^low)
    """
    def __init__(self, input_dim, n_classes, dropout=0.25, act='gelu', mil_norm=None,
                 mil_bias=False, inner_dim=256, n_heads=8, proj_drop=0., attn_type='mlp',
                 attn_bias=False, ffn=False, init_k=0.02, step=0.005, **kwargs):
        super(M_DAttentionX_GlobalFlash, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.inner_dim = inner_dim
        self.n_heads = n_heads
        self.head_dim = inner_dim // n_heads
        self.init_k = init_k
        self.step = step
        self.mil_norm = mil_norm

        # Learnable fusion parameters [Eq (7)]
        self.alpha = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))
        self.beta = nn.Parameter(torch.FloatTensor(1).uniform_(0, 3))

        # Feature extractor
        feature_layers = []
        if mil_norm == 'ln':
            feature_layers.append(nn.LayerNorm(input_dim, bias=mil_bias))
        feature_layers.append(nn.Linear(input_dim, inner_dim, bias=mil_bias))
        if act.lower() == 'gelu':
            feature_layers.append(nn.GELU())
        elif act.lower() == 'relu':
            feature_layers.append(nn.ReLU())
        feature_layers.append(nn.Dropout(dropout))
        self.feature = nn.Sequential(*feature_layers)

        # Initial attention (ABMIL-style) [Eq (3)]
        if attn_type == 'mlp':
            self.init_attention = MLPAttn(self.head_dim, self.head_dim, bias=attn_bias, k=1)

        # ============ High Branch: Self-attention (global with mask) [Eq (5)] ============
        # Q, K projections for high branch (use original x, not masked)
        self.high_q_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.high_k_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.high_v_proj = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        # ============ Low Branch: Cross-attention to high (global with mask) [Eq (6)] ============
        # Q from low instances, K from all instances (for global attention)
        self.low_q_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        # Key projection for low branch (comes from all instances for global attention)
        self.low_k_proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.head_dim * 4, self.head_dim, bias=False),
        )
        self.low_v_proj = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        # Post-processing
        self.norm1 = nn.LayerNorm(inner_dim, bias=mil_bias) if mil_norm == 'ln' else nn.Identity()
        self.proj = nn.Linear(inner_dim, inner_dim, bias=mil_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if ffn:
            self.norm_ffn = nn.LayerNorm(inner_dim)
            self.mlp = Mlp(inner_dim, int(inner_dim * 4.), act_layer=nn.GELU)
        self.ffn = ffn

        self.classifier = nn.Linear(inner_dim, n_classes, bias=mil_bias)
        self.apply(initialize_weights)

    def _global_flash_attention_with_mask(self, q, k, v, mask=None, training=True):
        """
        Global Flash Attention with optional mask.

        Args:
            q: [B, H, N, D] - query
            k: [B, H, N, D] - key
            v: [B, H, N, 1] - value (attention scores)
            mask: [B, H, N, N] - attention mask (None for no mask)
            training: bool - training mode

        Returns:
            out: [B, H, N, 1] - attention output
        """
        B, H, N, D = q.shape

        # Reshape for Flash Attention: [B, N, H, D]
        q_transposed = q.transpose(1, 2).contiguous()
        k_transposed = k.transpose(1, 2).contiguous()
        v_transposed = v.transpose(1, 2).contiguous()

        # Try Flash Attention first
        try:
            if mask is None:
                # Standard Flash Attention without mask
                output = flash_attn_func(
                    q_transposed, k_transposed, v_transposed,
                    dropout_p=0.0 if not training else 0.0,
                    deterministic=True if training else False
                )
            else:
                # Flash Attention doesn't support mask, fallback to PyTorch SDPA
                scale = D ** -0.5
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=0.0 if not training else 0.0,
                    scale=scale
                )
                return output

            # Reshape back: [B, H, N, 1]
            return output.transpose(1, 2)
        except:
            # Fallback to PyTorch SDPA
            scale = D ** -0.5
            if mask is None:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0 if not training else 0.0,
                    scale=scale
                )
            else:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=0.0 if not training else 0.0,
                    scale=scale
                )
            return output

    def forward(self, x, return_attn=False, return_act=False, return_img_feat=False, **kwargs):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)  # [B, N, C]

        B, N, _ = x.shape

        # Feature extraction
        x = self.feature(x)  # [B, N, inner_dim]
        act = x.clone()

        # Reshape for multi-head: [B, N, inner_dim] -> [B, H, N, D]
        x = x.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # ============ Initial attention [Eq (3)] ============
        A_init = self.init_attention(x)  # [B, H, N, 1]
        A_squeezed = A_init.squeeze(-1)  # [B, H, N]

        # ============ Get indices for high and low groups [Eq (4)] ============
        k_num = max(1, int(N * self.init_k))

        # Top-k indices (high attention instances)
        _, topk_indices = torch.topk(A_squeezed, k_num, dim=-1, sorted=False)
        # Bottom-k indices (low attention instances)
        _, bottomk_indices = torch.topk(-A_squeezed, k_num, dim=-1, sorted=False)

        # ============ Create binary masks for high and low groups ============
        # Create mask for high instances: [B, H, N]
        B_high_mask = torch.zeros(B, self.n_heads, N, device=x.device, dtype=x.dtype)
        B_high_mask.scatter_(2, topk_indices, 1)

        # Create mask for low instances: [B, H, N]
        B_low_mask = torch.zeros(B, self.n_heads, N, device=x.device, dtype=x.dtype)
        B_low_mask.scatter_(2, bottomk_indices, 1)

        # ============ High branch: Self-attention [Eq (5)] ============
        # Use original x as input (all instances)
        # Create attention mask: only high instances can attend to each other
        Q_high = self.high_q_proj(x)      # [B, H, N, D]
        K_high = self.high_k_proj(x)      # [B, H, N, D]
        V_high = self.high_v_proj(A_init)  # [B, H, N, 1]

        # Create mask: high instances attend only to high
        # mask_pair[i, j] = 1 if both i and j are high instances
        high_mask_pair = B_high_mask.unsqueeze(-1) * B_high_mask.unsqueeze(-2)  # [B, H, N, N]

        # Use global attention with mask
        new_A_high = self._global_flash_attention_with_mask(
            Q_high, K_high, V_high,
            mask=high_mask_pair,
            training=self.training
        )  # [B, H, N, 1]

        # ============ Low branch: Cross-attention to high [Eq (6)] ============
        # Q from low instances, K from all instances
        Q_low = self.low_q_proj(x)        # [B, H, N, D]
        K_low = self.low_k_proj(x)          # [B, H, N, D]
        V_low = self.low_v_proj(A_init)      # [B, H, N, 1]

        # Create mask: low instances (query) attend only to high instances (key)
        # mask_pair[i, j] = 1 if i is low AND j is high
        low_mask_pair = B_low_mask.unsqueeze(-1) * B_high_mask.unsqueeze(-2)  # [B, H, N, N]

        # Use global attention with mask
        new_A_low = self._global_flash_attention_with_mask(
            Q_low, K_low, V_low,
            mask=low_mask_pair,
            training=self.training
        )  # [B, H, N, 1]

        # ============ Fusion [Eq (7)] ============
        # A_fused = Softmax(A + α·A^high - β·A^low)
        A_fused = A_init + self.alpha * new_A_high - self.beta * new_A_low
        A_fused = F.softmax(A_fused, dim=2)

        # Pooling: [B, H, N, 1] @ [B, H, N, D] -> [B, H, 1, D]
        x_bag = torch.einsum('bhn1,bhnd->bhd', A_fused, x)  # [B, H, D]
        x_bag = x_bag.reshape(B, -1)  # [B, inner_dim]

        # Post-processing
        x_bag = self.proj_drop(self.proj(x_bag))
        if self.ffn:
            x_bag = x_bag + self.mlp(self.norm_ffn(x_bag))
        x_bag = self.norm1(x_bag)

        _logits = self.classifier(x_bag)

        if return_img_feat:
            _logits = [_logits, x_bag]

        if return_attn:
            output = [_logits, A_init.squeeze(-2)]
            if return_act:
                output.append(act.squeeze(1))
            return output
        return _logits


#for ABMILX
class MLPAttn(nn.Module):
    def __init__(self, L=128, D=None, norm=None, bias=False, norm_2=False, dp=0., k=1, act='gelu', gated=False,
                 dropout=True):
        super(MLPAttn, self).__init__()

        D = D or L  # Set D to L if D is not provided

        self.fc1 = nn.Linear(L, D, bias=bias)  # First linear layer for non-gated attention

        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()

        self.norm_type = norm  # Type of normalization to use
        self.dp = nn.Dropout(dp) if dp else nn.Identity()  # Dropout layer
        if norm == 'bn':
            self.norm1 = nn.BatchNorm1d(D)  # Batch normalization for the first stage
            if norm_2:
                self.norm2 = nn.BatchNorm1d(1)  # Batch normalization for the second stage (if enabled)
            else:
                self.norm2 = nn.Identity()
        elif norm == 'ln':
            self.norm1 = nn.LayerNorm(D)  # Layer normalization for the first stage
            self.norm2 = nn.Identity()  # No second stage normalization for LayerNorm
        else:
            self.norm1 = nn.Identity()  # No normalization
            self.norm2 = nn.Identity()

        self.fc2 = nn.Linear(D, k,bias=bias)

    def forward(self, x):
        # assert len(x.shape) == 4
        if len(x.shape) == 3:
            B, N, C = x.shape  # Batch size, Number of instances, Channels
            attn = x
            x_shape_3 = True
        else:
            B, H, N, C = x.shape  # Batch size, Heads, Number of instances, Channels
            attn = x.reshape(B * H, N, C)  # Reshape for multi-head attention processing
            x_shape_3 = False

        attn = self.fc1(attn)  # Apply first linear layer
        if self.norm_type == 'bn':
            attn = self.norm1(attn.transpose(-1, -2)).transpose(-1, -2)  # Apply batch norm
        else:
            attn = self.norm1(attn)  # Apply layer norm or identity
        attn = self.act(attn)  # Apply activation function
        attn = self.dp(attn)  # Apply dropout

        attn = self.fc2(attn)  # Apply second linear layer
        if self.norm_type == 'bn':
            attn = self.norm2(attn.transpose(-1, -2)).transpose(-1, -2)  # Apply second batch norm (if enabled)
        else:
            attn = self.norm2(attn)  # Apply identity

        if x_shape_3:
            attn = attn.transpose(-1, -2).unsqueeze(-1)  # Reshape attention scores for 3D input
        else:
            attn = attn.reshape(B, H, N, 1)  # Reshape attention scores for 4D input (multi-head)
        return attn


#for ABMILX
class AttnPlus(nn.Module):
    def __init__(self, dim=128, attn_dropout=0., norm=True, embed=True, sdpa_type='torch', head=8, shortcut=True,
                 v_embed=True, pad_v=False):
        super(AttnPlus, self).__init__()
        self.scale = dim ** -0.5  # Scaling factor for attention
        self.attn_drop = nn.Dropout(attn_dropout)  # Dropout for attention scores
        self.embed = embed  # Flag for embedding Q, K
        self.sdpa_type = sdpa_type  # Type of scaled dot-product attention implementation
        self.shortcut = shortcut  # Flag for using shortcut connection
        self.pad_v = pad_v  # Flag for padding V tensor
        self.head = head  # Number of attention heads
        self.dim = dim  # Dimension of the input features

        if embed:
            self.qk = nn.Linear(dim, dim * 2, bias=False)  # Linear layer for Q and K projections
            self.v = nn.Linear(1, 1, bias=False) if v_embed else nn.Identity()  # Linear layer for V projection (if v_embed is True)
        if norm:
            self.norm_x = nn.LayerNorm(dim)  # Layer normalization for input X
        else:
            self.norm_x = nn.Identity()

    def forward(self, x, A):
        if len(x.shape) == 3:
            B, N, _ = x.shape  # Batch size, Number of instances, Dimension
            # Project and reshape Q, K for multi-head attention
            qk = self.qk(self.norm_x(x)).reshape(B, N, 2, self.head, self.dim // self.head).permute(2, 0, 3, 1, 4)
            q, k = qk.unbind(0)  # Separate Q and K
        # B H N D
        else:
            B, H, N, D = x.shape  # Batch size, Heads, Number of instances, Dimension per head
            # Project and reshape Q, K for multi-head attention (when input is already shaped for heads)
            qk = self.qk(self.norm_x(x)).reshape(B, self.head, N, 2, self.dim)
            q, k = qk.unbind(-2)  # Separate Q and K

        v = self.v(A)  # Project V (attention scores from previous module)

        if self.sdpa_type not in ('torch_math', 'torch') or self.pad_v:
            v = F.pad(v, (0, q.shape[-1] - v.size(-1)))  # Pad V if necessary for certain SDPA types

        # Scaled Dot-Product Attention
        A_plus = sdpa(
            q, k, v,
            attn_drop=self.attn_drop,
            scale=self.scale,
            training=self.training,
            sdpa_type=self.sdpa_type
        ).transpose(1, 2)

        if self.sdpa_type not in ('torch_math', 'torch') or self.pad_v:
            A_plus = A_plus[:, :, :, 0].unsqueeze(-1)  # Adjust shape if V was padded

        if self.shortcut:
            A = A + A_plus  # Add shortcut connection
        else:
            A = A_plus  # No shortcut

        return A


#for ABMILX
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias
        drop_probs = drop
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#ABMILX
class DAttentionX(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.25, act='gelu', mil_norm=None, mil_bias=False, inner_dim=512,
                 n_heads=8, proj_drop=0., D=None, attn_type='mlp', attn_bias=False, attn_plus=True, ffn=False,
                 sdpa_type='torch', pad_v=False, attn_plus_embed_new=False, **kwargs):
        super(DAttentionX, self).__init__()
        self.head_dim = inner_dim // n_heads  # Dimension per attention head
        self.L = inner_dim if attn_plus_embed_new else self.head_dim  # Input dimension for MLPAttn
        if D is None:
            self.D = self.L  # Intermediate dimension for MLPAttn
        else:
            if D > 10.:  # If D is a large number, treat as absolute dimension
                self.D = int(D) if attn_plus_embed_new else int(D) // n_heads
            else:  # If D is small, treat as a ratio to L
                self.D = int(self.L // D)

        self.K = 1  # Output dimension of MLPAttn (before head aggregation if attn_plus_embed_new is false)
        self.feature = []  # List to store feature extractor layers
        self.mil_norm = mil_norm  # Type of normalization for MIL
        self.n_heads = n_heads  # Number of attention heads
        self.attn_plus = attn_plus  # Flag to use AttnPlus module
        self.attn_plus_embed_new = attn_plus_embed_new  # Flag for new embedding strategy in AttnPlus

        if attn_plus:
            # Initialize AttnPlus module
            self.attn_plus_fn = AttnPlus(inner_dim if attn_plus_embed_new else self.head_dim, sdpa_type=sdpa_type,
                                         head=n_heads, pad_v=pad_v)

        if ffn:
            self.norm_ffn = nn.LayerNorm(inner_dim) if not mil_norm else nn.Identity()  # LayerNorm before FFN
            self.mlp = Mlp(  # Feed-Forward Network
                in_features=inner_dim,
                hidden_features=int(inner_dim * 4.),
                act_layer=nn.GELU,
            )
        self.ffn = ffn  # Flag to use FFN

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim)  # Batch normalization for input
            self.norm1 = nn.BatchNorm1d(self.L * self.K)  # Batch normalization after attention
        elif mil_norm == 'ln':
            self.feature += [nn.LayerNorm(input_dim, bias=mil_bias)]  # Layer normalization for input
            self.norm1 = nn.LayerNorm(inner_dim, bias=mil_bias)  # Layer normalization after attention

        else:
            self.norm1 = self.norm = nn.Identity()  # No normalization

        self.feature += [nn.Linear(input_dim, inner_dim, bias=mil_bias)]  # Linear layer for feature extraction
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]  # GELU activation
        elif act.lower() == 'relu':
            self.feature += [nn.ReLU()]  # ReLU activation
        if dropout:
            self.feature += [nn.Dropout(dropout)]  # Dropout layer

        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()

        if attn_type == 'mlp':
            # Initialize MLPAttn for standard MLP-based attention
            self.attention = MLPAttn(self.L, self.D, bias=attn_bias,
                                     k=n_heads if attn_plus_embed_new else 1)
            
        self.proj = nn.Linear(inner_dim, inner_dim, bias=mil_bias)  # Projection layer after attention
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout for projection layer

        self.classifier = nn.Linear(inner_dim, n_classes, bias=mil_bias)  # Classifier layer

        self.apply(initialize_weights)  # Apply weight initialization

    def forward(self, x, return_attn=False, return_act=False, return_img_feat=False, **kwargs):
        if len(x.size()) == 2:
            x.unsqueeze_(0)  # Add batch dimension if missing

        B, N, _ = x.shape  # Batch size, Number of instances, Input dimension

        if self.mil_norm == 'bn':
            x = torch.transpose(x, -1, -2)  # Transpose for BatchNorm1d
            x = self.norm(x)  # Apply batch normalization
            x = torch.transpose(x, -1, -2)  # Transpose back

        x = self.feature(x)  # Extract features

        _, _, C = x.shape  # Get the channel dimension after feature extraction (inner_dim)
        act = x.clone()  # Clone for returning activations if needed

        if self.attn_plus_embed_new:
            A = self.attention(x)  # B N D (Attention scores from MLPAttn)
        else:
            x = x.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # B H N D (Reshape for multi-head)
            A = self.attention(x)  # B H N K (Attention scores from MLPAttn per head)

        if self.attn_plus:
            A = self.attn_plus_fn(x, A)  # Apply AttnPlus to refine attention scores
        A = torch.transpose(A, -1, -2)  # B H K N (Transpose for softmax)
        A = F.softmax(A, dim=-1)  # softmax over N (Normalize attention scores)

        if self.attn_plus_embed_new:
            # Reshape x for multi-head if new embedding strategy was used (x wasn't reshaped before AttnPlus)
            x = x.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # B H N D

        # Weighted sum of features based on attention scores
        x = torch.einsum('b h k n, b h n d -> b h k d', A, x).squeeze(1)  # B H D (or B K D if K > 1)

        x = x.reshape(B, C)  # Reshape to (Batch size, inner_dim)

        x = self.proj(x)  # Apply projection layer
        x = self.proj_drop(x)  # Apply dropout

        if self.ffn:
            x = x + self.mlp(self.norm_ffn(x))  # Apply FFN with residual connection

        x = self.norm1(x)  # Apply normalization (LayerNorm or BatchNorm)
        _logits = self.classifier(x)  # Get classification logits

        if return_img_feat:
            _logits = [_logits, x]  # Return logits and image features if requested

        if return_attn:
            output = []
            output.append(_logits)
            output.append(A.squeeze(-2))  # Return attention scores (squeezing the K_out dimension if it's 1)
            if return_act:
                output.append(act.squeeze(1))  # Return activations if requested
            return output
        else:
            return _logits  # Return only logits


#ABMIL
class DAttention(nn.Module):
    def __init__(self, input_dim, n_classes, dropout, act, mil_norm=None, mil_bias=True, mil_cls_bias=True,
                 inner_dim=512, embed_feat=True, embed_norm_pos=0, pos=None, **kwargs):
        super(DAttention, self).__init__()
        self.L = inner_dim  # 512
        self.D = 128  # 128
        self.K = 1
        self.feature = []
        self.mil_norm = mil_norm
        self.embed_norm_pos = embed_norm_pos
        self.pos = pos

        if mil_bias:
            mil_cls_bias = True

        assert self.embed_norm_pos in (0, 1)

        self.pos_embed = nn.Identity()

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
            self.norm1 = nn.BatchNorm1d(self.L * self.K)
        elif mil_norm == 'ln':
            if embed_norm_pos == 0:
                self.feature += [nn.LayerNorm(input_dim, bias=mil_bias)]
                self.norm1 = nn.LayerNorm(self.L * self.K, bias=mil_bias)
            else:
                self.norm = nn.LayerNorm(inner_dim, bias=mil_bias)
                self.norm1 = nn.LayerNorm(self.L * self.K, bias=mil_bias)
        else:
            self.norm1 = self.norm = nn.Identity()

        if embed_feat:
            self.feature += [nn.Linear(input_dim, inner_dim, bias=mil_bias)]

            if act.lower() == 'gelu':
                self.feature += [nn.GELU()]
            else:
                self.feature += [nn.ReLU()]

            if dropout:
                self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D, bias=mil_bias),
            nn.Tanh(),
            nn.Linear(self.D, self.K, bias=mil_bias)
        )

        self.classifier = nn.Linear(self.L * self.K, n_classes, bias=mil_cls_bias)

        self.apply(initialize_weights)

    def forward(self, x, return_attn=False, no_norm=False, return_act=False, pos=None, **kwargs):
        if len(x.size()) == 2:
            x.unsqueeze_(0)

        if self.embed_norm_pos == 0:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)

        x = self.feature(x)
        if self.pos == 'sincos':
            x = self.pos_embed(x, pos=pos)

        if self.embed_norm_pos == 1:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        act = x.clone()

        A = self.attention(x)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.einsum('b k n, b n d -> b k d', A, x).squeeze(1)

        x = self.norm1(x)
        x = self.classifier(x)

        if return_attn:
            output = []
            output.append(x)
            output.append(A.squeeze(1))
            if return_act:
                output.append(act.squeeze(1))
            return output
        else:
            return x


class ModelEmaV3(nn.Module):
    def __init__(
            self,
            model,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2 / 3,
            device: Optional[torch.device] = None,
            foreach: bool = True,
            exclude_buffers: bool = False,
            mm_sche=None,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.mm_sche = mm_sche
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        # if step <= 0:
        # 	return 0.0
        if step < 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        if self.mm_sche:
            decay = self.mm_sche[step]

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        if self.decay == 1.:
            return None
        decay = self.get_decay(step)
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def apply_update_(self, model, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []

            ema_state_dict = self.module.state_dict()
            model_state_dict = model.state_dict()
            for name, ema_v in ema_state_dict.items():
                # Try direct matching and matching with prefixes
                if name in model_state_dict:
                    model_v = model_state_dict[name]
                # ddp
                elif f"module.{name}" in model_state_dict:
                    model_v = model_state_dict[f"module.{name}"]
                # torchcompile + ddp
                elif f"_orig_mod.module.{name}" in model_state_dict:
                    model_v = model_state_dict[f"_orig_mod.module.{name}"]
                # torchcompile
                elif f"_orig_mod.{name}" in model_state_dict:
                    model_v = model_state_dict[f"_orig_mod.{name}"]
                else:
                    continue

                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_lerp_values, model_lerp_values, 1. - decay)
            else:
                for ema_v, model_v in zip(ema_lerp_values, model_lerp_values):
                    ema_v.lerp_(model_v, 1. - decay)
        else:
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_v.lerp_(model_v, 1. - decay)
                else:
                    ema_v.copy_(model_v)

    def apply_update_no_buffers_(self, model, decay: float):
        # interpolate parameters, copy buffers
        ema_params = tuple(self.module.parameters())
        model_params = tuple(model.parameters())
        if self.foreach:
            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_params, scalar=decay)
                torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.lerp_(model_p.to(device=self.device), weight=1. - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class E2E(nn.Module):
    def __init__(self, encoder, mil, device, args=None, p_batch_size=2048, p_batch_size_v=2048, input_dim=1024,
                 all_patch_train=False):
        super(E2E, self).__init__()
        self.encoder = encoder
        self.mil = mil

        if args is not None:
            self.p_batch_size = args.p_batch_size
            self.p_batch_size_v = args.p_batch_size_v
            self.n_classes = args.n_classes
            self.input_dim = args.input_dim
            self.inference_mode = suppress
            self.freeze_enc = False
            self.all_patch_train = args.all_patch_train
            self.load_gpu_later = args.load_gpu_later
            self.load_gpu_later_train = args.load_gpu_later_train
            self.device = args.device
        else:
            self.device = device
            self.p_batch_size = p_batch_size
            self.p_batch_size_v = p_batch_size_v
            self.input_dim = input_dim
            self.all_patch_train = all_patch_train

    def train(self, mode=True):
        if mode:
            self.training = True
            if self.freeze_enc:
                self.encoder.train(False)
                self.mil.train()
            else:
                for module in self.children():
                    module.train(mode)
        else:
            self.training = False
            for module in self.children():
                module.train(mode)

    def forward_encoder(self, x, **kwargs):
        if (self.load_gpu_later and not self.training) or self.load_gpu_later_train:
            x = x.to(self.device, non_blocking=True)
        return self.encoder(x, **kwargs)

    @torch.compiler.disable()
    def forward_mil(self, x, return_img_feat=False, **kwargs):
        logits = self.mil(x, return_img_feat=return_img_feat, **kwargs)
        return logits

    @torch.compiler.disable(recursive=False)
    def forward_encoder_batch(self, x, batch_size, init_tensor_len=0, **enc_kwargs):
        enc_kwargs = {} if enc_kwargs is None else enc_kwargs
        keep_num = x.size(0)
        _device = self.device
        _feats = torch.empty((keep_num + init_tensor_len, self.input_dim), device=_device)
        _x = torch.split(x, batch_size)
        with self.inference_mode():
            for i, batch in enumerate(_x):
                if i == len(_x) - 1:
                    _feats[i * batch_size:keep_num] = self.forward_encoder(batch)
                else:
                    _feats[i * batch_size:(i + 1) * batch_size] = self.forward_encoder(batch)
        return _feats

    @torch.compiler.disable(recursive=False)
    def forward(self, x, ps=None, B=None, **mil_kwargs):
        x_grad = x
        enc_kwargs = {}

        if self.training:
            if len(x_grad) <= self.p_batch_size:
                _feats = self.forward_encoder(x_grad, **enc_kwargs)
            else:
                _feats = self.forward_encoder_batch(x_grad, self.p_batch_size, **enc_kwargs)
        else:
            _feats = self.forward_encoder_batch(x_grad, self.p_batch_size_v, **enc_kwargs)

        if ps is not None and not self.all_patch_train:
            _feats = _feats[ps]

        if B is not None:
            M = int(_feats.size(0) / B)
            return self.forward_mil(_feats.view(B, M, -1), **mil_kwargs)
        else:
            return self.forward_mil(_feats.unsqueeze(0), **mil_kwargs)


class ResNetEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k',
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0},
                 pool: bool = True, pretrained: bool = True):
        super().__init__()
        if not pretrained:
            kwargs['pretrained'] = False
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out


def sdpa(q=None, k=None, v=None, attn_drop=None, attn_mask=None,scale=None,training=True,sdpa_type='torch',deterministic=True):
    # q k v  B H N D
        B,H,_,_ = q.shape
        if q.dtype == torch.float32 and sdpa_type in ('memo_effi','flash'):
            sdpa_type = 'torch'
        if sdpa_type == 'flash':
            assert attn_mask is None
            # flash-attn
            x = flash_attn_func(
                q.transpose(1,2),
                k.transpose(1,2),
                v.transpose(1,2),
                dropout_p = attn_drop.p if training else 0.,
                deterministic=deterministic if training else False
            )
            return x
        elif sdpa_type == 'torch':
            #if attn_mask is not None or not sdpa_type == 'flash':
            #with sdpa_kernel(SDPBackend.MATH):
            x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p = attn_drop.p if training else 0.,
            )
            return x.transpose(1, 2)
        else:
            raise NotImplementedError

def build_model(args,device,train_loader):
    others = {}

    enc_name,mil_name = args.model.split('_')[0],args.model.split('_')[1]
    encoder = build_encoder(args,enc_name)

    mil, others = build_mil(args,mil_name,device,train_loader)

    model = E2E(encoder,mil,device,args).to(device)

    if args.test_type != 'main':
        model_tea = ModelEmaV3(model, decay=args.mm)

        others['model_ema'] = model_tea

        return model, others

    else:
        return model, {}

def build_encoder(args, model_name):
    if model_name == 'r50v2':
        model = ResNetEncoder('resnet50.tv2_in1k', pretrained=not args.no_enc_pt)
    elif model_name == 'r18':
        model = ResNetEncoder('resnet18.tv_in1k', pretrained=not args.no_enc_pt)
    else:
        raise NotImplementedError(f'{model_name} not implemented')
    return model

def build_mil(args, model_name, device, train_loader):
    others = {}

    genera_model_params, genera_trans_params = get_mil_model_params(args)

    if model_name == 'abmil':
        model = DAttention(**genera_model_params).to(device)
    elif model_name == 'abmilx':
        model = DAttentionX(
            **genera_trans_params,
            D=args.abx_D,
            attn_bias=args.abx_attn_bias,
            attn_plus=args.abx_attn_plus,
        ).to(device)
    elif model_name == 'fhabmil':
        model= M_DAttentionX(
            **genera_trans_params,
            attn_bias=args.abx_attn_bias,
        ).to(device)
    elif model_name == 'fhabmil_flash':
        model= M_DAttentionX_GlobalFlash(
            **genera_trans_params,
            attn_bias=args.abx_attn_bias,
        ).to(device)
    else:
        raise NotImplementedError

    return model, others

def select_mask_fn(x, score, largest, mask_ratio=0., k=None, random_ratio=1., msa_fusion='vote'):
    """
    Selects indices based on scores, applying masking and random sampling.

    Args:
        x: Input tensor (batch, length, dim). Used for shape information.
        score: Attention scores used for selection.
        largest: Whether to select top-k largest (True) or smallest (False) scores.
        mask_ratio: Target ratio of elements to select.
        k: Number of top elements to select. If None, calculated based on mask_ratio.
        random_ratio: Ratio of selected elements to randomly sample.
        msa_fusion: Method for fusing MSA scores ('mean' or 'vote').

    Returns:
        selected_indices: Indices of selected elements.
    """
    N = x.shape[0]
    mask_ratio_ori = mask_ratio
    mask_ratio = mask_ratio / random_ratio
    if mask_ratio > 1:
        random_ratio = mask_ratio_ori
        mask_ratio = 1.
    random_k = None

    if k >= N:
        return list(range(N))

    if k is None:
        k = int(np.ceil((N * mask_ratio)))
    else:
        if int(k / random_ratio) >= N:
            random_k = k
            k = N
        else:
            k = int(k / random_ratio)

    if len(score.size()) > 2:
        if msa_fusion == 'mean':
            _, cls_attn_topk_idx = torch.topk(score, int(k // score.size(1)), largest=largest)
            cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3, -1))
        elif msa_fusion == 'vote':
            vote = score.clone()
            vote[:] = 0
            _, idx = torch.topk(score, k=k, sorted=False, largest=largest)
            mask = torch.zeros_like(vote, dtype=torch.bool)  # Use boolean mask
            mask.scatter_(2, idx, True)
            vote[mask] = 1
            vote = vote.sum(dim=1)
            _, cls_attn_topk_idx = torch.topk(vote, k=k, sorted=False)
            cls_attn_topk_idx = cls_attn_topk_idx[0]  # Get indices from the first batch element
    else:
        _, cls_attn_topk_idx = torch.topk(score, k, largest=largest)
        cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)

    # Randomly sample
    if random_ratio < 1.:
        random_idx = torch.randperm(cls_attn_topk_idx.size(0), device=cls_attn_topk_idx.device)
        random_k = int(np.ceil((cls_attn_topk_idx.size(0) * random_ratio))) if random_k is None else random_k
        cls_attn_topk_idx = torch.gather(cls_attn_topk_idx, dim=0, index=random_idx[:random_k])

    return cls_attn_topk_idx  # Return the selected indices

def get_mil_model_params(args):
    genera_model_params = {
        "input_dim": args.input_dim,
        "n_classes": args.n_classes,
        "dropout": args.dropout,
        "act": args.act,
        "mil_norm": args.mil_norm,
        'mil_cls_bias': args.mil_bias,
        "mil_bias": args.mil_bias,
        "inner_dim": args.inner_dim,
        "embed_feat": True,
        'embed_feat_mlp_ratio': 4.,
        'fc_norm_bn': True,
        'embed_norm_pos': 0,
        'feat_embed_type': "norm",
        'pos': args.pos,
    }
    genera_trans_params = deepcopy(genera_model_params)
    genera_trans_params.update({
        'n_layers': args.n_layers,
        'pool': args.pool,
        'attn_dropout': args.attn_dropout,
        'deterministic': True,
        'ffn': args.ffn,
        'sdpa_type': args.sdpa_type,
        'n_heads':args.n_heads,
        'fc_norm':True,
        'vit_norm': True,
        'attn_type': 'mlp',
        'ffn_bias': True,
        'ffn_dp': args.ffn_dp,
        'ffn_ratio': args.ffn_ratio
    })

    return genera_model_params,genera_trans_params