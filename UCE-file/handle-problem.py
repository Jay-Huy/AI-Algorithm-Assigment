import torch
torch.set_grad_enabled(False)
import argparse
import os
import time

from safetensors.torch import save_file, load_file
from diffusers import DiffusionPipeline


def encode_concept_vectors(pipe, prompt_list, device):
    """
    Encode prompts into concept vectors using the last non-special token embedding,
    similar to the UCE token selection behavior.
    Returns tensor of shape [N, hidden_dim].
    """
    if not prompt_list:
        return None

    tokenized = pipe.tokenizer(
        prompt_list,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    hidden = pipe.text_encoder(input_ids)[0]

    vectors = []
    for i in range(hidden.shape[0]):
        last_token_idx = int(attention_mask[i].sum().item()) - 2
        last_token_idx = max(0, min(last_token_idx, hidden.shape[1] - 1))
        vectors.append(hidden[i, last_token_idx, :])
    return torch.stack(vectors, dim=0)

# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 1: MÁY MRI CHỤP ĐA TIMESTEP (STABLE LOCALIZATION)
# ──────────────────────────────────────────────────────────────────────────────
def find_highly_activated_circuits(pipe, concept_prompt, device, null_prompt="", k=5):
    """
    Tìm Top-K layers phản ứng mạnh nhất bằng cách trung bình hóa qua nhiều timesteps.
    """
    activation_scores = {name: 0.0 for name, mod in pipe.unet.named_modules() 
                         if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k'))}
    hooks = []

    def encode(prompt):
        tokens = pipe.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
        )
        return pipe.text_encoder(tokens.input_ids.to(device))[0]

    concept_embed = encode(concept_prompt)
    null_embed    = encode(null_prompt)

    # Dictionary tạm để lưu activation của 1 forward pass
    current_acts = {}
    def make_hook(name):
        def hook_fn(module, input, output):
            current_acts[name] = output.detach()
        return hook_fn

    for name, module in pipe.unet.named_modules():
        if name in activation_scores:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Quét ở 3 giai đoạn của quá trình khuếch tán (Early, Mid, Late)
    timesteps_to_test = [800, 500, 200]
    latents = torch.randn(1, 4, 64, 64).to(device) # Cùng 1 seed cho công bằng

    for t_val in timesteps_to_test:
        t = torch.tensor([t_val]).to(device)
        
        # Pass 1: Concept
        pipe.unet(latents, t, encoder_hidden_states=concept_embed)
        concept_acts = {k: v.clone() for k, v in current_acts.items()}
        
        # Pass 2: Null (Baseline)
        pipe.unet(latents, t, encoder_hidden_states=null_embed)
        null_acts = {k: v.clone() for k, v in current_acts.items()}

        # Tích lũy Delta score
        for name in activation_scores:
            if name not in concept_acts or name not in null_acts:
                continue
            delta = (concept_acts[name] - null_acts[name]).abs().mean().item()
            activation_scores[name] += delta

    for h in hooks: h.remove()

    # Tính trung bình và sort
    for name in activation_scores:
        activation_scores[name] /= len(timesteps_to_test)

    sorted_circuits = sorted(activation_scores.items(), key=lambda x: x[1], reverse=True)
    targeted = [name for name, _ in sorted_circuits[:k]]
    
    print(f"   -> Top {k} localized layers: {targeted}")
    return targeted

# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 2: THUẬT TOÁN SPEED CÓ TÍCH HỢP PRESERVE CORRECTION
# ──────────────────────────────────────────────────────────────────────────────
def compute_speed_delta(w_old, c_erase, c_preserve=None, lamb=1e-6):
    device = w_old.device
    dtype  = w_old.dtype

    # 1. Null-space Projection (Xóa mục tiêu)
    U, S, Vh = torch.linalg.svd(c_erase.float(), full_matrices=False)
    P_perp = torch.eye(w_old.shape[1], device=device) - Vh.T @ Vh
    W_star = w_old.float() @ P_perp

    # 2. Closed-form Correction (Bảo vệ mục tiêu, giải quyết Nhánh A của Review)
    if c_preserve is not None and c_preserve.shape[0] > 0:
        Cp = c_preserve.float() # shape: [num_preserve_concepts, dim]
        
        # Ràng buộc: W_star @ Cp^T phải xấp xỉ W_old @ Cp^T
        target_preserve = w_old.float() @ Cp.T
        current_preserve = W_star @ Cp.T
        
        # Phương trình có lamb để tránh singular
        CpCp = Cp @ Cp.T + lamb * torch.eye(Cp.shape[0], device=device)
        
        correction = (target_preserve - current_preserve) @ torch.linalg.inv(CpCp) @ Cp
        W_star = W_star + correction

    return (W_star - w_old.float()).to(dtype)

# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 3: VÒNG LẶP CUMULATIVE ERASURE CHÍNH
# ──────────────────────────────────────────────────────────────────────────────
def OUR_ERASE(pipe, edit_concepts, preserve_concepts, k_layers, lamb, save_dir, exp_name, ours_save_path=None):
    start_time = time.time()
    device = pipe.device
    torch_dtype = pipe.unet.dtype

    cross_attn_modules = []
    module_names = []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            cross_attn_modules.append(module)
            module_names.append(name)

    # Encode tập preserve_concepts (làm 1 lần)
    c_preserve_embeds = encode_concept_vectors(pipe, preserve_concepts, device)

    # VÒNG LẶP CUMULATIVE (Giải quyết lỗi Ghi đè)
    for erase_concept in edit_concepts:
        print(f"\n[*] Đang thực hiện vi phẫu cho: '{erase_concept}'")
        
        targeted_circuits = find_highly_activated_circuits(pipe, erase_concept, device, k=k_layers)
        c_erase_embed = encode_concept_vectors(pipe, [erase_concept], device) # shape [1, hidden_dim]

        layers_edited = 0
        for idx, name in enumerate(module_names):
            if name not in targeted_circuits:
                continue
            
            # LỖI CŨ ĐÃ SỬA: Lấy live weights hiện tại, không dùng deepcopy ban đầu!
            w_live = cross_attn_modules[idx].weight.data
            
            # Validation Shape (Giải quyết Mục 8 của Review)
            if c_erase_embed.shape[1] != w_live.shape[1]:
                print(f" [!] Bỏ qua {name} do lệch Dimension ({c_erase_embed.shape[1]} vs {w_live.shape[1]})")
                continue

            delta = compute_speed_delta(w_live, c_erase_embed, c_preserve_embeds, lamb)
            
            # Cập nhật trực tiếp trọng số sống
            new_weight = w_live + delta
            cross_attn_modules[idx].weight.data.copy_(new_weight.to(torch_dtype))
            layers_edited += 1
            
        print(f" [✓] Đã tích lũy update thành công trên {layers_edited} layers.")

    # Lưu model an toàn (Giải quyết Mục 6 của Review)
    ours_state_dict = {name + '.weight': mod.weight for name, mod in zip(module_names, cross_attn_modules)}
    save_path = ours_save_path if ours_save_path else os.path.join(save_dir, f"{exp_name}.safetensors")
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    save_file(ours_state_dict, save_path)
    
    print(f"\n[!] Hoàn thành Continual Editing! Model lưu tại: {save_path} ({time.time()-start_time:.2f}s)")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
def load_previous_weights(unet, path, torch_dtype, device):
    if not path or not os.path.exists(path): return
    state_dict = load_file(path)
    loaded = 0
    for name, module in unet.named_modules():
        if 'attn2' not in name or not (name.endswith('to_v') or name.endswith('to_k')):
            continue
        key = name + '.weight'
        if key in state_dict and hasattr(module, 'weight') and module.weight is not None:
            module.weight.data.copy_(state_dict[key].to(device=device, dtype=torch_dtype))
            loaded += 1
    print(f"[*] Đã nạp {loaded} trọng số cũ từ {path} (Đảm bảo Continual Pipeline).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ours: Localized SPEED for Continual Concept Erasure')
    parser.add_argument('--edit_concepts', help='Concepts to erase (sep by ;)', type=str, required=True)
    parser.add_argument('--preserve_concepts', help='Concepts to protect (sep by ;)', type=str, default="")
    parser.add_argument('--lamb', help='Regularization cho preserve term', type=float, default=1e-6)
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k_layers', help='Số lượng layer Top-K để vi phẫu', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='ours_models')
    parser.add_argument('--exp_name', type=str, default='ours_continual_test')
    parser.add_argument('--previous_weights_path', type=str, default=None)
    parser.add_argument('--previous_ours_path', type=str, default=None)
    parser.add_argument('--ours_save_path', type=str, default=None)
    
    args = parser.parse_args()
    device = args.device
    torch_dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype, safety_checker=None).to(device)
    
    # Precedence: Nạp checkpoint cũ trước, sau đó mới edit mới tích lũy lên.
    previous_path = args.previous_ours_path or args.previous_weights_path
    load_previous_weights(pipe.unet, previous_path, torch_dtype, device)

    edit_list = [c.strip() for c in args.edit_concepts.split(';')]
    preserve_list = [c.strip() for c in args.preserve_concepts.split(';')] if args.preserve_concepts else []

    OUR_ERASE(
        pipe,
        edit_list,
        preserve_list,
        args.k_layers,
        args.lamb,
        args.save_dir,
        args.exp_name,
        args.ours_save_path,
    )