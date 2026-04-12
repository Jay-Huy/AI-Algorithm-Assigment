import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time

from safetensors.torch import save_file, load_file
from diffusers import DiffusionPipeline

# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 1: MÁY CHỤP X-QUANG (LOCALIZATION)
# ──────────────────────────────────────────────────────────────────────────────
def find_highly_activated_circuits(pipe, concept_prompt, device, null_prompt="", k=5):
    """
    Dùng forward hooks để tìm ra Top-K layers phản ứng mạnh nhất với concept.
    """
    activation_scores = {}
    hooks = []

    def encode(prompt):
        tokens = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True)
        return pipe.text_encoder(tokens.input_ids.to(device))[0]

    concept_embed = encode(concept_prompt)
    null_embed    = encode(null_prompt)

    def make_hook(name):
        def hook_fn(module, input, output):
            activation_scores[name] = output.detach()
        return hook_fn

    # Gắn hooks vào các lớp cross-attention (attn2)
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Chạy forward pass với timestep trung bình (t=500) để capture activation
    latents = torch.randn(1, 4, 64, 64).to(device)
    t = torch.tensor([500]).to(device)

    pipe.unet(latents, t, encoder_hidden_states=concept_embed)
    concept_acts = {k: v.clone() for k, v in activation_scores.items()}

    pipe.unet(latents, t, encoder_hidden_states=null_embed)
    null_acts = {k: v.clone() for k, v in activation_scores.items()}

    # Tính Delta score để tìm layer phản ứng đặc thù
    delta_scores = {
        name: (concept_acts[name] - null_acts[name]).abs().mean().item()
        for name in concept_acts
    }

    for h in hooks: h.remove()

    sorted_circuits = sorted(delta_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_circuits[:k]]

# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 2: THUẬT TOÁN SPEED (DELTA CALCULATION)
# ──────────────────────────────────────────────────────────────────────────────
def compute_speed_delta(w_old, concept_embed, lamb=1e-6):
    """
    SPEED: SVD-based Null-Space Projection.
    """
    device = w_old.device
    dtype  = w_old.dtype

    # Phân tích SVD để tìm basis của không gian khái niệm
    U, S, Vh = torch.linalg.svd(concept_embed.float(), full_matrices=False)
    
    # Null-space projector: P_perp = I - Vh^T @ Vh
    P_perp = torch.eye(w_old.shape[1], device=device) - Vh.T @ Vh

    # W_new = W_old @ P_perp (Ép output về 0 cho concept mục tiêu)
    W_star = w_old.float() @ P_perp
    
    return (W_star - w_old.float()).to(dtype)

# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 3: HÀM PHẪU THUẬT CHÍNH (OUR ALGORITHM)
# ──────────────────────────────────────────────────────────────────────────────
def OUR_ERASE(pipe, edit_concepts, preserve_concepts, k_layers, save_dir, exp_name, ours_save_path=None):
    start_time = time.time()
    device = pipe.device
    torch_dtype = pipe.unet.dtype

    # 1. Thu thập các module cross-attention
    cross_attn_modules = []
    module_names = []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            cross_attn_modules.append(module)
            module_names.append(name)
    
    original_modules = copy.deepcopy(cross_attn_modules)

    # 2. Xử lý từng concept cần xóa (Superman -> Van Gogh -> Snoopy)
    for erase_concept in edit_concepts:
        print(f"[*] Đang thực hiện vi phẫu cho concept: {erase_concept}")
        
        # MRI: Khoanh vùng circuits
        targeted_circuits = find_highly_activated_circuits(pipe, erase_concept, device, k=k_layers)

        # Chuẩn bị embedding
        tokens = pipe.tokenizer(erase_concept, return_tensors="pt", padding="max_length", truncation=True)
        c_erase_embed = pipe.text_encoder(tokens.input_ids.to(device))[0].squeeze(0) # [77, 768]

        # 3. Tiến hành chỉnh sửa trọng số cục bộ
        for idx, name in enumerate(module_names):
            if name not in targeted_circuits:
                continue
            
            w_old = original_modules[idx].weight.data
            
            # Tính Delta theo SPEED
            delta = compute_speed_delta(w_old, c_erase_embed)
            
            # Cập nhật trực tiếp vào pipe
            new_weight = w_old + delta
            cross_attn_modules[idx].weight = torch.nn.Parameter(new_weight.to(torch_dtype))
            print(f" [✓] Đã tiêm thuốc vào layer: {name}")

    # 4. Lưu kết quả
    ours_state_dict = {name + '.weight': mod.weight for name, mod in zip(module_names, cross_attn_modules)}
    save_path = ours_save_path if ours_save_path else os.path.join(save_dir, f"{exp_name}.safetensors")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file(ours_state_dict, save_path)
    
    print(f"\n[!] Hoàn thành! Model đã được lưu tại {save_path} trong {time.time()-start_time:.2f}s")

# ──────────────────────────────────────────────────────────────────────────────
# BOILERPLATE (DỮ NGUYÊN TỪ UCE ĐỂ TƯƠNG THÍCH)
# ──────────────────────────────────────────────────────────────────────────────
def load_previous_weights(unet, path, torch_dtype, device):
    if path is None or not os.path.exists(path): return
    state_dict = load_file(path)
    for name, module in unet.named_modules():
        key = name + '.weight'
        if key in state_dict:
            module.weight.data.copy_(state_dict[key].to(device=device, dtype=torch_dtype))
    print(f"[*] Đã nạp trọng số cũ từ {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ours: Localized SPEED for Continual Concept Erasure')
    parser.add_argument('--edit_concepts', help='Concepts to erase (sep by ;)', type=str, required=True)
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k_layers', help='Số lượng layer Top-K để vi phẫu', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='ours_models')
    parser.add_argument('--exp_name', type=str, default='ours_continual_test')
    parser.add_argument('--previous_weights_path', type=str, default=None)
    parser.add_argument('--ours_save_path', type=str, default=None)
    
    args = parser.parse_args()
    device = args.device
    torch_dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype, safety_checker=None).to(device)
    
    # Hỗ trợ Continual: Nạp lại vết mổ cũ
    load_previous_weights(pipe.unet, args.previous_weights_path, torch_dtype, device)

    edit_list = [c.strip() for c in args.edit_concepts.split(';')]

    OUR_ERASE(
        pipe,
        edit_list,
        preserve_concepts=[], # SPEED tự xử lý qua Null-space
        k_layers=args.k_layers,
        save_dir=args.save_dir,
        exp_name=args.exp_name,
        ours_save_path=args.ours_save_path
    )