import torch
from torch import nn, einsum
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from openpyxl import Workbook
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 多尺度卷积嵌入
class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=2):
        super(CrossEmbedLayer, self).__init__()
        kernel_size = sorted(kernel_size)
        num_scales = len(kernel_size)
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.conv = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_size, dim_scales):
            self.conv.append(
                nn.Conv2d(in_channels=dim_in, out_channels=dim_scale, kernel_size=kernel,
                          stride=stride, padding=(kernel - stride) // 2)
            )

    def forward(self, x):
        f = tuple(map(lambda conv: conv(x), self.conv))
        return torch.cat(f, dim=1)

# 计算动态位置偏置
def DynamicPositionBias(dim):
    return nn.Sequential(
        nn.Linear(2, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, 1),
        nn.Flatten(start_dim=0)
    )

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return ((x - mean) / (var + self.eps)) * self.g + self.b

# 前馈传播
def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )

class Attention(nn.Module):
    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.):
        super(Attention, self).__init__()
        assert attn_type in {'short', 'long'}, 'attention type 必须是long或者short'
        heads = dim // dim_head
        assert dim >= dim_head, 'dim 必须大于等于 dim_head'
        if heads == 0:
            raise ValueError('heads 不能为零，请确保 dim >= dim_head')
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.attn_type = attn_type
        self.window_size = window_size
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        self.dpb = DynamicPositionBias(dim // 4)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        _, w1, w2 = grid.size()
        grid = grid.view(-1, w1 * w2).permute(1, 0).contiguous()
        rel_pos = grid.view(w1 * w2, 1, 2) - grid.view(1, w1 * w2, 2)
        rel_pos = rel_pos + window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x, return_attention=False):
        b, dim, h, w, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device
        x = self.norm(x)
        if self.attn_type == 'short':
            x = x.view(b, dim, h // wsz, wsz, w // wsz, wsz)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(-1, dim, wsz, wsz)
        elif self.attn_type == 'long':
            x = x.view(b, dim, h // wsz, wsz, w // wsz, wsz)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(-1, dim, wsz, wsz)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda x: x.view(-1, self.heads, wsz * wsz, self.dim_head), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        pos = torch.arange(-wsz, wsz + 1, device=device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        _, size1, size2 = rel_pos.size()
        rel_pos = rel_pos.permute(1, 2, 0).view(size1 * size2, 2)
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]
        sim = sim + rel_pos_bias
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(-1, self.heads * self.dim_head, wsz, wsz)
        out = self.to_out(out)
        if self.attn_type == 'short':
            b, d, h, w = b, dim, h // wsz, w // wsz
            out = out.view(b, h, w, d, wsz, wsz)
            out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
            out = out.view(b, d, h * wsz, w * wsz)
        elif self.attn_type == 'long':
            b, d, l1, l2 = b, dim, h // wsz, w // wsz
            out = out.view(b, l1, l2, d, wsz, wsz)
            out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
            out = out.view(b, d, l1 * wsz, l2 * wsz)
        if return_attention:
            return out, attn
        return out

class Transformer(nn.Module):
    def __init__(self, dim, local_window_size, global_window_size, depth=4, dim_head=32,
                 attn_dropout=0., ff_dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attn_type='short', window_size=local_window_size,
                          dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim=dim, dropout=ff_dropout),
                Attention(dim=dim, attn_type='long', window_size=global_window_size,
                          dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim=dim, dropout=ff_dropout)
            ]))

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x
        return x

# CrossFormer（支持批量坐标跟踪）
class CrossFormer(nn.Module):
    def __init__(self, dim=(64, 128, 256, 512), depth=(2, 2, 8, 2), global_window_size=(8, 4, 2, 1),
                 local_window_size=16, cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
                 cross_embed_strides=(4, 2, 2, 2), num_classes=10, attn_dropout=0., ff_dropout=0.,
                 channels=3, feature_dim=20):
        super(CrossFormer, self).__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)
        last_dim = dim[-1]
        dims = [channels, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride=cel_stride),
                Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz,
                            depth=layers, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ]) for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(
                dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides)
        ])
        self.to_logits = nn.Sequential(nn.Linear(last_dim, num_classes))
        self.feature_reducer = nn.Linear(last_dim, feature_dim)

    def forward(self, x, tracker=None):
        feature_maps = []
        x = x.requires_grad_(True)
        for cel, transformer in self.layers:
            x = cel(x)
            if tracker:
                coords_list = [track_point_in_layer_batch(conv, tracker.get_coords()[-1], x.shape)
                               for conv in cel.conv]
                num_coords = len(coords_list[0])
                num_convs = len(coords_list)
                avg_coords = []
                for i in range(num_coords):
                    avg_x = sum(coords[i][0] for coords in coords_list) // num_convs
                    avg_y = sum(coords[i][1] for coords in coords_list) // num_convs
                    avg_coords.append((avg_x, avg_y))
                tracker.update_coords(avg_coords)
            feature_maps.append(x)
            z = x
            for layer in transformer.layers:
                short_attn, short_ff, long_attn, long_ff = layer
                y, attn = short_attn(z, return_attention=True)
                if tracker:
                    new_coords = track_point_in_attention_batch(attn, tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                y = short_ff(y)
                if tracker:
                    new_coords = track_point_in_layer_batch(short_ff[1], tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                    new_coords = track_point_in_layer_batch(short_ff[4], tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                y, attn = long_attn(y, return_attention=True)
                if tracker:
                    new_coords = track_point_in_attention_batch(attn, tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                y = long_ff(y)
                if tracker:
                    new_coords = track_point_in_layer_batch(long_ff[1], tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                    new_coords = track_point_in_layer_batch(long_ff[4], tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                z = y
            x = transformer(x)
            feature_maps.append(x)
        x = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        return self.to_logits(x), feature_maps

# 批量坐标跟踪器
class BatchTrackPoint:
    def __init__(self, initial_coords_list):
        self.coords = [initial_coords_list]

    def update_coords(self, new_coords_list, ap=True):
        if ap:
            self.coords.append(new_coords_list)

    def get_coords(self):
        return self.coords

# 批量跟踪函数
def track_point_in_layer_batch(layer, coords_batch, input_shape):
    batch_size, channels, height, width = input_shape
    coords_tensor = torch.tensor(coords_batch, device=device)
    x, y = coords_tensor[:, 0], coords_tensor[:, 1]
    padding = cast_tuple(layer.padding, 2)
    kernel_size = cast_tuple(layer.kernel_size, 2)
    stride = cast_tuple(layer.stride, 2)
    new_x = (x + padding[0] - kernel_size[0] // 2) // stride[0]
    new_y = (y + padding[1] - kernel_size[1] // 2) // stride[1]
    return torch.stack([new_x, new_y], dim=1).tolist()

def track_point_in_attention_batch(attention_weights, coords_batch, input_shape):
    batch_size, channels, height, width = input_shape
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    coords_tensor = torch.tensor(coords_batch, device=device)
    x, y = coords_tensor[:, 0], coords_tensor[:, 1]
    y = y.clamp(0, seq_len - 1)
    valid_mask = (y >= 0) & (y < seq_len)
    target_weights = attention_weights[0, :, y.long(), :]
    avg_coords = target_weights.mean(dim=0).argmax(dim=-1)
    new_x = torch.where(valid_mask, avg_coords, x)
    return torch.stack([new_x, y], dim=1).tolist()

# 计算全局重要性掩码
def get_global_importance_mask(feature_maps, logits, target_class):
    probs = F.softmax(logits, dim=-1)
    target_prob = probs[:, target_class].sum()
    global_mask = None
    for feature_map in feature_maps:
        feature_map.retain_grad()
        grad = torch.autograd.grad(target_prob, feature_map, retain_graph=True, create_graph=True)[0]
        mask = torch.sum(grad * feature_map, dim=1, keepdim=True)
        mask = F.interpolate(mask, size=(256, 256), mode='bilinear', align_corners=False)
        if global_mask is None:
            global_mask = mask
        else:
            global_mask += mask
    return global_mask.squeeze().detach().cpu().numpy()

# 批量计算重要性
def find_important_regions_batch(feature_maps, tracker_coords_batch, logits, target_class):
    probs = F.softmax(logits, dim=-1)
    target_prob = probs[:, target_class].sum()
    importance_masks = []
    for feature_map in feature_maps:
        feature_map.retain_grad()
        grad = torch.autograd.grad(target_prob, feature_map, retain_graph=True, create_graph=True)[0]
        importance_mask = torch.sum(grad * feature_map, dim=1, keepdim=True)
        importance_masks.append(importance_mask)
    coords_tensor = torch.tensor(tracker_coords_batch, device=device)
    importance_weights = []
    for i, mask in enumerate(importance_masks):
        x = coords_tensor[:, i, 0].long()
        y = coords_tensor[:, i, 1].long()
        valid_mask = (x >= 0) & (x < mask.shape[3]) & (y >= 0) & (y < mask.shape[2])
        weights = torch.zeros(len(x), device=device)
        weights[valid_mask] = mask[0, 0, y[valid_mask], x[valid_mask]]
        importance_weights.append(weights)
    importance_weights = torch.stack(importance_weights, dim=1)
    max_importance = importance_weights.max(dim=1)[0] + 1e-10
    normalized_importance = importance_weights / max_importance.unsqueeze(1)
    total_importance = normalized_importance.sum(dim=1)
    return importance_masks, total_importance.tolist()

# 模型加载
def load_model():
    model = CrossFormer(
        dim=(32, 64, 128, 256), depth=(2, 2, 2, 2), global_window_size=(8, 4, 2, 1), local_window_size=16,
        cross_embed_kernel_sizes=((2, 4, 6, 8), (2, 4), (2, 4), (2, 4)), cross_embed_strides=(2, 2, 2, 2),
        num_classes=4, attn_dropout=0.2, ff_dropout=0.2, channels=3).to(device)
    state_dict = torch.load(r"D:\6666\CVT\2.pth")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 生成Excel文件
def generate_excel(image_path, exc_path, target_class):
    model = load_model()
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 将图像转换为灰度图并检测大脑轮廓
    img_np = np.array(image.resize((256, 256)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 选择最大的轮廓（假设是大脑区域）
    if contours:
        brain_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.drawContours(mask, [brain_contour], -1, (255), thickness=cv2.FILLED)
    else:
        mask = np.ones((256, 256), dtype=np.uint8) * 255

    # 生成所有初始坐标并筛选出轮廓内的点
    coords = [(i, j) for i in range(25, 226) for j in range(18, 233) if mask[i, j] > 0]
    tracker = BatchTrackPoint(coords)
    output, feature_maps = model(input_tensor, tracker)

    # 计算全局重要性掩码并筛选高重要性区域
    global_mask = get_global_importance_mask(feature_maps, output, target_class)
    threshold = np.percentile(global_mask, 80)
    high_importance_coords = [(i, j) for i, j in coords if global_mask[i, j] > threshold]

    # 批量计算重要性得分
    tracker_coords_batch = []
    coord_to_idx = {coord: idx for idx, coord in enumerate(coords)}
    for coord in high_importance_coords:
        idx = coord_to_idx[coord]
        tracker_coords_batch.append([
            tracker.get_coords()[1][idx], tracker.get_coords()[13][idx],
            tracker.get_coords()[14][idx], tracker.get_coords()[26][idx],
            tracker.get_coords()[27][idx], tracker.get_coords()[39][idx],
            tracker.get_coords()[40][idx], tracker.get_coords()[47][idx]
        ])

    total_importance_batch = []
    if tracker_coords_batch:
        _, total_importance_batch = find_important_regions_batch(feature_maps, tracker_coords_batch,
                                                                 output, target_class)
        scores = np.array(total_importance_batch)
        valid_scores = scores[scores > 0]
    else:
        valid_scores = np.array([])

    # 多种阈值计算方法
    threshold_methods = {}
    best_threshold = 0
    best_method = "no_data"

    if len(valid_scores) > 0:
        try:
            from skimage.filters import threshold_otsu
            if valid_scores.max() > valid_scores.min():
                otsu_thresh = threshold_otsu(valid_scores)
                threshold_methods['otsu_adj'] = otsu_thresh * 0.9
        except (ImportError, Exception) as e:
            print(f"Otsu阈值不可用: {str(e)}")

        median = np.median(valid_scores)
        mad = 1.4826 * np.median(np.abs(valid_scores - median))
        threshold_methods['mad_2sigma'] = median + 2 * mad

        q75 = np.percentile(valid_scores, 75)
        q25 = np.percentile(valid_scores, 25)
        iqr = q75 - q25
        dynamic_percentile = 85 if iqr > (q75 * 0.2) else 92
        threshold_methods[f'percentile_{dynamic_percentile}'] = np.percentile(valid_scores, dynamic_percentile)

        try:
            from sklearn.mixture import GaussianMixture
            if len(valid_scores) >= 10:
                scores_reshaped = valid_scores.reshape(-1, 1)
                gmm = GaussianMixture(n_components=3, tol=1e-4, random_state=0).fit(scores_reshaped)
                means = sorted(gmm.means_.flatten())
                if len(means) >= 2:
                    threshold_methods['gmm_mid'] = np.mean(means[-2:])
        except (ImportError, Exception) as e:
            print(f"GMM不可用: {str(e)}")

        if len(valid_scores) >= 2:
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            threshold_methods['mean_1std'] = mean_score + std_score

        max_quality = -np.inf
        for method_name, thresh in threshold_methods.items():
            selected = valid_scores[valid_scores > thresh]
            if len(selected) == 0:
                continue
            area_ratio = len(selected) / len(valid_scores)
            mean_score = np.mean(selected)
            std_score = np.std(selected)
            quality_score = (
                mean_score * 0.4 +
                (1 - abs(area_ratio - 0.25)) * 0.5 +
                (1 / (std_score + 1e-7)) * 0.1
            )
            if quality_score > max_quality:
                max_quality = quality_score
                best_threshold = thresh
                best_method = method_name

        if best_method == "no_data" or len(valid_scores) == 0:
            fallback_percentile = 80
            best_threshold = np.percentile(valid_scores, fallback_percentile) if len(valid_scores) > 0 else 0
            best_method = f"percentile_fallback_{fallback_percentile}"

        final_selected = valid_scores[valid_scores > best_threshold]
        if len(final_selected) / len(valid_scores) < 0.1:
            best_threshold = np.percentile(valid_scores, 80)
            best_method = "force_80_percentile"

    # 生成Excel文件
    wb = Workbook()
    ws = wb.active
    ws.title = "Tracked Importance"
    headers = ["Coordinate", "Importance", "Threshold Method", "Global Mask Value"]
    ws.append(headers)

    for coord, importance in zip(high_importance_coords, total_importance_batch):
        if importance > best_threshold:
            global_val = global_mask[coord[0], coord[1]]
            ws.append([
                str(coord),
                round(importance, 4),
                best_method,
                round(float(global_val), 4)
            ])

    # 保存Excel文件
    wb.save(exc_path)

# 标注图像
def mark_image(exc_path, image_path, output_image_path):
    df = pd.read_excel(exc_path)
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    input_tensor = preprocess(image)
    image_resized = transforms.ToPILImage()(input_tensor)
    image_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    heatmap = np.zeros((256, 256), dtype=np.float32)
    for coord in df.iloc[:, 0]:
        x, y = map(int, coord.strip('()').split(','))
        if 25 <= x < 226 and 18 <= y < 233:
            heatmap[x, y] = 255
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    transparent_layer = np.zeros_like(image_cv, dtype=np.uint8)
    transparent_layer[heatmap > 0] = heatmap_color[heatmap > 0]
    overlay = cv2.addWeighted(image_cv, 1.0, transparent_layer, 0.6, 0)
    cv2.imwrite(output_image_path, overlay)