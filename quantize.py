import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor(x, scale, zero_point, qmin, qmax):
    return (x / scale + zero_point).round().clamp(qmin, qmax)


def dequantize_tensor(q, scale, zero_point):
    return (q - zero_point) * scale


def get_per_channel_params(weight, n_bits):
    qmax = 2 ** (n_bits - 1) - 1
    max_abs = weight.detach().abs().flatten(1).max(dim=1).values
    scale = (max_abs / qmax).clamp(min=1e-8)
    zero_point = torch.zeros_like(scale)
    return scale, zero_point


def get_per_tensor_params(x, n_bits, percentile=99.9):
    qmin, qmax = 0, 2 ** n_bits - 1
    x_flat = x.detach().float().flatten()
    x_min = torch.quantile(x_flat, (100.0 - percentile) / 100.0)
    x_max = torch.quantile(x_flat, percentile / 100.0)
    scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-8)
    zero_point = (qmin - x_min / scale).round().clamp(qmin, qmax)
    return scale, zero_point


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.frozen = False
        self.gptq_applied = False

    def freeze(self):
        scale, zero_point = get_per_channel_params(self.weight, self.weight_bits)
        self.register_buffer('w_scale', scale)
        self.register_buffer('w_zero_point', zero_point)
        self.frozen = True

    def freeze_with_scales(self, scale, zero_point):
        self.register_buffer('w_scale', scale)
        self.register_buffer('w_zero_point', zero_point)
        self.frozen = True
        self.gptq_applied = True

    def forward(self, x):
        if self.frozen:
            qmin = -(2 ** (self.weight_bits - 1))
            qmax = 2 ** (self.weight_bits - 1) - 1
            s = self.w_scale.view([-1] + [1] * (self.weight.dim() - 1))
            z = self.w_zero_point.view([-1] + [1] * (self.weight.dim() - 1))
            if self.gptq_applied:
                w = dequantize_tensor(self.weight, s, z)
            else:
                w = dequantize_tensor(quantize_tensor(self.weight, s, z, qmin, qmax), s, z)
            return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.frozen = False
        self.gptq_applied = False

    def freeze(self):
        qmax = 2 ** (self.weight_bits - 1) - 1
        scale = (self.weight.detach().abs().max() / qmax).clamp(min=1e-8)
        self.register_buffer('w_scale', scale.reshape(1))
        self.register_buffer('w_zero_point', torch.zeros(1, device=self.weight.device))
        self.frozen = True

    def freeze_with_scales(self, scale, zero_point):
        self.register_buffer('w_scale', scale)
        self.register_buffer('w_zero_point', zero_point)
        self.frozen = True
        self.gptq_applied = True

    def forward(self, x):
        if self.frozen:
            qmin = -(2 ** (self.weight_bits - 1))
            qmax = 2 ** (self.weight_bits - 1) - 1
            if self.gptq_applied:
                w = dequantize_tensor(self.weight, self.w_scale, self.w_zero_point)
            else:
                w = dequantize_tensor(quantize_tensor(self.weight, self.w_scale, self.w_zero_point, qmin, qmax), self.w_scale, self.w_zero_point)
            return F.linear(x, w, self.bias)
        return F.linear(x, self.weight, self.bias)


class ActFakeQuant(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.frozen = False
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def calibrate(self, x):
        x_flat = x.detach().float().flatten()
        lo = torch.quantile(x_flat, 0.001)
        hi = torch.quantile(x_flat, 0.999)
        self.running_min = torch.min(self.running_min, lo.to(self.running_min.device))
        self.running_max = torch.max(self.running_max, hi.to(self.running_max.device))

    def freeze(self):
        qmin, qmax = 0, 2 ** self.n_bits - 1
        scale = ((self.running_max - self.running_min) / (qmax - qmin)).clamp(min=1e-8)
        zero_point = (qmin - self.running_min / scale).round().clamp(qmin, qmax)
        self.scale.copy_(scale.reshape(1).to(self.scale.device))
        self.zero_point.copy_(zero_point.reshape(1).to(self.zero_point.device))
        self.frozen = True

    def forward(self, x):
        if self.frozen:
            qmin, qmax = 0, 2 ** self.n_bits - 1
            return dequantize_tensor(quantize_tensor(x, self.scale, self.zero_point, qmin, qmax), self.scale, self.zero_point)
        self.calibrate(x)
        return x


def _conv_to_quant(conv, weight_bits):
    q = QuantConv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                    groups=conv.groups, bias=conv.bias is not None,
                    padding_mode=conv.padding_mode, weight_bits=weight_bits)
    q.weight = nn.Parameter(conv.weight.data.clone())
    if conv.bias is not None:
        q.bias = nn.Parameter(conv.bias.data.clone())
    return q


def _linear_to_quant(linear, weight_bits):
    q = QuantLinear(linear.in_features, linear.out_features,
                    bias=linear.bias is not None, weight_bits=weight_bits)
    q.weight = nn.Parameter(linear.weight.data.clone())
    if linear.bias is not None:
        q.bias = nn.Parameter(linear.bias.data.clone())
    return q


def swap_layers(model, weight_bits=8, act_bits=8):
    first_conv_id = id(model.features[0][0])
    classifier_id = id(model.classifier[1])

    def _swap(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Sequential):
                new_mods = []
                for _, gc in child.named_children():
                    if isinstance(gc, nn.Conv2d):
                        bits = 8 if id(gc) == first_conv_id else weight_bits
                        new_mods.append(_conv_to_quant(gc, bits))
                    elif isinstance(gc, nn.Linear):
                        bits = 8 if id(gc) == classifier_id else weight_bits
                        new_mods.append(_linear_to_quant(gc, bits))
                    elif isinstance(gc, (nn.ReLU6, nn.ReLU)):
                        new_mods.append(gc)
                        new_mods.append(ActFakeQuant(n_bits=act_bits))
                    else:
                        _swap(gc)
                        new_mods.append(gc)
                setattr(parent, name, nn.Sequential(*new_mods))
            elif isinstance(child, nn.Conv2d):
                bits = 8 if id(child) == first_conv_id else weight_bits
                setattr(parent, name, _conv_to_quant(child, bits))
            elif isinstance(child, nn.Linear):
                bits = 8 if id(child) == classifier_id else weight_bits
                setattr(parent, name, _linear_to_quant(child, bits))
            elif isinstance(child, (nn.ReLU6, nn.ReLU)):
                setattr(parent, name, nn.Sequential(child, ActFakeQuant(n_bits=act_bits)))
            else:
                _swap(child)

    _swap(model)
    return model


def calibrate_model(model, loader, device, n_batches=4):
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(inputs.to(device))


def freeze_model(model):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, ActFakeQuant)):
            m.freeze()


def gptq_quantize(model, loader, device, n_batches=2):
    model.eval()
    layers = [
        m for m in model.modules()
        if (isinstance(m, QuantConv2d) and m.groups == 1) or isinstance(m, QuantLinear)
    ]
    for idx, layer in enumerate(layers):
        print(f'GPTQ layer {idx+1}/{len(layers)}: {layer.__class__.__name__} {tuple(layer.weight.shape)}')
        captured = []
        handle = layer.register_forward_hook(lambda m, inp, out, c=captured: c.append(inp[0].detach()))
        with torch.no_grad():
            for i, (inputs, _) in enumerate(loader):
                if i >= n_batches:
                    break
                model(inputs.to(device))
        handle.remove()
        bits = layer.weight_bits
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        if isinstance(layer, QuantConv2d):
            cols = []
            for b in captured:
                unf = F.unfold(b, layer.kernel_size, dilation=layer.dilation,
                               padding=layer.padding, stride=layer.stride)
                B, d, n = unf.shape
                cols.append(unf.reshape(d, B * n))
            X = torch.cat(cols, dim=1).float()
            out_channels = layer.out_channels
            W = layer.weight.data.reshape(out_channels, -1).clone().float().cpu()
        else:
            cols = [b.reshape(-1, layer.in_features).T.float() for b in captured]
            X = torch.cat(cols, dim=1)
            out_channels = layer.out_features
            W = layer.weight.data.clone().float().cpu()
        H = (X @ X.T) / X.shape[1]
        H += (0.01 * H.diag().mean()) * torch.eye(H.shape[0], device=device)
        H_inv = torch.linalg.inv(H).cpu()
        scales = torch.zeros(out_channels)
        for i in range(out_channels):
            if isinstance(layer, QuantConv2d):
                scale_i = get_per_channel_params(W[i].unsqueeze(0), bits)[0][0].item()
            else:
                scale_i = (W[i].abs().max() / qmax).clamp(min=1e-8).item()
            scales[i] = scale_i
            w = W[i].clone()
            for j in range(len(w)):
                q_j = w[j].div(scale_i).round().clamp(qmin, qmax)
                err = (w[j] - q_j * scale_i) / H_inv[j, j].clamp(min=1e-8)
                w[j + 1:] -= err * (H_inv[j, j + 1:] / H_inv[j, j])
                w[j] = q_j * scale_i
            W[i] = w
        W = W.to(device)
        if isinstance(layer, QuantConv2d):
            layer.weight.data.copy_(W.reshape_as(layer.weight.data))
            layer.freeze_with_scales(scales.to(device), torch.zeros_like(scales).to(device))
        else:
            layer.weight.data.copy_(W)
            layer.freeze_with_scales(scales.mean().reshape(1).to(device), torch.zeros(1, device=device))
        del X, H, H_inv, captured
        torch.cuda.empty_cache()
    calibrate_model(model, loader, device, n_batches=4)
    for m in model.modules():
        if (isinstance(m, QuantConv2d) and m.groups > 1) or isinstance(m, ActFakeQuant):
            m.freeze()


def compute_model_size(model, weight_bits, act_bits):
    fp32_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    quant_bits = 0
    overhead_bits = 0
    counted = set()
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            quant_bits += m.weight.numel() * m.weight_bits
            counted.add(id(m.weight))
            if m.bias is not None:
                quant_bits += m.bias.numel() * 32
                counted.add(id(m.bias))
            overhead_bits += 2 * m.out_channels * 32
        elif isinstance(m, QuantLinear):
            quant_bits += m.weight.numel() * m.weight_bits
            counted.add(id(m.weight))
            if m.bias is not None:
                quant_bits += m.bias.numel() * 32
                counted.add(id(m.bias))
            overhead_bits += 2 * 32
        elif isinstance(m, ActFakeQuant):
            overhead_bits += 2 * 32
    for p in model.parameters():
        if id(p) not in counted:
            quant_bits += p.numel() * 32
    quant_weight_mb = quant_bits / (8 * 1024 ** 2)
    overhead_mb = overhead_bits / (8 * 1024 ** 2)
    total_mb = quant_weight_mb + overhead_mb
    weight_ratio = fp32_mb / quant_weight_mb
    act_ratio = 32 / act_bits  # theoretical activation compression ratio
    return {
        'fp32_mb': fp32_mb,
        'quant_weight_mb': quant_weight_mb,
        'overhead_mb': overhead_mb,
        'total_mb': total_mb,
        'weight_ratio': weight_ratio,
        'act_ratio': act_ratio,
    }
