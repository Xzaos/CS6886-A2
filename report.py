from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def build_pdf():
    doc = SimpleDocTemplate(
        'CS6886_Assignment2_ME23B237.pdf',
        pagesize=A4,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    body = styles['Normal']
    body.fontSize = 11
    body.leading = 16

    header = ParagraphStyle('SectionHeader', parent=styles['Normal'], fontSize=14, fontName='Helvetica-Bold', spaceAfter=6)
    title_style = ParagraphStyle('Title', parent=styles['Normal'], fontSize=16, fontName='Helvetica-Bold', alignment=1, spaceAfter=4)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12, alignment=1, spaceAfter=12)

    gap = Spacer(1, 0.2 * inch)

    def table(data, col_widths=None):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return t

    def p(text):
        return Paragraph(text, body)

    story = []

    # Title
    story.append(Paragraph('CS6886 Assignment 2 - MobileNet-v2 Compression', title_style))
    story.append(Paragraph('Ritwik Mishra | ME23B237', subtitle_style))
    story.append(gap)

    # Q1
    story.append(Paragraph('Question 1 - Training Baseline', header))
    story.append(gap)

    story.append(p('1a) CIFAR-10 was loaded with the following transforms. Train: RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]). Test: ToTensor and the same Normalize. No AutoAugment or Cutout was used.'))
    story.append(gap)

    story.append(p('1b) Model and training configuration:'))
    story.append(gap)
    story.append(table([
        ['Parameter', 'Value'],
        ['Width multiplier', '1.0'],
        ['First conv stride', '1 (modified from 2 for 32x32 CIFAR-10 input)'],
        ['Classifier', 'nn.Linear(1280, 10) replacing default 1000-class head'],
        ['Dropout', '0.2 (preserved from pretrained Sequential)'],
        ['Pretrained weights', 'ImageNet IMAGENET1K_V1'],
        ['Optimizer', 'SGD momentum=0.9 nesterov=True wd=4e-5'],
        ['Learning rate', '0.05 with 5-epoch linear warmup'],
        ['Scheduler', 'CosineAnnealingLR T_max=100'],
        ['Loss', 'CrossEntropyLoss label_smoothing=0.1'],
        ['Epochs', '100'],
        ['Batch size', '128'],
        ['Seed', '42'],
    ], col_widths=[2.2 * inch, 4.0 * inch]))
    story.append(gap)

    story.append(p('1c) Final test top-1 accuracy: 93.59%. Loss and accuracy curves are shown in the wandb project cs6886-a2. Training loss decreased steadily from 1.03 at epoch 1 to 0.50 at epoch 100. Train accuracy reached 99.91% while test accuracy plateaued at 93.59%, indicating mild overfitting in later epochs. The primary failure mode was a 6.3% generalization gap attributable to CIFAR-10s small image size (32x32) relative to the ImageNet-pretrained feature distribution. Occasional single-epoch test accuracy dips (e.g. epoch 27: 81.41%) reflect SGD noise with high momentum at this learning rate.'))
    story.append(gap)

    # Q2
    story.append(Paragraph('Question 2 - Model Compression Implementation', header))
    story.append(gap)

    story.append(p('2a) Post-training quantization (PTQ) with GPTQ-style layer-wise weight optimization was implemented from scratch in quantize.py without any quantization libraries.'))
    story.append(gap)
    story.append(p('Weight quantization: Per-channel symmetric uniform quantization for all Conv2d layers. Each output channel gets an independent scale computed as max(|W_i|) / (2^(bits-1) - 1) with zero-point fixed at zero. Per-tensor symmetric quantization for the Linear classifier.'))
    story.append(gap)
    story.append(p('Activation quantization: Per-tensor asymmetric uniform quantization inserted after every ReLU6 via ActFakeQuant modules. Scale and zero-point computed from percentile-clipped (99.9th percentile) running min/max over 4 calibration batches to reduce outlier sensitivity.'))
    story.append(gap)
    story.append(p('GPTQ: For each pointwise Conv2d (groups=1) and the Linear layer, optimal quantized weights are found by minimizing the layer-wise output error ||WX - W_q X|| using the inverse Hessian of the input activations. Error is propagated sequentially across weight elements within each output channel. Depthwise convs (groups>1) use standard PTQ.'))
    story.append(gap)

    story.append(p('2b) Layer quantization summary:'))
    story.append(gap)
    story.append(table([
        ['Layer', 'Quantized', 'Weight Bits', 'Notes'],
        ['First Conv2d (features[0][0])', 'Yes', '8', 'Pinned at 8-bit'],
        ['Pointwise Conv2d (groups=1)', 'Yes', 'Configurable', 'GPTQ applied'],
        ['Depthwise Conv2d (groups>1)', 'Yes', 'Configurable', 'Standard PTQ per-channel'],
        ['BatchNorm2d', 'No', '32 (FP32)', 'Kept in FP32'],
        ['ReLU6 outputs', 'Yes', 'Configurable', 'ActFakeQuant inserted after'],
        ['Classifier Linear', 'Yes', '8', 'Pinned at 8-bit'],
    ], col_widths=[2.0 * inch, 0.9 * inch, 1.0 * inch, 2.3 * inch]))
    story.append(gap)

    story.append(p('2c) Storage overheads are accounted for as follows: conv weights at N*bits/8 bytes, biases at N*4 bytes (FP32), per-channel scales at out_channels*4 bytes per layer (FP32), per-channel zero-points at out_channels*4 bytes per layer (FP32), BN parameters at N*4 bytes (FP32), ActFakeQuant scale and zero-point at 2*4 bytes per activation site. At 4-bit, per-channel scales add approximately 0.13 MB overhead across all layers.'))
    story.append(gap)

    # Q3
    story.append(Paragraph('Question 3 - Compression Results', header))
    story.append(gap)

    story.append(p('A sweep over weight_bits in {8,6,4,3,2} and act_bits in {8,6,4} was run with GPTQ enabled, yielding 15 configurations logged to wandb project cs6886-a2. The parallel coordinates chart from wandb is included below.'))
    story.append(gap)
    story.append(p('[SEE ATTACHED WANDB PARALLEL COORDINATES SCREENSHOT]'))
    story.append(gap)

    story.append(table([
        ['weight_bits', 'act_bits', 'quant_acc (%)', 'weight_ratio', 'total_mb'],
        ['8', '8', '93.31', '3.82', '2.36'],
        ['8', '6', '93.29', '3.82', '2.36'],
        ['8', '4', '90.66', '3.82', '2.36'],
        ['6', '8', '93.07', '4.99', '1.84'],
        ['6', '6', '93.22', '4.99', '1.84'],
        ['6', '4', '90.30', '4.99', '1.84'],
        ['4', '8', '87.15', '7.19', '1.32'],
        ['4', '6', '87.21', '7.19', '1.32'],
        ['4', '4', '82.88', '7.19', '1.32'],
        ['3', '8', '52.35', '9.21', '1.06'],
        ['3', '6', '52.51', '9.21', '1.06'],
        ['3', '4', '51.54', '9.21', '1.06'],
        ['2', '8', '10.00', '12.83', '0.80'],
        ['2', '6', '10.01', '12.83', '0.80'],
        ['2', '4', '9.99', '12.83', '0.80'],
    ], col_widths=[1.1 * inch, 0.9 * inch, 1.3 * inch, 1.2 * inch, 1.0 * inch]))
    story.append(gap)

    # Q4
    story.append(Paragraph('Question 4 - Compression Analysis', header))
    story.append(gap)

    story.append(p('Selected configuration: W4A8 (4-bit weights, 8-bit activations) with GPTQ.'))
    story.append(gap)
    story.append(p('4a) Weight compression ratio: 7.19x. FP32 model weights occupy 8.53 MB. At 4-bit, quantized weights occupy 1.19 MB. Per-channel scales and zero-points add 0.13 MB overhead, giving a total compressed size of 1.32 MB.'))
    story.append(gap)
    story.append(p('4b) Activation compression ratio: 4.00x (theoretical: 32/8). Activations were measured as the sum of all output tensor sizes over one forward pass with batch=1, comparing FP32 (4 bytes per element) against 8-bit quantized (1 byte per element). The 4.00x ratio reflects the quantization of all ReLU6 outputs via ActFakeQuant modules.'))
    story.append(gap)
    story.append(p('4c) Quantized model accuracy: 87.15% at W4A8. This represents a 6.44% drop from the FP32 baseline of 93.59%. The drop is primarily attributable to depthwise conv layers which use standard PTQ without GPTQ error correction, as their small weight tensors (9 parameters per channel) make Hessian-based optimization less effective.'))
    story.append(gap)
    story.append(p('4d) Final approximated model size: 1.32 MB. This includes quantized weights (1.19 MB at 4-bit), per-channel scales and zero-points (0.13 MB FP32), biases (FP32), and BN parameters (FP32). The FP32 baseline is 8.53 MB.'))
    story.append(gap)

    # Q5
    story.append(Paragraph('Question 5 - Reproducibility', header))
    story.append(gap)

    story.append(p('GitHub repository: https://github.com/Xzaos/CS6886-A2'))
    story.append(gap)

    story.append(table([
        ['Command', 'Description'],
        ['python train.py', 'Train MobileNet-v2 for 100 epochs on CIFAR-10'],
        ['python test.py --weight_bits 4 --act_bits 8 --use_gptq', 'Run W4A8 GPTQ quantization'],
        ['python sweep.py', 'Run full 15-configuration sweep'],
    ], col_widths=[3.0 * inch, 3.2 * inch]))
    story.append(gap)

    story.append(p('Environment: Python 3.11, PyTorch 2.x, torchvision, wandb 0.28.1. Random seed: 42 (torch.manual_seed and torch.cuda.manual_seed set at the start of every script). CIFAR-10 downloads automatically on first run.'))

    doc.build(story)
    print('Saved CS6886_Assignment2_ME23B237.pdf')

if __name__ == '__main__':
    build_pdf()
