# AI_GUI - Current Implementation State

This document describes the **current, implemented state** of the AI_GUI project.
It is intended for developers and AI-assisted contributors.

---

## 1. Project Status

**Phase:** Functional Beta  
**Backend:** Fully Implemented  
**AI Providers:** Ollama, Gemini, Native Diffusers

The project is a working desktop application with:
- Multi-provider chat (Ollama local, Gemini cloud)
- Native image generation (no ComfyUI dependency)
- AMD GPU support via ZLUDA
- Low VRAM optimizations (8GB capable)

---

## 2. Implemented Features

### Chat System
- **Ollama Provider:** Local LLM integration with model discovery
- **Gemini Provider:** Google AI API integration
- **Streaming:** Real-time token streaming to UI
- **Markdown:** Code syntax highlighting via Pygments

### Image Generation Backend
- **Framework:** Native Diffusers (not ComfyUI)
- **Pipeline:** Singleton pattern with model caching
- **Threading:** QThread-based async generation

#### Supported Model Families
| Family | Detection | Pipeline Class | Status |
|--------|-----------|----------------|--------|
| SD 1.5 | Default fallback | StableDiffusionPipeline | ✅ |
| SD 2.x | `sd2`, `v2-`, `768-v` | StableDiffusionPipeline | ✅ |
| SDXL | `sdxl`, `xl_`, `pony` | StableDiffusionXLPipeline | ✅ |
| Pony | `pony`, `illustrious` | StableDiffusionXLPipeline | ✅ |
| Flux | `flux` | FluxPipeline | ✅ |

#### Model Formats Supported
- `.safetensors` - Standard format
- `.ckpt` - Legacy checkpoint format
- `.gguf` - Quantized format (Flux transformers, T5 encoders)
- `.sft` - Alternate safetensors extension (Flux VAE)

#### Memory Optimizations
- **GGUF Transformers:** 4-bit quantized Flux for 8GB VRAM
- **GGUF T5 Encoder:** Disk-offloaded text encoder
- **CPU Offload:** Automatic for <16GB VRAM systems
- **VAE Warmup:** Kernel pre-compilation
- **Dtype Management:** bfloat16 (GGUF) / float16 (safetensors)

### Custom Components
- **VAE:** Auto-detection of Flux vs SDXL VAE
- **LoRA:** Multi-LoRA stacking with strength controls
- **Text Encoders:** Custom CLIP and T5 GGUF support

### Hardware Support
- **AMD GPUs:** ZLUDA integration (RX 6000/7000 series)
- **NVIDIA GPUs:** Native CUDA support
- **Monitoring:** Real-time VRAM/RAM via psutil

---

## 3. File Structure

```
AI_GUI/
├── main.py                 # Application entry point
├── main_window.py          # Main window controller
├── backend/
│   ├── image_generator.py  # Diffusers pipeline manager
│   └── image_worker.py     # QThread worker for generation
├── providers/
│   ├── ollama_provider.py  # Ollama LLM integration
│   └── gemini_provider.py  # Gemini API integration
├── views/
│   ├── chat_view.py        # Chat interface
│   ├── image_gen_view.py   # Image generation UI
│   ├── gallery_tab.py      # Image gallery browser
│   └── settings_view.py    # Configuration panel
├── widgets/
│   ├── sidebar.py          # Navigation sidebar
│   └── sidebar_panels.py   # Collapsible control panels
├── models/
│   ├── checkpoints/        # Model files
│   ├── loras/              # LoRA files
│   ├── vae/                # VAE files
│   └── text_encoders/      # CLIP/T5 files
└── outputs/
    └── images/             # Generated images
```

---

## 4. Key Technical Details

### Flux GGUF Loading
```python
# Dtype determined by file extension
is_gguf_model = model_id.lower().endswith(".gguf")
flux_dtype = torch.bfloat16 if is_gguf_model else torch.float16

# GGUF uses native diffusers support
from diffusers import GGUFQuantizationConfig
transformer = FluxTransformer2DModel.from_single_file(
    path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
```

### T5 GGUF Loading (Disk Offload)
```python
text_encoder_2 = T5EncoderModel.from_pretrained(
    str(t5_path.parent),
    gguf_file=t5_path.name,
    device_map="auto",
    max_memory={0: 0, "cpu": "4GB"},
    offload_folder=str(temp_dir / "offload_t5"),
    offload_state_dict=True
)
```

### CPU Offload Protection
```python
# T5 is managed by Accelerate, must be protected from Diffusers' offload
temp_te2 = self.pipe.text_encoder_2
self.pipe.text_encoder_2 = None
self.pipe.enable_model_cpu_offload()
self.pipe.text_encoder_2 = temp_te2
```

---

## 5. Development Rules

- **DO NOT** block the Qt main thread - use QThread for all inference
- **DO NOT** mix dtypes - GGUF requires bfloat16 throughout
- **DO** use the singleton pattern for the generator instance
- **DO** clear CUDA cache between model loads
- **DO** handle both GGUF and safetensors code paths

---

## 6. Known Limitations

- Flux safetensors requires 12GB+ VRAM
- ControlNet not yet implemented
- No inpainting/outpainting support yet
- Single image generation only (no batching)

---

## 7. Next Steps

Planned features:
1. ControlNet support for SDXL/Flux
2. Inpainting with mask editor
3. Batch generation queue
4. Prompt templates/presets
5. Image-to-image workflows

---

## 8. Notes for AI Contributors

- This project targets **AMD GPUs** primarily (ZLUDA)
- Low VRAM (8GB) is a key requirement
- GGUF support is critical for Flux accessibility
- Test with both GGUF and safetensors models
- Maintain backwards compatibility with SD 1.5/SDXL
