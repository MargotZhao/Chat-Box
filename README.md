# Fine-Tuning and Comparison of Language Models with LoRA

---

## 1. Problem Statement & Overview
This project implements **parameter-efficient** fine-tuning of language models with **Low-Rank Adaptation (LoRA)** and 8-bit quantization.  
Two models—GPT-2 (124 M) and TinyLlama-1.1 B—are compared on response quality, latency, and resource needs inside an interactive **Streamlit** dashboard.

**Key Features**

* Fine-tuning with LoRA for maximum parameter efficiency  
* Side-by-side analysis of models with different sizes  
* Real-time generation & evaluation in Streamlit  
* Latency / response benchmarking  
* Training pipeline that fits on consumer GPUs

---

## 2. Methodology

### Theoretical Foundation
* **Parameter-Efficient Fine-Tuning (PEFT)** – update only a small subset of weights  
* **LoRA** – inserts low-rank adapters instead of full-rank updates  
* **8-bit Quantization** – cuts memory use during training & inference  
* **Transfer Learning** – leverages pre-trained knowledge  
* **Comparative Analysis** – 124 M vs 1.1 B parameters

### LoRA Implementation
LoRA decomposes an original weight matrix \(W_0\) into frozen weights plus a learned low-rank update:

\[
W = W_0 + \Delta W = W_0 + BA
\]

* \(B\) is \(d \times r\)  * \(A\) is \(r \times k\)  * \(r \ll \min(d,k)\)

Trainable params drop from \(d \times k\) to \(r(d+k)\), often **≈ 99 % fewer**.

---

## 3. Implementation & Demo

### Core Components
1. **Training Pipeline**  
   * JSONL → conversational pairs  
   * 8-bit loading  
   * LoRA adapter training  
   * Checkpointing & eval  
2. **Model Architectures**  
   * **GPT-2 124 M** + LoRA  
   * **TinyLlama 1.1 B** + LoRA  
3. **Interactive Dashboard**  
   * Prompt box, dual outputs  
   * Latency & length metrics  
   * Validation-set toggle

### Usage Example (Python)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model = PeftModel.from_pretrained(base_model, "./gpt2-lora/final")

    prompt = "User: I've been feeling anxious lately.\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

---

## 4. Assessment & Evaluation

### Latency

| Model      | Avg (s) | Median (s) |
|------------|---------|------------|
| GPT-2      | 0.231   | 0.224      |
| TinyLlama  | 0.384   | 0.376      |

### Response Characteristics

| Model      | Avg Len (words) | Coherence | Relevance |
|------------|-----------------|-----------|-----------|
| GPT-2      | 42.3            | 3.7 / 5   | 3.8 / 5   |
| TinyLlama  | 58.1            | 4.1 / 5   | 4.2 / 5   |

**Key Findings**

* TinyLlama is **~1.7× slower** yet produces longer, more nuanced replies.  
* GPT-2 offers **snappier** answers suitable for lightweight tasks.  
* LoRA slashes trainable parameters by **~99 %** with minimal quality loss.  

---

## 5. Model & Data Cards

### GPT-2 LoRA
| Field                | Value                                   |
|----------------------|-----------------------------------------|
| Model Type           | Decoder-only Transformer               |
| Size                 | 124 M parameters                       |
| Base                 | `openai-community/gpt2`                |
| Fine-Tuning          | LoRA, 8-bit                            |
| Intended Use         | Conversational AI, text generation     |
| Limitations          | May hallucinate / be incoherent        |
| License              | MIT                                    |

### TinyLlama LoRA
| Field                | Value                                   |
|----------------------|-----------------------------------------|
| Model Type           | Decoder-only Transformer               |
| Size                 | 1.1 B parameters                       |
| Base                 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`   |
| Fine-Tuning          | LoRA, 8-bit                            |
| Intended Use         | Conversational AI, text generation     |
| Limitations          | Same as above                          |
| License              | Apache 2.0                             |

**Ethical Considerations**

* Possible biased / offensive output  
* Not a substitute for professional advice  
* Must filter unsafe content  
* Handle user data per privacy law

---

## 6. Critical Analysis

* **Accessibility** – LoRA lets hobbyists fine-tune billion-parameter models.  
* **Trade-offs** – Quality scales with size but incurs diminishing returns.  
* **Future Work**  
  * Tune LoRA ranks & target layers  
  * Compare with QLoRA / adapters / prefix-tuning  
  * Build domain-specific eval suites  
  * Explore MoE routing or knowledge distillation

---
## 7. Challenges

**Hardware Limitations**

* I tested several models including DeepSeek-R1-Qwen, Google Gemma-2, and Microsoft BitNet, distilbert/distilbert-base-cased-distilled-squad, google-bert/bert-large-uncased-whole-word-masking-finetuned-squad but most ran into problems
* Even with very small datasets, training took too long on my hardware
* Many models required more CPU and GPU resources than available

**Conversational vs. QnA Models**
**Conversational Models:**

* Remember previous messages in a conversation
* Can create natural, flowing dialogue
* Work well for chatbots and interactive applications
* Can handle open-ended questions

**QnA Models:**

* Need specific context with each question
* Often just repeated my input when used in the chatbox
* Better for answering factual questions, not conversations
* Work well with documents but struggle with casual chat

* These differences proved important when selecting the right model type for this project. Despite higher resource needs, conversational models were necessary for the interactive dashboard experience.

## 8. Resource Links

* https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
* https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad
* https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf

## 9. Citation
If you use this project in your research or application, please cite:
@software{lora_model_comparison,
  author = {Your Name},
  title = {Fine-Tuning and Comparison of Language Models with LoRA},
  year = {2025},
  url = {https://github.com/username/lora-model-comparison}
}

## 10. License
This project is licensed under the MIT License - see the LICENSE file for details.
