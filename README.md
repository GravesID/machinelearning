# Analisis Sentimen Review Umamusume

## Ringkasan
Proyek ini melakukan **analisis sentimen** pada review game *Umamusume*. Dataset memiliki dua kolom:

- `content`: teks review.
- `auto_sentiment`: label sentimen (`positive`, `neutral`, `negative`, `others`).

Workflow mencakup model ML tradisional, model deep learning, XAI (LIME & SHAP), dan fine-tuning efisien parameter menggunakan LoRA.

---

## 1. Persiapan Dataset
- Load dataset dari `umamusume-reviews.xlsx`.
- Buang missing value pada kolom `content` dan `auto_sentiment`.
- Encode label sentimen menggunakan `LabelEncoder`.
- Bagi data menjadi **train** dan **test** (80/20).
- Tokenisasi teks:
  - **TF-IDF** untuk Naive Bayes.
  - **Tokenisasi & padding** untuk LSTM dan BERT.

---

## 2. Model

### 2.1 Naive Bayes (TF-IDF)
- Menggunakan `sklearn` `MultinomialNB`.
- Input: TF-IDF (`max_features=1000`, stop words dihapus).
- Evaluasi pada test set untuk klasifikasi sentimen dasar.

### 2.2 LSTM (Deep Learning)
- Arsitektur: Embedding → LSTM → Dense output.
- Loss: `sparse_categorical_crossentropy`.
- Optimizer: Adam.
- Tokenisasi dan padding diterapkan.

### 2.3 BERT (Transformer)
- Menggunakan `bert-base-uncased` dengan `TFBertForSequenceClassification`.
- Tokenisasi dengan max length 100.
- Bisa difine-tune dengan LoRA untuk adaptasi parameter efisien.

---

## 3. Explainable AI (XAI)

### 3.1 LIME
- Memberikan penjelasan lokal per review.
- Menyoroti **kata-kata yang berkontribusi** pada prediksi sentimen.
- Output: bar plot kontribusi kata (merah=positif, biru=negatif).

### 3.2 SHAP
- Kernel SHAP untuk Naive Bayes.
- Deep SHAP untuk LSTM.
- PartitionExplainer untuk BERT.
- Menghasilkan **nilai SHAP per token/fitur**, divisualisasikan dengan bar plot.

> Catatan:
> - SHAP untuk Naive Bayes butuh konversi sparse → dense.
> - LSTM/BERT memori intensif untuk sequence panjang.

---

## 4. Fine-Tuning Parameter-Efisien (LoRA)
- **LoRA** memungkinkan fine-tuning model besar dengan **sedikit parameter**.
- Target modul: `query` dan `value` di layer attention.
- Matriks low-rank (`r=8`) diupdate, bukan seluruh model.
- Mengurangi penggunaan memori dan mempercepat training.
- LoRA sendiri tidak ada visualisasi; interpretasi bisa pakai **LIME/SHAP** atau plot metrik.

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

## 5. Visualisasi
- **Bar plot** untuk kontribusi kata LIME & SHAP.
- **Kurva training** (accuracy/loss) untuk LSTM, BERT, LoRA.
- Warna:
  - Kontribusi positif → Merah
  - Kontribusi negatif → Biru

---

## 6. Catatan / Tips
- SHAP & LIME **memori-intensif**, gunakan sampel kecil untuk demo.
- Untuk BERT / LoRA:
  - CPU saja lambat, disarankan pakai GPU.
  - Batasi panjang sequence (`max_length=100`) untuk cepat.
- WandB logging **dimatikan**:
```python
import os
os.environ["WANDB_MODE"] = "disabled"
```
- LoRA memungkinkan adaptasi cepat ke dataset baru tanpa retraining full model.

---

## 7. Referensi
1. Ribeiro et al., *"Why Should I Trust You?" Explaining the Predictions of Any Classifier* (LIME)
2. Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP)
3. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*
4. Hugging Face Transformers Documentation
5. 🤗 PEFT library: https://github.com/huggingface/peft
