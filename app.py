import gradio as gr
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

# ── Define Missing Globals ──
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model and tokenizer ──
REPO_NAME = "AhmedJaber/biobert-main"

inference_tokenizer = AutoTokenizer.from_pretrained(REPO_NAME)
inference_model = AutoModelForTokenClassification.from_pretrained(REPO_NAME)

inference_model.to(DEVICE)
inference_model.eval()

# ── Get labels dynamically from the model config ──
# This prevents you from having to hardcode BC5CDR_ID2LABEL
BC5CDR_ID2LABEL = inference_model.config.id2label

# ── Color mapping ──
# ... (Keep your ENTITY_COLORS and the rest of the file exactly the same below here)



# ── Color mapping ──
ENTITY_COLORS = {
    "Chemical": {"bg": "#B5D4F4", "text": "#0C447C"},
    "Disease":  {"bg": "#F5C4B3", "text": "#712B13"},
    "Symptom":  {"bg": "#9FE1CB", "text": "#085041"},
}
DEFAULT_COLOR = {"bg": "#E0E0E0", "text": "#333333"}

def predict_ner(text: str):
    """Run NER inference on input text and return highlighted HTML + JSON.
    
    Args:
        text (str): Clinical text input.
    
    Returns:
        tuple: (html_output, json_output) — highlighted text and structured entities.
    """
    if not text.strip():
        return "<p>Please enter some text.</p>", {"entities": []}
    
    # Tokenize
    words = text.split()
    inputs = inference_tokenizer(
        words, is_split_into_words=True,
        return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = inference_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
    
    # Reconstruct word-level predictions from subtokens
    word_ids = inputs.get("word_ids", None)
    if word_ids is None:
        # Manual word_ids extraction
        word_ids_list = inference_tokenizer(
            words, is_split_into_words=True,
            truncation=True, max_length=MAX_LENGTH
        ).word_ids()
    else:
        word_ids_list = word_ids[0].cpu().tolist()
    
    word_preds = {}
    for idx, word_id in enumerate(word_ids_list):
        if word_id is not None and word_id not in word_preds:
            word_preds[word_id] = BC5CDR_ID2LABEL[predictions[idx]]
    
    # Build HTML output
    html_parts = []
    entities_json = []
    char_offset = 0
    
    for i, word in enumerate(words):
        label = word_preds.get(i, "O")
        
        if label != "O":
            etype = label[2:]  # Remove B- or I- prefix
            colors = ENTITY_COLORS.get(etype, DEFAULT_COLOR)
            html_parts.append(
                f'<span style="background-color:{colors["bg"]};color:{colors["text"]};'
                f'padding:2px 6px;border-radius:4px;margin:1px;font-weight:500;">'
                f'{word} <sup style="font-size:0.7em;opacity:0.8;">{etype}</sup></span>'
            )
            if label.startswith("B-"):
                entities_json.append({
                    "text": word,
                    "label": etype,
                    "start": char_offset,
                    "end": char_offset + len(word),
                })
            elif entities_json and entities_json[-1]["label"] == etype:
                entities_json[-1]["text"] += " " + word
                entities_json[-1]["end"] = char_offset + len(word)
        else:
            html_parts.append(f'<span style="margin:1px;">{word}</span>')
        
        char_offset += len(word) + 1
    
    html_output = (
        '<div style="font-family:Inter,sans-serif;font-size:16px;line-height:2.2;'
        'padding:20px;background:#FAFAFA;border-radius:12px;border:1px solid #E0E0E0;">'
        + ' '.join(html_parts) + '</div>'
    )
    
    json_output = {"entities": entities_json, "total_entities": len(entities_json)}
    
    return html_output, json_output

# ── Example sentences ──
EXAMPLES = [
    "Patient presents with chest pain and shortness of breath. Started on aspirin 325mg and metoprolol 50mg twice daily.",
    "The study found that ibuprofen was effective in reducing inflammation associated with rheumatoid arthritis.",
    "Adverse effects of cisplatin include nephrotoxicity, ototoxicity, and severe nausea requiring ondansetron prophylaxis.",
    "Patient diagnosed with type 2 diabetes mellitus and hypertension, prescribed metformin 1000mg and lisinopril 10mg.",
    "Doxorubicin-induced cardiotoxicity remains a significant challenge in breast cancer chemotherapy regimens.",
]

# ── Build Gradio Interface ──
# ── Build Gradio Interface ──
# ── Build Gradio Interface ──
demo = gr.Interface(
    fn=predict_ner,
    inputs=gr.Textbox(
        label="🏥 Enter Clinical Text",
        placeholder="Type or paste medical text here...",
        lines=4,
    ),
    outputs=[
        gr.HTML(label="🔍 Recognized Entities"),
        gr.JSON(label="📋 Structured Output"),
    ],
    title="🏥 Medical Named Entity Recognition",
    description=(
        "Extract **chemicals/drugs**, **diseases**, and other clinical entities "
        "from biomedical text using a fine-tuned BioBERT model.\n\n"
        "🔵 Chemical/Drug &nbsp; 🔴 Disease &nbsp; 🟢 Symptom"
    ),
    examples=EXAMPLES,
    flagging_mode="never", # ✅ Updated parameter name for Gradio 6.0
)

# ✅ Move theme to the launch method if you are calling it at the bottom of the file
# If you let Hugging Face Spaces launch it automatically, it will ignore the theme, 
# but it won't crash!
print("✅ Gradio demo built. Launch with demo.launch(share=True)")

# If you have a demo.launch() line below this, update it to:
# demo.launch(theme=gr.themes.Soft(), share=True)

print("✅ Gradio demo built. Launch with demo.launch(share=True)")

# ── Launch Gradio Demo ──
# Note: share=True creates a public link (valid for 72 hours)
try:
    demo.launch(theme=gr.themes.Soft(), share=True)
except Exception as e:
    print(f"⚠️ Gradio launch note: {e}")
    print("Try running demo.launch(share=False) for local-only access.")