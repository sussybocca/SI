# app.py
import sys
import os
import subprocess

# Try to build the module if .so doesn't exist
module_path = os.path.join(os.path.dirname(__file__), "si_engine")
if not os.path.exists(module_path + ".so") and not os.path.exists(module_path + ".pyd"):
    print("Building SI Engine module...")
    subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
    print("Build complete!")

# Now import
import si_engine
import gradio as gr

si = si_engine.SyntheticIntelligence()

def chat(message, history):
    if not message or message.strip() == "":
        return "Please ask me something!"
    return si.ask(message)

def teach(word, meaning):
    if not word or not meaning:
        return "Please provide both a word and its meaning."
    si.teach(word, meaning)
    return f"✅ Learned: **{word}** = {meaning}"

def train_text(text):
    if not text or text.strip() == "":
        return "Please provide some training text."
    si.train(text)
    return f"✅ Trained on: {text[:100]}..."

def get_stats():
    stats = si.get_stats()
    return f"""
| Metric | Value |
|--------|-------|
| Vocabulary | {stats['vocabulary_size']} |
| Knowledge | {stats['knowledge_base_size']} |
| Web Searches | {stats['web_search_count']} |
"""

with gr.Blocks(title="SI Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 SI Engine - Synthetic Intelligence")
    
    with gr.Tab("💬 Chat"):
        gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(height=400),
            textbox=gr.Textbox(placeholder="Ask me anything...")
        )
    
    with gr.Tab("📚 Teach"):
        word_input = gr.Textbox(label="Word/Concept")
        meaning_input = gr.Textbox(label="Meaning")
        teach_btn = gr.Button("Teach SI", variant="primary")
        teach_output = gr.Markdown()
        teach_btn.click(teach, inputs=[word_input, meaning_input], outputs=teach_output)
    
    with gr.Tab("🏋️ Train"):
        train_input = gr.Textbox(label="Training Text", lines=3)
        train_btn = gr.Button("Train", variant="primary")
        train_output = gr.Markdown()
        train_btn.click(train_text, inputs=train_input, outputs=train_output)
    
    with gr.Tab("📊 Stats"):
        stats_btn = gr.Button("Refresh")
        stats_output = gr.Markdown()
        stats_btn.click(get_stats, outputs=stats_output)

if __name__ == "__main__":
    demo.launch()