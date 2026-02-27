from kokoro import KModel, KPipeline
import gradio as gr
import os
import random
import torch

IS_DUPLICATE = not os.getenv("SPACE_ID", "").startswith("hexgrad/")
CHAR_LIMIT = None if IS_DUPLICATE else 100000


REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
en_pipeline = KPipeline(lang_code="a", repo_id=REPO_ID, model=False)


def en_callable(text):
    if text == "Kokoro":
        return "kˈOkəɹO"
    elif text == "Sol":
        return "sˈOl"
    return next(en_pipeline(text)).phonemes


CUDA_AVAILABLE = torch.cuda.is_available()
models = {
    gpu: KModel(repo_id=REPO_ID).to("cuda" if gpu else "cpu").eval()
    for gpu in [False] + ([True] if CUDA_AVAILABLE else [])
}
pipelines = {
    lang_code: KPipeline(
        lang_code=lang_code, repo_id=REPO_ID, model=False, en_callable=en_callable
    )
    for lang_code in "abjzefhpi"
}
pipelines["a"].g2p.lexicon.golds["kokoro"] = "kˈOkəɹO"
pipelines["b"].g2p.lexicon.golds["kokoro"] = "kˈQkəɹQ"
# 🇺🇸 'a' => American English
# 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese
# 🇨🇳 'z' => Mandarin Chinese
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇧🇷 'p' => Brazilian Portuguese pt-br


def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)


def generate_first(text, voice="af_maple", speed=1, use_gpu=CUDA_AVAILABLE):
    text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed, split_pattern=r"\n+"):
        ref_s = pack[len(ps) - 1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info(
                    "Retrying with CPU. To avoid this error, change Hardware to CPU."
                )
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        return (24000, audio.numpy()), ps
    return None, ""


# Arena API
def predict(text, voice="af_maple", speed=1):
    return generate_first(text, voice, speed, use_gpu=False)[0]


def tokenize_first(text, voice="af_maple"):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ""


def generate_all(text, voice="af_maple", speed=1, use_gpu=CUDA_AVAILABLE):
    text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info("Switching to CPU")
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        yield 24000, audio.numpy()
        if first:
            first = False
            yield 24000, torch.zeros(1).numpy()


random_texts = {}
for lang in ["en", "ja", "zh", "es", "fr", "hi", "it", "pt-br"]:
    with open(f"{lang}.txt", "r", encoding="utf-8") as r:
        random_texts[lang] = [line.strip() for line in r]


def get_random_text(voice):
    lang = dict(
        a="en", b="en", j="ja", z="zh", e="es", f="fr", h="hi", i="it", p="pt-br"
    )[voice[0]]
    return random.choice(random_texts[lang])


def get_gatsby():
    with open("gatsby5k.md", "r", encoding="utf-8") as r:
        return r.read().strip()


def get_frankenstein():
    with open("frankenstein5k.md", "r", encoding="utf-8") as r:
        return r.read().strip()


CHOICES = {
    "🇺🇸 🚺 Maple ❤️": "af_maple",
    "🇺🇸 🚺 Sol 🔥": "af_sol",
    "🇬🇧 🚺 Vale 🎧": "bf_vale",
}


# 定义不连续的数字范围
zf_ranges = [
    range(1, 9),
    range(17, 20),
    range(21, 25),
    range(26, 29),
    range(32, 33),
    range(36, 37),
    range(38, 41),
    range(42, 45),
    range(46, 50),
    range(51, 52),
    range(59, 60),
    range(60, 61),
    range(67, 68),
    range(70, 80),
    range(83, 89),
    range(90, 91),
    range(92, 95),
    range(99, 100),
]
zm_ranges = [
    range(9, 17),
    range(20, 21),
    range(25, 26),
    range(29, 32),
    range(33, 36),
    range(37, 38),
    range(41, 42),
    range(45, 46),
    range(50, 51),
    range(52, 59),
    range(61, 67),
    range(68, 70),
    range(80, 83),
    range(89, 90),
    range(91, 92),
    range(95, 99),
    range(100, 101),
]



# 使用列表推导式为中国的选项添加到字典中
CHOICES.update({f"🇨🇳 🚹 {i:03d}": f"zf_{i:03d}" for r in zf_ranges for i in r})
CHOICES.update({f"🇨🇳 🚺 {i:03d}": f"zm_{i:03d}" for r in zm_ranges for i in r})


# 使用循环语句为中国的选项添加到字典中
""" for r in zf_ranges:
    for i in r:
        key = f"🇨🇳 🚹 {i:03d}"  # 格式化字符串，确保数字是三位数
        value = f"zf_{i:03d}"  # 格式化字符串，确保数字是三位数
        CHOICES[key] = value
for r in zm_ranges:
    for i in r:
        key = f"🇨🇳 🚺 {i:03d}"  # 格式化字符串，确保数字是三位数
        value = f"zm_{i:03d}"  # 格式化字符串，确保数字是三位数
        CHOICES[key] = value """

print(CHOICES)




for v in CHOICES.values():
    pipelines[v[0]].load_voice(v)

TOKEN_NOTE = """
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`

💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
"""

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(
        label="Output Audio", interactive=False, streaming=False, autoplay=True
    )
    generate_btn = gr.Button("Generate", variant="primary")
    with gr.Accordion("Output Tokens", open=True):
        out_ps = gr.Textbox(
            interactive=False,
            show_label=False,
            info="Tokens used to generate the audio, up to 510 context length.",
        )
        tokenize_btn = gr.Button("Tokenize", variant="secondary")
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button("Predict", variant="secondary", visible=False)

STREAM_NOTE = [
    "⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`."
]
if CHAR_LIMIT is not None:
    STREAM_NOTE.append(f"✂️ Each stream is capped at {CHAR_LIMIT} characters.")
    STREAM_NOTE.append(
        "🚀 Want more characters? You can [use Kokoro directly](https://huggingface.co/hexgrad/Kokoro-82M#usage) or duplicate this space:"
    )
STREAM_NOTE = "\n\n".join(STREAM_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(
        label="Output Audio Stream", interactive=False, streaming=True, autoplay=True
    )
    with gr.Row():
        stream_btn = gr.Button("Stream", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop")
    with gr.Accordion("Note", open=True):
        gr.Markdown(STREAM_NOTE)
        gr.DuplicateButton()

BANNER_TEXT = """
[***Kokoro*** **is an open-weight TTS model with 82 million parameters.**](https://huggingface.co/hexgrad/Kokoro-82M)

As of January 31st, 2025, Kokoro was the most-liked [**TTS model**](https://huggingface.co/models?pipeline_tag=text-to-speech&sort=likes) and the most-liked [**TTS space**](https://huggingface.co/spaces?sort=likes&search=tts) on Hugging Face.

This demo primarily showcases English, including variations such as American English 🇺🇸 ('a') and British English 🇬🇧 ('b'). However, you can directly use the model to access it in other languages like Japanese 🇯🇵 ('j'), Mandarin Chinese 🇨🇳 ('z'), Spanish 🇪🇸 ('e'), French 🇫🇷 ('f'), Hindi 🇮🇳 ('h'), Italian 🇮🇹 ('i'), and Brazilian Portuguese 🇧🇷 ('p').
"""
API_OPEN = os.getenv("SPACE_ID") != "hexgrad/Kokoro-TTS"
API_NAME = None if API_OPEN else False
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown(BANNER_TEXT, container=True)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Input Text",
                info=f"Up to ~500 characters per Generate, or {'∞' if CHAR_LIMIT is None else CHAR_LIMIT} characters per Stream",
            )
            with gr.Row():
                voice = gr.Dropdown(
                    list(CHOICES.items()),
                    value="af_maple",
                    label="Voice",
                    info="Quality and availability vary by language",
                )
                use_gpu = gr.Dropdown(
                    [("ZeroGPU 🚀", True), ("CPU 🐌", False)],
                    value=CUDA_AVAILABLE,
                    label="Hardware",
                    info="GPU is usually faster, but has a usage quota",
                    interactive=CUDA_AVAILABLE,
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="Speed")
            random_btn = gr.Button("🎲 Random Quote 💬", variant="secondary")
            with gr.Row():
                gatsby_btn = gr.Button("🥂 Gatsby 📕", variant="secondary")
                frankenstein_btn = gr.Button("💀 Frankenstein 📗", variant="secondary")
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ["Generate", "Stream"])
    random_btn.click(
        fn=get_random_text, inputs=[voice], outputs=[text], api_name=API_NAME
    )
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text], api_name=API_NAME)
    frankenstein_btn.click(
        fn=get_frankenstein, inputs=[], outputs=[text], api_name=API_NAME
    )
    generate_btn.click(
        fn=generate_first,
        inputs=[text, voice, speed, use_gpu],
        outputs=[out_audio, out_ps],
        api_name=API_NAME,
    )
    tokenize_btn.click(
        fn=tokenize_first, inputs=[text, voice], outputs=[out_ps], api_name=API_NAME
    )
    stream_event = stream_btn.click(
        fn=generate_all,
        inputs=[text, voice, speed, use_gpu],
        outputs=[out_stream],
        api_name=API_NAME,
    )
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(
        fn=predict, inputs=[text, voice, speed], outputs=[out_audio], api_name=API_NAME
    )

if __name__ == "__main__":
    app.queue(api_open=API_OPEN).launch(ssr_mode=True, inbrowser=True)
