from kokoro import KModel, KPipeline
import gradio as gr
import os
import random
import torch

IS_DUPLICATE = not os.getenv("SPACE_ID", "").startswith("hexgrad/")
CHAR_LIMIT = None if IS_DUPLICATE else 100000
REPO_ID = "hexgrad/Kokoro-82M"
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
    lang_code: KPipeline(lang_code=lang_code, model=False, repo_id=REPO_ID, en_callable=en_callable)
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


def generate_first(text, voice="af_heart", speed=1, use_gpu=CUDA_AVAILABLE):
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
def predict(text, voice="af_heart", speed=1):
    return generate_first(text, voice, speed, use_gpu=False)[0]


def tokenize_first(text, voice="af_heart"):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ""


def generate_all(text, voice="af_heart", speed=1, use_gpu=CUDA_AVAILABLE):
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
    "🇺🇸 🚺 Heart ❤️": "af_heart",
    "🇺🇸 🚺 Bella 🔥": "af_bella",
    "🇺🇸 🚺 Nicole 🎧": "af_nicole",
    "🇺🇸 🚺 Aoede": "af_aoede",
    "🇺🇸 🚺 Kore": "af_kore",
    "🇺🇸 🚺 Sarah": "af_sarah",
    "🇺🇸 🚺 Nova": "af_nova",
    "🇺🇸 🚺 Sky": "af_sky",
    "🇺🇸 🚺 Alloy": "af_alloy",
    "🇺🇸 🚺 Jessica": "af_jessica",
    "🇺🇸 🚺 River": "af_river",
    "🇺🇸 🚹 Michael": "am_michael",
    "🇺🇸 🚹 Fenrir": "am_fenrir",
    "🇺🇸 🚹 Puck": "am_puck",
    "🇺🇸 🚹 Echo": "am_echo",
    "🇺🇸 🚹 Eric": "am_eric",
    "🇺🇸 🚹 Liam": "am_liam",
    "🇺🇸 🚹 Onyx": "am_onyx",
    "🇺🇸 🚹 Santa": "am_santa",
    "🇺🇸 🚹 Adam": "am_adam",
    "🇬🇧 🚺 Emma": "bf_emma",
    "🇬🇧 🚺 Isabella": "bf_isabella",
    "🇬🇧 🚺 Alice": "bf_alice",
    "🇬🇧 🚺 Lily": "bf_lily",
    "🇬🇧 🚹 George": "bm_george",
    "🇬🇧 🚹 Fable": "bm_fable",
    "🇬🇧 🚹 Lewis": "bm_lewis",
    "🇬🇧 🚹 Daniel": "bm_daniel",
    "🇯🇵 🚺 Alpha": "jf_alpha",
    "🇯🇵 🚺 Gongitsune": "jf_gongitsune",
    "🇯🇵 🚺 Nezumi": "jf_nezumi",
    "🇯🇵 🚺 Tebukuro": "jf_tebukuro",
    "🇯🇵 🚹 Kumo": "jm_kumo",
    "🇨🇳 🚺 小北": "zf_xiaobei",
    "🇨🇳 🚺 小妮": "zf_xiaoni",
    "🇨🇳 🚺 小晓": "zf_xiaoxiao",
    "🇨🇳 🚺 小依": "zf_xiaoyi",
    "🇨🇳 🚺 云霞": "zm_yunxia",
    "🇨🇳 🚹 云健": "zm_yunjian",
    "🇨🇳 🚹 云溪": "zm_yunxi",
    "🇨🇳 🚹 云阳": "zm_yunyang",
    "🇪🇸 🚺 Dora": "ef_dora",
    "🇪🇸 🚹 Alex": "em_alex",
    "🇪🇸 🚹 Santa": "em_santa",
    "🇫🇷 🚺 Siwis": "ff_siwis",
    "🇮🇳 🚺 Alpha": "hf_alpha",
    "🇮🇳 🚺 Beta": "hf_beta",
    "🇮🇳 🚹 Omega": "hm_omega",
    "🇮🇳 🚹 Psi": "hm_psi",
    "🇮🇹 🚺 Sara": "if_sara",
    "🇮🇹 🚹 Nicola": "im_nicola",
    "🇧🇷 🚺 Dora": "pf_dora",
    "🇧🇷 🚹 Alex": "pm_alex",
    "🇧🇷 🚹 Santa": "pm_santa",
}
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
                    value="af_heart",
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
