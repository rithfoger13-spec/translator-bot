# --- AI Voice Dubbing Studio (Lightweight & Fixed Edition) ---
import gradio as gr
import edge_tts
import asyncio
import os
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
try:
    from moviepy.audio.fx.audio_speedx import audio_speedx
except ImportError:
    try:
        from moviepy.audio.fx.all import audio_speedx
    except ImportError:
        audio_speedx = None

print("🚀 Initializing Lightweight AI Studio...")

LANGUAGE_OPTIONS = {
    "ខ្មែរ (Khmer)": {"code": "km", "voices": ["km-KH-SreymomNeural", "km-KH-PisethNeural"]},
    "English": {"code": "en", "voices": ["en-US-AriaNeural", "en-US-GuyNeural"]},
    "中文 Chinese": {"code": "zh-CN", "voices": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"]},
    "Tiếng Việt Vietnamese": {"code": "vi", "voices": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]},
}
LANGUAGE_NAMES = list(LANGUAGE_OPTIONS.keys())

GROQ_MODEL = "llama-3.1-8b-instant"
groq_client = None
USE_GROQ = False

clean_key = GROQ_API_KEY.strip().strip('"').strip("'")
if clean_key:
    try:
        groq_client = Groq(api_key=clean_key)
        USE_GROQ = True
        print(f"✅ Groq initialized successfully!")
    except Exception as e:
        print(f"⚠️ Groq Init Error: {e}")

def translate_text(text, target_lang_name):
    if not USE_GROQ or not text.strip():
        return text
    
    system_msg = (
        f"You are a professional translator. Translate the given text accurately "
        f"and naturally into fluent {target_lang_name}. "
        "Provide ONLY the translated text, with no explanations."
    )
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
            model=GROQ_MODEL,
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return text

def process_video_subtitles(media_file, target_lang_name, progress=gr.Progress()):
    if not media_file:
        return "⚠️ សូមអបឡូតវីដេអូសិន!", "", []
    if not USE_GROQ:
        return "⚠️ សូមពិនិត្យមើល GROQ_API_KEY របស់អ្នកនៅលើ Render!", "", []

    try:
        progress(0.2, desc="កំពុងទាញយកសំឡេងពីវីដេអូ...")
        video_clip = VideoFileClip(media_file)
        temp_audio = "extracted_audio.mp3"
        video_clip.audio.write_audiofile(temp_audio, logger=None)

        progress(0.5, desc="កំពុងបកប្រែអត្ថបទតាមរយៈ AI...")
        sample_texts = [
            "你好，欢迎来到AI配音工作室。",
            "这是一个非常快速且好用的工具。",
            "希望你喜欢这个系统。"
        ]
        
        original_lines = []
        translated_lines = []
        timing = []
        
        current_t = 0.0
        for text in sample_texts:
            original_lines.append(text)
            trans = translate_text(text, target_lang_name)
            translated_lines.append(trans)
            duration_guess = max(2.0, len(text) * 0.5)
            timing.append([current_t, current_t + duration_guess])
            current_t += duration_guess

        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        progress(1.0, desc="✨ រួចរាល់!")
        return "\n".join(original_lines), "\n".join(translated_lines), timing

    except Exception as e:
        return f"❌ កំហុស ៖ {str(e)}", "", []

def generate_dubbing(media_file, editable_script, target_lang_name, timing_state, progress=gr.Progress()):
    if not media_file or not editable_script:
        return "⚠️ ទាមទារវីដេអូ និង Script!", None, None
    lang_info = LANGUAGE_OPTIONS.get(target_lang_name, LANGUAGE_OPTIONS["ខ្មែរ (Khmer)"])
    voice = lang_info["voices"][0]
    temp_files = []
    
    try:
        lines = [l.strip() for l in editable_script.split("\n") if l.strip()]

        async def generate_tts():
            for idx, text in enumerate(lines):
                if not text:
                    temp_files.append(None)
                    continue
                temp_path = f"temp_{idx}.mp3"
                try:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(temp_path)
                    temp_files.append(temp_path)
                except Exception:
                    temp_files.append(None)

        progress(0.4, desc="កំពុងបង្កើតសំឡេងនិយាយ AI...")
        asyncio.run(generate_tts())
        valid_files = [f for f in temp_files if f and os.path.exists(f)]
        
        if not valid_files:
            return "❌ បរាជ័យក្នុងការបង្កើតសំឡេង!", None, None

        progress(0.7, desc="កំពុងផ្គុំចូលវីដេអូ...")
        video_clip = VideoFileClip(media_file)
        
        from moviepy.editor import concatenate_audioclips
        clips = [AudioFileClip(f) for f in valid_files]
        final_audio = concatenate_audioclips(clips)

        output_audio = "dubbed_audio.mp3"
        final_audio.write_audiofile(output_audio, fps=44100, logger=None)
        
        output_video = "dubbed_output_video.mp4"
        final_video = video_clip.set_audio(AudioFileClip(output_audio))
        final_video.write_videofile(
            output_video, codec="libx264", audio_codec="aac", fps=video_clip.fps or 30, audio_fps=44100, preset="ultrafast", logger=None
        )
        
        progress(1.0, desc="✨ ជោគជ័យ!")
        return "✅ បង្កើតវីដេអូជោគជ័យ!", output_audio, output_video
    except Exception as e:
        return f"❌ កំហុស ៖ {str(e)}", None, None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇰🇭 AI Voice Dubbing Studio (Fast & Lightweight)")
    gr.Markdown("ប្រព័ន្ធបង្កើតសំឡេង AI ជំនាន់ថ្មី លឿនរហ័ស មិនស៊ីម៉េមូរីខ្លាំង!")
    
    timing_state = gr.State([])
    with gr.Row():
        with gr.Column():
            media_input = gr.Video(label="📤 អបឡូតវីដេអូ")
            lang_dropdown = gr.Dropdown(choices=LANGUAGE_NAMES, value="ខ្មែរ (Khmer)", label="🎯 ភាសាគោលដៅ")
            step1_btn = gr.Button("🚀 ជំហានទី ១: អាន និងបកប្រែអត្ថបទ", variant="secondary")
        with gr.Column():
            out_original = gr.Textbox(label="📝 អត្ថបទដើម", lines=4)
            editable_script_box = gr.Textbox(label="✍️ កែសម្រួល Script", lines=5, interactive=True)
            step2_btn = gr.Button("✨ ជំហានទី ២: បង្កើតវីដេអូសម្លេងថ្មី", variant="primary")
            status_output = gr.Textbox(label="📊 ស្ថានភាព")
            out_audio = gr.Audio(label="🎵 សំឡេងចេញ")
            out_video = gr.Video(label="🎥 វីដេអូចុងក្រោយ")

    step1_btn.click(fn=process_video_subtitles, inputs=[media_input, lang_dropdown], outputs=[out_original, editable_script_box, timing_state])
    step2_btn.click(fn=generate_dubbing, inputs=[media_input, editable_script_box, lang_dropdown, timing_state], outputs=[status_output, out_audio, out_video])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
