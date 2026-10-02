# --- AI Video & Voice Dubbing Studio (Web & Tiny Model Edition) ---
import gradio as gr
import edge_tts
import asyncio
import os
import torch
import easyocr
from groq import Groq

# ទាញយក API Key ពី Environment Variables របស់ Render
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
try:
    from moviepy.audio.fx.audio_speedx import audio_speedx
except ImportError:
    try:
        from moviepy.audio.fx.all import audio_speedx
    except ImportError:
        audio_speedx = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU_OCR = (DEVICE == "cuda")
print(f"🚀 Initializing EasyOCR on {DEVICE.upper()} (GPU Enabled: {USE_GPU_OCR})...")

ocr_reader = easyocr.Reader(['ch_sim'], gpu=USE_GPU_OCR)

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
        print(f"✅ Groq ដំណើរការបានជោគជ័យ — ប្រើប្រាស់ model តូចលឿន: {GROQ_MODEL}")
    except Exception as e:
        print(f"⚠️ Groq Client Init Error: {e}")

def translate_with_groq_llm(text, target_lang_name):
    if not USE_GROQ or not text.strip():
        return f"[未翻译] {text}"
    
    system_msg = (
        f"You are a professional subtitle translator. Translate the given Chinese subtitle text "
        f"accurately and naturally into fluent {target_lang_name}. "
        "Provide ONLY the translated text in the target language script, with no explanations, no notes."
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

def step1_extract_subtitles_and_translate(media_file, target_lang_name, progress=gr.Progress()):
    if not media_file:
        return "⚠️ សូមអបឡូតវីដេអូរឿងសិន!", "", []
    if not USE_GROQ:
        return "⚠️ សូមពិនិត្យមើល GROQ_API_KEY របស់អ្នកនៅលើ Render!", "", []

    try:
        progress(0.1, desc="កំពុងបើកវីដេអូ...")
        video_clip = VideoFileClip(media_file)
        duration = video_clip.duration
        
        temp_audio = "extracted_audio.mp3"
        video_clip.audio.write_audiofile(temp_audio, logger=None)

        progress(0.2, desc="កំពុងអានអក្សរចិនពីវីដេអូ (OCR)...")
        fps_sample = 1.0 
        original_lines = []
        translated_lines = []
        timing = []
        
        current_time = 0.0
        seen_texts = set()
        
        while current_time < duration:
            frame = video_clip.get_frame(current_time)
            results = ocr_reader.readtext(frame)
            
            frame_texts = []
            for bbox, text, prob in results:
                if prob > 0.3:
                    y_coord = bbox[0][1]
                    if y_coord > frame.shape[0] * 0.65:
                        frame_texts.append(text)
            
            combined_text = "".join(frame_texts).strip()
            if len(combined_text) > 1 and combined_text not in seen_texts:
                seen_texts.add(combined_text)
                start_t = max(0.0, current_time - 0.5)
                end_t = min(duration, current_time + 0.5)
                
                original_lines.append(combined_text)
                trans_text = translate_with_groq_llm(combined_text, target_lang_name)
                translated_lines.append(trans_text)
                timing.append([start_t, end_t])

            current_time += fps_sample
            progress(0.2 + (current_time / duration) * 0.6, desc=f"កំពុងអាន OCR ដល់វិនាទីទី {current_time:.1f}s...")

        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        if not original_lines:
            return "❌ រកមិនឃើញអក្សរ Subtitle ក្នុងវីដេអូទេ!", "", []

        progress(1.0, desc="✨ អាន និងបកប្រែជោគជ័យ!")
        return "\n".join(original_lines), "\n".join(translated_lines), timing

    except Exception as e:
        return f"❌ កំហុស ៖ {str(e)}", "", []

def step2_generate_dubbing(media_file, editable_script, target_lang_name, timing_state, progress=gr.Progress()):
    if not media_file or not editable_script:
        return "⚠️ ទាមទារវីដេអូ និង Script!", None, None
    lang_info = LANGUAGE_OPTIONS.get(target_lang_name, LANGUAGE_OPTIONS["ខ្មែរ (Khmer)"])
    voice = lang_info["voices"][0]
    temp_files = []
    
    try:
        lines = [l.strip() for l in editable_script.split("\n") if l.strip()]
        use_timing = bool(timing_state) and len(timing_state) == len(lines)

        async def generate_tts_files():
            for idx, text in enumerate(lines):
                if not text:
                    temp_files.append(None)
                    continue
                temp_path = f"temp_line_{idx}.mp3"
                try:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(temp_path)
                    temp_files.append(temp_path)
                except Exception:
                    temp_files.append(None)

        progress(0.4, desc="កំពុងបង្កើតសំឡេងនិយាយ AI (Edge-TTS)...")
        asyncio.run(generate_tts_files())
        valid_files = [f for f in temp_files if f and os.path.exists(f)]
        
        if not valid_files:
            return "❌ បរាជ័យក្នុងការបង្កើតសំឡេង!", None, None

        progress(0.6, desc="កំពុងផ្គុំសំឡេង និងវីដេអូ...")
        video_clip = VideoFileClip(media_file)
        video_duration = video_clip.duration
        
        if use_timing:
            positioned_clips = []
            for idx, f in enumerate(temp_files):
                if not f or not os.path.exists(f):
                    continue
                start_sec = timing_state[idx][0]
                next_start = timing_state[idx + 1][0] if idx + 1 < len(timing_state) else video_duration
                slot_duration = max(0.5, next_start - start_sec)
                clip = AudioFileClip(f)
                actual_duration = clip.duration
                if audio_speedx and actual_duration > slot_duration:
                    speed_factor = min(actual_duration / slot_duration, 1.8)
                    clip = clip.fx(audio_speedx, speed_factor)
                positioned_clips.append(clip.set_start(start_sec))
            final_audio = CompositeAudioClip(positioned_clips).set_duration(video_duration)
        else:
            from moviepy.editor import concatenate_audioclips
            clips = [AudioFileClip(f) for f in valid_files]
            final_audio = concatenate_audioclips(clips)

        output_audio = "dubbed_audio.mp3"
        final_audio.write_audiofile(output_audio, fps=44100, logger=None)
        
        output_video = "dubbed_output_video.mp4"
        final_video = video_clip.set_audio(AudioFileClip(output_audio))
        final_video.write_videofile(
            output_video, codec="libx264", audio_codec="aac", fps=video_clip.fps or 30, audio_fps=44100, preset="medium", logger=None
        )
        
        progress(1.0, desc="✨ ជោគជ័យ ១០០%!")
        return "✅ បង្កើតវីដេអូ និងសំឡេងជោគជ័យ!", output_audio, output_video
    except Exception as e:
        return f"❌ កំហុស ៖ {str(e)}", None, None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇰🇭 AI Video & Voice Dubbing Studio (Web Version)")
    gr.Markdown("ប្រព័ន្ធបកប្រែវីដេអូរឿងចិន និងបង្កើតសំឡេងនិយាយជាភាសាខ្មែរ ២៤/៧!")
    
    timing_state = gr.State([])
    with gr.Row():
        with gr.Column():
            media_input = gr.Video(label="📤 អបឡូតវីដេអូរឿងចិនរបស់អ្នក")
            lang_dropdown = gr.Dropdown(choices=LANGUAGE_NAMES, value="ខ្មែរ (Khmer)", label="🎯 ភាសាគោលដៅ (Target Language)")
            step1_btn = gr.Button("🚀 ជំហានទី ១: អានអក្សរ Subtitle ពីវីដេអូ + បកប្រែ", variant="secondary")
        with gr.Column():
            out_original = gr.Textbox(label="📝 អក្សរចិន OCR တွေ့ក្នុងវីដេអូ (Original)", lines=5)
            editable_script_box = gr.Textbox(label="✍️ ជំហានទី ២: កែសម្រួល Script", lines=7, interactive=True)
            step2_btn = gr.Button("✨ ជំហានទី ៣: បង្កើតសំឡេង និងវីដេអូចុងក្រោយ", variant="primary")
            status_output = gr.Textbox(label="📊 ស្ថានភាព (Status)")
            out_audio = gr.Audio(label="🎵 សំឡេងចេញជាភាសាខ្មែរ")
            out_video = gr.Video(label="🎥 វីដេអូចុងក្រោយ")

    step1_btn.click(fn=step1_extract_subtitles_and_translate, inputs=[media_input, lang_dropdown], outputs=[out_original, editable_script_box, timing_state])
    step2_btn.click(fn=step2_generate_dubbing, inputs=[media_input, editable_script_box, lang_dropdown, timing_state], outputs=[status_output, out_audio, out_video])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
import gradio as gr
import edge_tts
import asyncio
import os
import torch
import easyocr
from groq import Groq

# ទាញយក API Key ពី Environment Variables របស់ Render
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
try:
    from moviepy.audio.fx.audio_speedx import audio_speedx
except ImportError:
    try:
        from moviepy.audio.fx.all import audio_speedx
    except ImportError:
        audio_speedx = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU_OCR = (DEVICE == "cuda")
print(f"🚀 Initializing EasyOCR on {DEVICE.upper()} (GPU Enabled: {USE_GPU_OCR})...")

ocr_reader = easyocr.Reader(['ch_sim'], gpu=USE_GPU_OCR)

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
        print(f"✅ Groq ដំណើរការបានជោគជ័យ — ប្រើប្រាស់ model តូចលឿន: {GROQ_MODEL}")
    except Exception as e:
        print(f"⚠️ Groq Client Init Error: {e}")

def translate_with_groq_llm(text, target_lang_name):
    if not USE_GROQ or not text.strip():
        return f"[未翻译] {text}"
    
    system_msg = (
        f"You are a professional subtitle translator. Translate the given Chinese subtitle text "
        f"accurately and naturally into fluent {target_lang_name}. "
        "Provide ONLY the translated text in the target language script, with no explanations, no notes."
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

def step1_extract_subtitles_and_translate(media_file, target_lang_name, progress=gr.Progress()):
    if not media_file:
        return "⚠️ សូមអបឡូតវីដេអូរឿងសិន!", "", []
    if not USE_GROQ:
        return "⚠️ សូមពិនិត្យមើល GROQ_API_KEY របស់អ្នកនៅលើ Render!", "", []

    try:
        progress(0.1, desc="កំពុងបើកវីដេអូ...")
        video_clip = VideoFileClip(media_file)
        duration = video_clip.duration
        
        temp_audio = "extracted_audio.mp3"
        video_clip.audio.write_audiofile(temp_audio, logger=None)

        progress(0.2, desc="កំពុងអានអក្សរចិនពីវីដេអូ (OCR)...")
        fps_sample = 1.0 
        original_lines = []
        translated_lines = []
        timing = []
        
        current_time = 0.0
        seen_texts = set()
        
        while current_time < duration:
            frame = video_clip.get_frame(current_time)
            results = ocr_reader.readtext(frame)
            
            frame_texts = []
            for bbox, text, prob in results:
                if prob > 0.3:
                    y_coord = bbox[0][1]
                    if y_coord > frame.shape[0] * 0.65:
                        frame_texts.append(text)
            
            combined_text = "".join(frame_texts).strip()
            if len(combined_text) > 1 and combined_text not in seen_texts:
                seen_texts.add(combined_text)
                start_t = max(0.0, current_time - 0.5)
                end_t = min(duration, current_time + 0.5)
                
                original_lines.append(combined_text)
                trans_text = translate_with_groq_llm(combined_text, target_lang_name)
                translated_lines.append(trans_text)
                timing.append([start_t, end_t])

            current_time += fps_sample
            progress(0.2 + (current_time / duration) * 0.6, desc=f"កំពុងអាន OCR ដល់វិនាទីទី {current_time:.1f}s...")

        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        if not original_lines:
            return "❌ រកមិនឃើញអក្សរ Subtitle ក្នុងវីដេអូទេ!", "", []

        progress(1.0, desc="✨ អាន និងបកប្រែជោគជ័យ!")
        return "\n".join(original_lines), "\n".join(translated_lines), timing

    except Exception as e:
        return f"❌ កំហុស ៖ {str(e)}", "", []

def step2_generate_dubbing(media_file, editable_script, target_lang_name, timing_state, progress=gr.Progress()):
    if not media_file or not editable_script:
        return "⚠️ ទាមទារវីដេអូ និង Script!", None, None
    lang_info = LANGUAGE_OPTIONS.get(target_lang_name, LANGUAGE_OPTIONS["ខ្មែរ (Khmer)"])
    voice = lang_info["voices"][0]
    temp_files = []
    
    try:
        lines = [l.strip() for l in editable_script.split("\n") if l.strip()]
        use_timing = bool(timing_state) and len(timing_state) == len(lines)

        async def generate_tts_files():
            for idx, text in enumerate(lines):
                if not text:
                    temp_files.append(None)
                    continue
                temp_path = f"temp_line_{idx}.mp3"
                try:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(temp_path)
                    temp_files.append(temp_path)
                except Exception:
                    temp_files.append(None)

        progress(0.4, desc="កំពុងបង្កើតសំឡេងនិយាយ AI (Edge-TTS)...")
        asyncio.run(generate_tts_files())
        valid_files = [f for f in temp_files if f and os.path.exists(f)]
        
        if not valid_files:
            return "❌ បរាជ័យក្នុងការបង្កើតសំឡេង!", None, None

        progress(0.6, desc="កំពុងផ្គុំសំឡេង និងវីដេអូ...")
        video_clip = VideoFileClip(media_file)
        video_duration = video_clip.duration
        
        if use_timing:
            positioned_clips = []
            for idx, f in enumerate(temp_files):
                if not f or not os.path.exists(f):
                    continue
                start_sec = timing_state[idx][0]
                next_start = timing_state[idx + 1][0] if idx + 1 < len(timing_state) else video_duration
                slot_duration = max(0.5, next_start - start_sec)
                clip = AudioFileClip(f)
                actual_duration = clip.duration
                if audio_speedx and actual_duration > slot_duration:
                    speed_factor = min(actual_duration / slot_duration, 1.8)
                    clip = clip.fx(audio_speedx, speed_factor)
                positioned_clips.append(clip.set_start(start_sec))
            final_audio = CompositeAudioClip(positioned_clips).set_duration(video_duration)
        else:
            from moviepy.editor import concatenate_audioclips
            clips = [AudioFileClip(f) for f in valid_files]
            final_audio = concatenate_audioclips(clips)

        output_audio = "dubbed_audio.mp3"
        final_audio.write_audiofile(output_audio, fps=44100, logger=None)
        
        output_video = "dubbed_output_video.mp4"
        final_video = video_clip.set_audio(AudioFileClip(output_audio))
        final_video.write_videofile(
            output_video, codec="libx264", audio_codec="aac", fps=video_clip.fps or 30, audio_fps=44100, preset="medium", logger=None
        )
        
        progress(1.0, desc="✨ ជោគជ័យ ១០០%!")
        return "✅ បង្កើតវីដេអូ និងសំឡេងជោគជ័យ!", output_audio, output_video
    except Exception as e:
        return f"❌ កំហុស ៖ {str(e)}", None, None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇰🇭 AI Video & Voice Dubbing Studio (Web Version)")
    gr.Markdown("ប្រព័ន្ធបកប្រែវីដេអូរឿងចិន និងបង្កើតសំឡេងនិយាយជាភាសាខ្មែរ ២៤/៧!")
    
    timing_state = gr.State([])
    with gr.Row():
        with gr.Column():
            media_input = gr.Video(label="📤 អបឡូតវីដេអូរឿងចិនរបស់អ្នក")
            lang_dropdown = gr.Dropdown(choices=LANGUAGE_NAMES, value="ខ្មែរ (Khmer)", label="🎯 ភាសាគោលដៅ (Target Language)")
            step1_btn = gr.Button("🚀 ជំហានទី ១: អានអក្សរ Subtitle ពីវីដេអូ + បកប្រែ", variant="secondary")
        with gr.Column():
            out_original = gr.Textbox(label="📝 អក្សរចិន OCR တွေ့ក្នុងវីដេអូ (Original)", lines=5)
            editable_script_box = gr.Textbox(label="✍️ ជំហានទី ២: កែសម្រួល Script", lines=7, interactive=True)
            step2_btn = gr.Button("✨ ជំហានទី ៣: បង្កើតសំឡេង និងវីដេអូចុងក្រោយ", variant="primary")
            status_output = gr.Textbox(label="📊 ស្ថានភាព (Status)")
            out_audio = gr.Audio(label="🎵 សំឡេងចេញជាភាសាខ្មែរ")
            out_video = gr.Video(label="🎥 វីដេអូចុងក្រោយ")

    step1_btn.click(fn=step1_extract_subtitles_and_translate, inputs=[media_input, lang_dropdown], outputs=[out_original, editable_script_box, timing_state])
    step2_btn.click(fn=step2_generate_dubbing, inputs=[media_input, editable_script_box, lang_dropdown, timing_state], outputs=[status_output, out_audio, out_video])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
