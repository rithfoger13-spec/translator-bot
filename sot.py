import telebot
import yt_dlp
import os
import subprocess
import whisper
import uuid
import asyncio
import edge_tts
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# ទាញយក Telegram Token ពី Environment Variables របស់ Render ដោយសុវត្ថិភាព
TELEGRAM_TOKEN = os.environ.get("BOT_TOKEN")

if not TELEGRAM_TOKEN:
    print("❌ កំហុស៖ រកមិនឃើញ BOT_TOKEN ក្នុង Environment Variables ទេ។")

bot = telebot.TeleBot(TELEGRAM_TOKEN)
user_sessions = {}

print("🤖 កំពុងផ្ទុក AI Whisper (Small Model)...")
whisper_model = whisper.load_model("tiny")
print("✅ AI Model រួចរាល់ហើយ!")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "👋 សួស្តី! សូមផ្ញើ Link ឬ File វីដេអូមក បូតនឹងទាញយក Script ដើម (មាន Timestamp) ដើម្បីឱ្យអ្នកងាយស្រួលផ្ទៀងផ្ទាត់ និងកែសម្រួលដោយខ្លួនឯង។")

@bot.message_handler(content_types=['video'])
def handle_uploaded_video(message):
    sent_msg = bot.reply_to(message, "📥 កំពុងទទួលបានវីដេអូ...")
    try:
        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        local_path = "user_video.mp4"
        with open(local_path, 'wb') as f:
            f.write(downloaded_file)
        
        user_sessions[message.chat.id] = {"type": "file", "path": local_path}
        process_video_transcription(message.chat.id, sent_msg.message_id)
    except Exception as e:
        bot.reply_to(message, f"❌ មានបញ្ហា៖ {e}")

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    chat_id = message.chat.id
    if user_sessions.get(chat_id, {}).get("state") == "waiting_for_edit":
        new_script = message.text
        user_sessions[chat_id]["translated_script"] = new_script
        user_sessions[chat_id]["state"] = None
        
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton("🎙️ បម្លែងសំឡេង Edge AI & ដាក់ចូលវីដេអូ", callback_data="make_video")
        )
        bot.send_message(
            chat_id,
            f"✍️ **Script ដែលបានកែសម្រួល៖**\n\n{new_script}",
            reply_markup=markup
        )
        return

    if message.text.startswith("http"):
        sent_msg = bot.reply_to(message, "⏳ កំពុងទាញយកវីដេអូតាម Link...")
        try:
            local_path = "user_video.mp4"
            if os.path.exists(local_path):
                os.remove(local_path)
                
            ydl_opts = {'format': 'best[height<=720]/best', 'outtmpl': local_path}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([message.text])
                
            user_sessions[chat_id] = {"type": "url", "path": local_path}
            bot.delete_message(chat_id, sent_msg.message_id)
            process_video_transcription_new(chat_id)
        except Exception as e:
            bot.reply_to(message, f"❌ ទាញយកមិនបានទេ៖ {e}")
    else:
        bot.reply_to(message, "⚠️ សូមផ្ញើ Link វីដេអូ ឬ File វីដេអូមក!")

def process_video_transcription(chat_id, message_id):
    bot.edit_message_text("⏳ កំពុងប្រើប្រាស់ Whisper AI ទាញយក Script ដើម...", chat_id, message_id)
    run_transcription(chat_id)

def process_video_transcription_new(chat_id):
    sent_msg = bot.send_message(chat_id, "⏳ កំពុងប្រើប្រាស់ Whisper AI ទាញយក Script ដើម...")
    run_transcription(chat_id, sent_msg.message_id)

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def run_transcription(chat_id, message_id=None):
    try:
        session = user_sessions.get(chat_id, {})
        video_path = session.get("path", "user_video.mp4")
        audio_path = "audio.mp3"
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', audio_path], 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        result = whisper_model.transcribe(audio_path)
        segments = result.get("segments", [])
        
        full_script = ""
        for i, seg in enumerate(segments):
            start_time = format_time(seg["start"])
            end_time = format_time(seg["end"])
            seg_text = seg["text"].strip()
            
            default_role = "ស្រី" if i % 2 == 0 else "ប្រុស"
            full_script += f"[{default_role}]: {seg_text}  ({start_time} - {end_time})\n"
            
        user_sessions[chat_id]["translated_script"] = full_script
        
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton("🎙️ បម្លែងសំឡេង Edge AI & ដាក់ចូលវីដេអូ", callback_data="make_video"),
            InlineKeyboardButton("❌ កែសម្រួល Script", callback_data="process_edit")
        )
        
        bot.send_message(
            chat_id,
            f"📜 **Script ភាសាដើម (មាន Timestamp សម្រាប់ផ្ទៀងផ្ទាត់)：**\n\n{full_script}\n\n💡 *អ្នកអាចចុច 'កែសម្រួល' ដើម្បីបកប្រែ និងរៀបចំតួអង្គប្រុស-ស្រីដោយខ្លួនឯង!*",
            reply_markup=markup
        )
    except Exception as e:
        bot.send_message(chat_id, f"❌ កំហុសឆ្គង៖ {e}")

async def generate_edge_audio(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

@bot.callback_query_handler(func=lambda call: call.data in ["make_video", "process_edit"])
def callback_actions(call):
    chat_id = call.message.chat.id
    
    if call.data == "process_edit":
        bot.answer_callback_query(call.id, text="Edit Mode")
        user_sessions[chat_id]["state"] = "waiting_for_edit"
        bot.send_message(chat_id, "✍️ សូមផ្ញើ Script ដែលអ្នកបានបកប្រែ និងកែសម្រួលរួច (ទម្រង់ `[ស្រី]: ...` ឬ `[ប្រុស]: ...`) មកទីនេះ:")
        
    elif call.data == "make_video":
        bot.answer_callback_query(call.id, text="កំពុងបង្កើតសំឡេង Edge AI...")
        bot.send_message(chat_id, "🎬 កំពុងបង្កើតសំឡេង AI ប្រុស-ស្រី និងបញ្ចូលទៅក្នុងវីដេអូ... សូមរង់ចាំបន្តិច!")
        
        try:
            session = user_sessions.get(chat_id, {})
            script_text = session.get("translated_script", "")
            video_path = session.get("path", "user_video.mp4")
            
            unique_id = str(uuid.uuid4())[:8]
            lines = script_text.split('\n')
            
            audio_files = []
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                
                if "(" in line and ")" in line and "-" in line:
                    line = line.split("(")[0].strip()

                role = "ស្រី"
                text_to_read = line
                if "[ប្រុស]:" in line:
                    role = "ប្រុស"
                    text_to_read = line.replace("[ប្រុស]:", "").strip()
                elif "[ស្រី]:" in line:
                    role = "ស្រី"
                    text_to_read = line.replace("[ស្រី]:", "").strip()
                
                if not text_to_read:
                    continue
                
                temp_audio = f"temp_{unique_id}_{idx}.mp3"
                voice_name = "km-KH-PisethNeural" if role == "ប្រុស" else "km-KH-SreymomNeural"
                
                asyncio.run(generate_edge_audio(text_to_read, voice_name, temp_audio))
                
                if os.path.exists(temp_audio):
                    audio_files.append(temp_audio)
            
            concat_file = f"concat_list_{unique_id}.txt"
            with open(concat_file, 'w', encoding='utf-8') as f:
                for af in audio_files:
                    f.write(f"file '{af}'\n")
            
            final_tts_audio = f"final_tts_{unique_id}.mp3"
            concat_cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_file, '-c', 'copy', final_tts_audio
            ]
            subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            output_video = f"output_video_{unique_id}.mp4"
            video_cmd = [
                'ffmpeg', '-y', '-i', video_path, '-i', final_tts_audio,
                '-c:v', 'libx264', '-crf', '28', '-preset', 'fast',
                '-vf', 'scale=-2:720',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', output_video
            ]
            subprocess.run(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            bot.send_message(chat_id, "📤 កំពុងផ្ញើវីដេអូទៅកាន់ Telegram...")
            with open(output_video, 'rb') as vid:
                bot.send_video(chat_id, vid, caption="🎉 វីដេអូបែងចែកសំឡេង AI ប្រុស-ស្រី រួចរាល់ដោយជោគជ័យ!", timeout=120)
                
            for af in audio_files:
                if os.path.exists(af):
                    os.remove(af)
            for f in [concat_file, final_tts_audio, output_video]:
                if os.path.exists(f):
                    os.remove(f)
                
        except Exception as e:
            bot.send_message(chat_id, f"❌ មានបញ្ហា៖ {e}")

print("🤖 Bot កំពុងដំណើរការ...")
bot.infinity_polling()
