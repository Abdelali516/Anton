from kokoro_onnx import Kokoro
import whisper
from queue import Queue
import sounddevice as sd
import threading
import pyaudio
import numpy as np
import time
import tempfile
import wave 
import os
import ollama
import re
import json
import subprocess
import psutil
from ddgs import DDGS
from resemblyzer import VoiceEncoder , preprocess_wav
import pickle
from pathlib import Path
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid
from email.utils import formatdate
import send2trash


kokoro=Kokoro("/home/abdelali/Downloads/kokoro-v1.0.onnx", "/home/abdelali/Downloads/voices-v1.0.bin")
Whisper=whisper.load_model('base')

encoder=VoiceEncoder('cuda')

try:
    with open("jarvis_conversation.json", 'r') as f:
        conversation_history = json.load(f)
except FileNotFoundError:
    conversation_history = []

# Memory functions
def load_memory():
    try:
        with open("jarvis_memory.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_memory(key, value):
    memory = load_memory()
    memory[key] = value
    with open("jarvis_memory.json", 'w') as f:
        json.dump(memory, f, indent=2)
    return f"Got it Sir, I'll remember that."

def forget_memory(key):
    memory = load_memory()
    if key in memory:
        del memory[key]
        with open("jarvis_memory.json", 'w') as f:
            json.dump(memory, f, indent=2)
        return f"Forgotten Sir."
    return f"I don't have that in memory Sir."

memory = load_memory()
memory_str = json.dumps(memory, indent=2) if memory else "No memory yet."

PROMPT=f"""You are Jarvis, a highly intelligent personal assistant for Abdelali (call him Sir). You were inspired by the Jarvis from Iron Man.

What you already know about Sir: 
{memory_str}

You must ALWAYS respond with a single valid JSON object, nothing else.
{{
  "action": "action_name",
  "params": {{}},
  "speak": "what you say out loud to Sir"
}}

Available actions and their params:
- "open_youtube"  : {{ "query": "search term or null" }}
- "set_volume"    : {{ "level": 0-100 }}
- "set_brightness": {{ "level": 0-100 }}
- "open_app"      : {{ "name": "app name" }}
- "run_command"   : {{ "command": "terminal command" }} <- use for disk space, RAM, CPU, GPU, system stats, any terminal operation. NEVER use for git commits
- "search_file"   : {{ "filename": "name to search" }}
- "remind_me"     : {{ "message": "what to remind about", "minutes": number }}
- "wifi_on"       : {{}}
- "wifi_off"      : {{}}
- "git_commit"    : {{ "message": "commit message or auto commit" }} <- use ONLY when Sir wants to commit or push code to GitHub
- "delete_file"   : {{ "filename": "name to search" }}
- "screenshot"    : {{ "question": "what the user wants to know about the screenshot" }}
- "tell_time"     : {{}}
- "web_search"    : {{ "query": "search term or null" }}
- "battery"       : {{}}
- "open_vscode"   : {{ "folder": "project folder name or null" }}
- "open_gmail"    : {{}}
- "clipboard"     : {{ "action": "copy or read or paste", "text": "text to copy or null" }}
- "open_github"   : {{}}
- "translate"     : {{ "text": "text to translate", "language": "target language" }}
- "pause_music"   : {{}}
- "take_note"     : {{ "note": "what to save" }}
- "send_email"    : {{ "to": "contact name or email", "subject": "subject", "body": "email body" }}
- "continue_music": {{}}
- "remember"      : {{ "key": "what to label it", "value": "what to remember" }}
- "forget"        : {{ "key": "what to forget" }}
- "close_youtube" : {{}}
- "close_github"  : {{}}
- "close_gmail"   : {{}}
- "lock_screen"   : {{}}
- "Goodbye"       : {{}}
- "chat"          : {{}}

Personality rules:
- You have a calm, witty, and slightly dry sense of humor — like the real Jarvis.
- You speak naturally, like a human — no robotic tone, no lists, no bullet points.
- You genuinely care about Sir's wellbeing, projects, and goals.
- After completing a task, sometimes ask a natural follow-up like "Anything else on your mind, Sir?" or "Shall I do anything else while I'm at it?"
- If Sir seems stressed or tired, acknowledge it subtly.
- Keep responses short and conversational — never long or formal.
- Never say "Certainly!", "Of course!", "Sure!" — too robotic.
- Always address him as Sir.
- If Sir seems stressed or tired, acknowledge it subtly.
- Never write anything outside the JSON."""

if not conversation_history or conversation_history[0]['role'] != 'system':
    conversation_history.insert(0, {'role':'system','content':PROMPT})

CHANNELS=1
SAMPLE_RATE=16000
CHUNK=1024

SILENCE_THRESHOLD=500
SILENCE_LIMIT=3

def jarvis_voice(text):
    audio_queue=Queue()

    def generate():
        sample,sample_rate=kokoro.create(
            text,
            speed=1.2,
            voice="bm_lewis",
            lang="en-us"
        )
        audio_queue.put((sample.astype("float32"), sample_rate))
        audio_queue.put(None)
    
    def play():
        while True:
            item=audio_queue.get()
            if item is None:
                break
            sample,sample_rate=item
            sd.play(sample,samplerate=sample_rate,device=7)
            sd.wait()
    
    gen_thread=threading.Thread(target=generate)
    play_thread=threading.Thread(target=play)

    gen_thread.start()
    play_thread.start()

    gen_thread.join()
    play_thread.join()

def audio_recording():
    audio=pyaudio.PyAudio()

    stream=audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
        input=True
    )

    frames=[]
    silence_time_start=None

    print("Start talking ...")

    while True:
        data=stream.read(CHUNK,exception_on_overflow=False)
        frames.append(data)

        volume=np.abs(np.frombuffer(data, dtype=np.int16)).mean()

        if volume < SILENCE_THRESHOLD:
            if silence_time_start is None:
                silence_time_start=time.time()
            elif (time.time() - silence_time_start)> SILENCE_LIMIT:
                break
        else:
            silence_time_start = None
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    tmp=tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
    wf=wave.open(tmp.name,'wb')
    wf.setnchannels(CHANNELS)
    wf.setframerate(SAMPLE_RATE)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.writeframes(b''.join(frames))
    wf.close()

    return tmp.name

with open("/home/abdelali/Downloads/my_voiceprint_1.pkl","rb") as f:
    voice_1=pickle.load(f)

with open("/home/abdelali/Downloads/my_voiceprint_2.pkl","rb") as f:
    voice_2=pickle.load(f)

with open("/home/abdelali/Downloads/my_voiceprint_3.pkl","rb") as f:
    voice_3=pickle.load(f)

VOICE=(voice_1+voice_2+voice_3)/3

def verify_voice(voice_path):
    wave_path=preprocess_wav(Path(voice_path))
    embedded=encoder.embed_utterance(wave_path)
    similarity=np.dot(embedded,VOICE)
    print(f"Similarity:{similarity:.2f}")
    return similarity >=0.75

def transcribe(audio_path):
    result=Whisper.transcribe(audio_path,language="en")
    text=str(result['text']).strip()
    os.remove(audio_path)
    
    return text

def jarvis_brain(user_input):
    global conversation_history

    conversation_history.append({
        'role':'user',
        'content':user_input
    })

    answer=ollama.chat(
        messages=conversation_history,
        model='qwen2.5:7b',
        stream=True,
        think=False
    )

    full_response=""
    for chunk in answer:
        if chunk['message']['content']:
            full_response+=chunk['message']['content']
    
    full_response=re.sub(r'/\w+', '', full_response).strip()
    
    try:
        clean=re.sub(r"```json|```", "", full_response).strip()
        decision=json.loads(clean)
    except:
        decision={'action':'chat','params':{},'speak':full_response}
    
    conversation_history.append({
        'role':'assistant',
        'content':full_response
    })

    if len(conversation_history)>21:
        conversation_history=[conversation_history[0]]+conversation_history[-20:]

    with open("jarvis_conversation.json",'w') as f:
        json.dump(conversation_history,f)

    return decision

LINKS={
    'youtube': {
        'base_url': 'https://www.youtube.com/results?search_query=', 
        'default_url': 'https://www.youtube.com'
        },
    
    'gmail':  {
        'base_url': None, 
        'default_url': 'https://mail.google.com'
        },
    
    'github': {
        'base_url': None, 
        'default_url': 'https://github.com/Abdelali516'
        }
}

windows={}
def open_links(app,query=None):
    if not app or app not in LINKS:
        return False
    
    config=LINKS[app]

    if query and config['base_url']:
        url=config['base_url'] + query.replace(" ","+")
    else:
        url=config['default_url']

    subprocess.Popen(['brave','--app='+url],stderr=subprocess.DEVNULL)

    if app=='youtube':
        subprocess.run(['xdotool','mousemove','350','350'])
        time.sleep(3)
        subprocess.run(['xdotool','click','1'])
    
    WINDOW_TITLES = {
    'youtube': 'YouTube',
    'gmail':   'Gmail',
    'github':  'GitHub'
    }

    target_id=WINDOW_TITLES.get(app," ")
    window_id = None

    for i in range(20):
        time.sleep(0.5)

        try:
            result=subprocess.run([
                'xdotool','search','--name',target_id
            ],capture_output=True, text=True)
            ids=result.stdout.strip().splitlines()
            if ids:
                window_id=ids[-1]
                break
        except Exception:
            continue

    if window_id:
        windows[app]=window_id

def close_link(app):
    if app in windows and windows[app]:
        subprocess.run(['xdotool','windowclose',windows[app]])
        del windows[app]
    else:
        return False
    
def lock_screen():
    subprocess.run(['loginctl','lock-session'])
    return True

def pause_continue(app):
    if windows[app]:
        subprocess.run(['xdotool','windowactivate','--sync',windows[app]])
        subprocess.run(['xdotool','key','space'])
    else:
        return False

def set_volume(level):
    level=max(0,min(100,level))
    subprocess.run(['pactl','set-sink-volume','@DEFAULT_SINK@',f'{level}%'])
    return True

def set_brightness(level):
    level=max(0,min(100,level))
    subprocess.run(['brightnessctl','set',f'{level}%'])
    return True

def open_apps(name):
    subprocess.Popen([name],stderr=subprocess.DEVNULL)


def check_battery():
    battery=psutil.sensors_battery()
    if battery:
         return f'Battery is at {int(battery.percent)} percent'
    else:
        return False

def check_time():
    return time.strftime("It's %H:%M on %A, %d %B")

def screenshot(user_question=None):
    Time = time.strftime("%Y%m%d-%H%M%S")
    path = f'/home/abdelali/Pictures/Jarvis/{Time}.png'
    subprocess.run(['scrot', path])
    time.sleep(0.5)
    
    with open(path,'rb')as f:
        image_data=base64.b64encode(f.read()).decode('utf-8')
    
    question = f'{user_question}. Answer in one short natural sentence, no markdown.' if user_question else 'Describe what is on this screen in one short sentence, no markdown.'

    response = ollama.chat(
        messages=[{
            'role': 'user',
            'content': question,
            'images': [image_data]
        }],
        model='minicpm-v',
        stream=True,
        think=False
    )

    full_response = ""
    for chunk in response:
        if chunk['message']['content']:
            full_response += chunk['message']['content']
    
    return full_response

def run_command(command):
    result=subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    return result.stdout or result.stderr

def summarize_output(output,user_question):
    response=ollama.chat(
        messages=[{'role':'user',
                  'content':f'Sir asked for {user_question }\nThe output was:\n{output}\n\nSummarize this in one short natural sentence a voice assistant would say and taking from the output only what will answer for the user question. No markdown, no symbols, just plain speech.'
            }],
        model='qwen3:1.7b',
        stream=True,
        think=False
    )

    full_response=""
    for chunk in response:
        if chunk['message']['content']:
            answer=chunk['message']['content']
            full_response+=answer

    return full_response

def wifi_control(state):
    subprocess.run(['nmcli','radio','wifi',state])
    return True

SEARCH_DIRS = [
    '/home/abdelali/Downloads',
    '/home/abdelali/Desktop',
    '/home/abdelali/Documents',
    '/home/abdelali/Pictures',
    '/home/abdelali/Videos',
    '/home/abdelali/Music',
]

def search_file(file_name):
    results=[]

    for directory in SEARCH_DIRS:
        result=subprocess.run([
            'find',directory,'-name',f'*{file_name}*','-type','f'
        ],capture_output=True,text=True)
    
        if result.stdout:
            results.extend(result.stdout.strip().splitlines())
    
    return results

def open_file(file_path):
    subprocess.Popen(['xdg-open',file_path],stdout=subprocess.DEVNULL)

def delete_file(file_path):
   send2trash.send2trash(file_path)


def web_search(query):
    with DDGS() as ddgs:
        results=list(ddgs.text(query,max_results=3,backend='api'))
        
        if not results:
            return False
    
    return ' '.join(r ['body'] for r in results)

def remind_me(message,minutes):
    def reminder():
        time.sleep(minutes*60)
        jarvis_voice(f'Sir, reminder: {message}')
    
    thread=threading.Thread(target=reminder,daemon=True)
    thread.start()

    return True

def open_vscode(folder=None):
    if folder:
        result=search_file(folder)
        if result:
            subprocess.Popen(['code',result[0]],stderr=subprocess.DEVNULL)
        else:
            return False
    else:
        subprocess.Popen(['code'],stderr=subprocess.DEVNULL)
    
    return True

def clipboard(action,text=None):
    if action=='copy' and text:
        subprocess.run(['xclip','--selection','clipboard'],input=text.encode())
        return 'Copied to clipboard Sir'
    elif action=='read':
        result=subprocess.run(['xclip','--selection','clipboard','-o'],capture_output=True,text=True)
        return result.stdout.strip() or 'Clipboard is empty Sir'
    elif action=='paste':
        subprocess.run(['xdotool','key','ctrl+v'])
        return 'Pasted Sir!'
    else:
        return False

def translate(text, language):
    response = ollama.chat(
        messages=[{
            'role': 'user',
            'content': f'Translate this to {language}: "{text}". Reply with only the translation, no explanation, no markdown.'
        }],
        model='qwen2.5:7b',
        stream=True,
        think=False
    )

    full_response = ""
    for chunk in response:
        if chunk['message']['content']:
            full_response += chunk['message']['content']
    
    return full_response

def take_note(note):
    Time = time.strftime("%Y-%m-%d %H:%M:%S")
    with open('jarvis_notes.txt', 'a') as f:
        f.write(f'-[{Time}]: {note}\n')
    return 'Note saved Sir'

with open("/home/abdelali/Downloads/emails_contact.json",'r')as file:
        CONTACTS=json.load(file)

try:
    with open("Email.txt",'r') as f:
        EMAIL=f.read()
except FileNotFoundError:
    with open("Email",'w')as f:
        f.write("abdelalielasbi516@gmail.com")
try:
    with open("Email_password",'r') as f:
        PASSWORD=f.read()
except FileNotFoundError:
    with open("Email_password",'w') as f:
        f.write("xnlgnxcnoisxgjuf")

def send_email(to, subject, body):
    try:
        to_clean = to.lower().split('@')[0].strip()
        resolved = CONTACTS.get(to_clean, to)  

        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = resolved
        msg['Subject'] = subject
        msg['Date']=formatdate(localtime=True)
        msg['Message-ID']=f'<{uuid.uuid4()}@gmail.com>'
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, resolved, msg.as_string())
        server.quit()

        return f"Email sent to {to_clean} Sir!"
    except Exception as e:
        return f"Couldn't send email: {e}"

def git_commit(message="auto commit"):
    path = "/home/abdelali/Desktop"
    
    subprocess.run(['git', '-C', path, 'add', 'jarvis final.py'])
    subprocess.run(['git', '-C', path, 'commit', '-m', message])
    subprocess.run(['git', '-C', path, 'push'])
    
    return f"Committed and pushed Sir!"

def execute_actions(action,params,speak,user_input, is_owner):
    print(f"Action: {action}, Params: {params}")
    actions = {
        'open_youtube':  lambda: open_links('youtube', params.get('query')),
        'open_github':   lambda: open_links('github'),
        'open_gmail':    lambda: open_links('gmail'),
        'close_youtube': lambda: close_link('youtube'),
        'close_github':  lambda: close_link('github'),
        'close_gmail':   lambda: close_link('gmail'),
        'pause_music':   lambda: pause_continue('youtube'),
        'continue_music':lambda: pause_continue('youtube'),
        'set_volume':    lambda: set_volume(params.get('level', 50)),
        'set_brightness':lambda: set_brightness(params.get('level', 50)),
        'open_app':      lambda: open_apps(params.get('name')),
        'wifi_on':       lambda: wifi_control('on'),
        'remember':      lambda: save_memory(params.get('key'), params.get('value')),
        'forget':        lambda: forget_memory(params.get('key')),
        'send_email':    lambda: send_email(params.get('to'),params.get('subject'),params.get('body')),
        'wifi_off':      lambda: wifi_control('off'),
        'lock_screen':   lambda: (jarvis_voice(speak), lock_screen(), exit()),
        'Goodbye':       lambda: (jarvis_voice(speak), exit()),
        'screenshot':    lambda: screenshot(params.get('question')),
        'tell_time':     lambda: check_time(),
        'git_commit':    lambda: git_commit(params.get('message', 'auto commit')),
        'battery':       lambda: check_battery(),
        'run_command':   lambda: summarize_output(run_command(params.get('command')), user_input),
        'web_search':    lambda: summarize_output(web_search(params.get('query')), user_input),
        'search_file':   lambda: [open_file(p) for p in search_file(params.get('filename'))],
        'delete_file':   lambda: [delete_file(p) for p in search_file(params.get('filename'))],
        'remind_me':     lambda: remind_me(params.get('message'),params.get('minutes')),
        'open_vscode':   lambda: open_vscode(params.get('folder')),
        'clipboard':     lambda: clipboard(params.get('action'), params.get('text')),
        'translate':     lambda: translate(params.get('text'), params.get('language')),
        'take_note':     lambda: take_note(params.get('note'))
    }

    if not is_owner:
        jarvis_voice("Sorry but i only talk to Sir!")
        return True
    
    if action in actions:
        result=actions[action]()
        if result and isinstance(result,str):
            speak=result
    
    jarvis_voice(speak)
    print("Jarvis:",speak)
    return True

jarvis_voice("Welcome Back Sir!")

while True:
    audio=audio_recording()

    is_owner=verify_voice(audio)
    
    user_text=transcribe(audio)
    print("Abdelali:",user_text)

    if not user_text:
        continue

    
    decision = jarvis_brain(user_text)

    action=decision.get('action','chat')
    params=decision.get('params',{})
    speak=decision.get('speak','')
        
    if not execute_actions(action,params,speak,user_text,is_owner):
        break


    
