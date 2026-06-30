from kokoro_onnx import Kokoro
import whisper
from queue import Queue
import sounddevice as sd
import threading
import pyaudio
import numpy as np
import time
import tempfile
from anton_hud import hud_state, start_hud_thread
start_hud_thread()
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
Whisper=whisper.load_model('base',device='cpu')
hud_state.set_mode('listening')
encoder=VoiceEncoder()

try:
    with open("jarvis_conversation.json", 'r') as f:
        conversation_history = json.load(f)
except FileNotFoundError:
    conversation_history = []

CHANNELS=1
SAMPLE_RATE=16000
CHUNK=1024

SILENCE_THRESHOLD=500
SILENCE_LIMIT=3

def jarvis_voice(text):
    if not text:
        return False
    
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
            item = audio_queue.get()
            if item is None:
                break
            sample, sample_rate = item
            hud_state.set_mode('speaking')

            pos = 0
            chunk_size = int(sample_rate * 0.05)  # 50ms, just for amplitude granularity

            def callback(outdata, frames, time_info, status):
                nonlocal pos
                end = pos + frames
                chunk = sample[pos:end]

                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:, 0] = chunk

                hud_state.set_amplitude(np.abs(chunk).mean() * 8)
                pos = end

            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                callback=callback,
                blocksize=chunk_size,
                device=7
            )
            with stream:
                while stream.active:
                    sd.sleep(50)

            hud_state.set_mode('idle')
    
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
        hud_state.set_amplitude(volume / 3000) 

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
        model='jarvis',
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
        time.sleep(5)
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
    name=name.lower()
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
    
    if user_question: 
        response = ollama.chat(
            messages=[{
                'role': 'user',
                'content': user_question,
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

    else:
        jarvis_voice('Done Sir!')
    
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
        model='qwen2.5:7b',
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
    
    subprocess.run(['git', '-C', path, 'add', '-A'], capture_output=True, text=True)
    r = subprocess.run(['git', '-C', path, 'commit', '-m', message], capture_output=True, text=True)
    
    if 'nothing to commit' in r.stdout:
        return "Nothing new to commit Sir"
    
    subprocess.run(['git', '-C', path, 'push', '--force'])
    
    return "Committed and pushed Sir!"

def execute_actions(action,params,speak,user_input):
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
        'take_note':     lambda: take_note(params.get('note')),
        'chat':          lambda: speak
    }

    
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
    user_text=transcribe(audio)


    if not user_text:
        continue
    
    print("Abdelali:",user_text)



    decision = jarvis_brain(user_text)
    
    action=decision.get('action','chat')
    params=decision.get('params',{})
    speak=decision.get('speak','')
        
    if not execute_actions(action,params,speak,user_text):
        break


    
