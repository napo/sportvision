from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import uvicorn
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips, TextClip, CompositeVideoClip
from ultralytics import YOLO
from track import track_ball, track_players  # Importa le funzioni di tracking
import shutil
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = "videos"
CLIPS_DIRECTORY = "clips"
VIDEOS_AI_DIRECTORY = "videos-ai"
PROCESSED_VIDEO_DIRECTORY = "processed_videos"  # Directory per i video processati

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

if not os.path.exists(CLIPS_DIRECTORY):
    os.makedirs(CLIPS_DIRECTORY)

if not os.path.exists(VIDEOS_AI_DIRECTORY):
    os.makedirs(VIDEOS_AI_DIRECTORY)

if not os.path.exists(PROCESSED_VIDEO_DIRECTORY):
    os.makedirs(PROCESSED_VIDEO_DIRECTORY)

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...), shortcuts: str = Form(...), duration: int = Form(...)):
    try:
        # Salva il file video caricato
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        
        # Salva i dati dei shortcuts in un file JSON
        shortcuts_data = json.loads(shortcuts)
        with open("shortcuts.json", "w") as f:
            json.dump(shortcuts_data, f, indent=4)
        
        # Carica il file video
        video_clip = VideoFileClip(file_location)
        clips = []

        # Crea i subclip e aggiungi il testo
        for shortcut in shortcuts_data:
            start_time = float(shortcut['time'])
            end_time = start_time + duration  # Usa la durata inviata dal client
            clip = video_clip.subclip(start_time, min(end_time, video_clip.duration))

            text = f"{shortcut['title']} - {shortcut['description']}"
            txt_clip = (TextClip(text, fontsize=50, font='Amiri-Bold', color='white', bg_color='#5cb9ff')
                        .set_position(("right", "top"))
                        .set_duration(clip.duration))

            video = CompositeVideoClip([clip, txt_clip])
            clips.append(video)

        # Concatenazione dei subclip
        final_clip = concatenate_videoclips(clips)
        final_clip_location = os.path.join(CLIPS_DIRECTORY, "final_clip.mp4")
        final_clip.write_videofile(final_clip_location, codec="libx264")

        # Restituisce il file video finale
        return FileResponse(final_clip_location, media_type='video/mp4', filename="final_clip.mp4")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

@app.post("/ai-video/")
async def ai_video(file: UploadFile = File(...), recognitionOption: str = Form(...)):
    try:
        # Salva il file caricato
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        
        # Carica il modello YOLO
        model = YOLO('weights/players_2.pt')
        confidence = 0.3

        # Seleziona la funzionalità appropriata in base alla scelta dell'utente
        if recognitionOption == "Ball Recognition":
            final_clip_location = "processed_videos/ball.mp4"
            # track_ball(model, file_location, confidence)
        elif recognitionOption == "Player Recognition":
            final_clip_location = "processed_videos/player.mp4"
            # track_players(model, file_location, confidence)
        else:
            return JSONResponse(status_code=400, content={"message": "Invalid recognition option selected."})

        # Duplica il video processato e salva come final.mp4
        # processed_file_location = os.path.join(PROCESSED_VIDEO_DIRECTORY, "output.mp4")
        # final_clip_location = os.path.join(PROCESSED_VIDEO_DIRECTORY, "final.mp4")
        
        # video_clip = VideoFileClip(processed_file_location)
        # video_clip.write_videofile(final_clip_location, codec="libx264")
        
        # Restituisce il file video finale
        return FileResponse(final_clip_location, media_type='video/mp4', filename="final.mp4")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
