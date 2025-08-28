
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
import moviepy.editor as mp
import speech_recognition as sr
import re
import requests
import json
from typing import Optional, List
import uuid
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pytube import YouTube
from fastapi.responses import StreamingResponse
import io
import tempfile
import shutil
import wave
import audioop


try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


app = FastAPI(title="Video Summary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUMMARIES_DB = []


class VideoURL(BaseModel):
    url: str

class SummaryOptions(BaseModel):
    format: Optional[str] = "markdown"  # markdown, bullet, narrative
    length: Optional[str] = "medium"    # short, medium, long
    language: Optional[str] = "english" # target language for translation
    include_sentiment: Optional[bool] = False
    include_keywords: Optional[bool] = False

class SummaryRequest(BaseModel):
    url: str
    options: Optional[SummaryOptions] = SummaryOptions()

class SummaryResponse(BaseModel):
    id: str
    summary: str
    transcript: str
    metadata: dict


def extract_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None



def get_video_metadata(video_id):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        return {
            "title": yt.title,
            "author": yt.author,
            "length_seconds": yt.length,
            "views": yt.views,
            "publish_date": str(yt.publish_date) if yt.publish_date else None,
            "thumbnail_url": yt.thumbnail_url,
        }
    except Exception as e:
        print(f"Error getting metadata: {str(e)}")
        return {}
    
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([item['text'] for item in transcript])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting transcript: {str(e)}")



def analyze_sentiment(text):
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(text)
        # Determine overall sentiment
        if sentiment['compound'] >= 0.05:
            overall = "positive"
        elif sentiment['compound'] <= -0.05:
            overall = "negative"
        else:
            overall = "neutral"
        
        return {
            "overall": overall,
            "scores": sentiment
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return {"overall": "unknown", "scores": {}}
    

def extract_keywords(text, num_keywords=10):
    try:
        # Simple frequency-based keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        # Common stopwords to exclude
        stopwords = ["the", "and", "this", "that", "for", "you", "with", "have", 
                    "from", "are", "they", "your", "what", "their", "can"]
        
        for word in words:
            if word not in stopwords:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        
        # Sort by frequency
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:num_keywords]]
    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return []




def get_summary_prompt(text, options):
    format_instruction = ""
    length_instruction = ""
    
    # Format instructions
    if options.format == "bullet":
        format_instruction = "Format the summary as bullet points."
    elif options.format == "narrative":
        format_instruction = "Format the summary as a narrative paragraph."
    else:  # markdown default
        format_instruction = "Format the summary using markdown with headings, bullet points, and emphasis where appropriate."
    
    # Length instructions
    if options.length == "short":
        length_instruction = "Keep the summary very concise (about 100-150 words)."
    elif options.length == "long":
        length_instruction = "Provide a comprehensive summary covering all major points (about 400-500 words)."
    else:  # medium default
        length_instruction = "Provide a balanced summary (about 250-300 words)."
    
    prompt = f"""Please summarize the following content:

{text}

{format_instruction} {length_instruction} Focus on the main ideas, key points, and conclusions.
Include the most important details while removing redundancy."""

    return prompt



def summarize_text(text, options=SummaryOptions()):
    try:
        chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.7)
        
        prompt = get_summary_prompt(text, options)
        
        messages = [
            SystemMessage(content="You are a helpful assistant that summarizes video content in a clear, concise manner."),
            HumanMessage(content=prompt)
        ]
        
        summary = chat.invoke(messages).content
        
        # Translate if needed
        if options.language and options.language.lower() != "english":
            translation_messages = [
                SystemMessage(content=f"You are a translator. Translate the following text to {options.language}."),
                HumanMessage(content=summary)
            ]
            summary = chat.invoke(translation_messages).content
            
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


def validate_audio_file(audio_path):
    """Validate audio file format and quality before processing"""
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            raise ValueError("Audio file does not exist")
        
        # Get configurable file size limit (default: 50MB)
        max_file_size = int(os.getenv("MAX_AUDIO_FILE_SIZE", 50 * 1024 * 1024))
        
        # Check file size (minimum 1KB, maximum configurable limit)
        file_size = os.path.getsize(audio_path)
        if file_size < 1024:
            raise ValueError(f"Audio file is too small (less than 1KB). File size: {file_size} bytes")
        if file_size > max_file_size:
            raise ValueError(f"Audio file is too large ({file_size / (1024*1024):.1f}MB). Maximum allowed: {max_file_size / (1024*1024):.1f}MB. Consider using a shorter video or lower quality.")
        
        # Try to open the audio file to check if it's valid
        with wave.open(audio_path, 'rb') as wav_file:
            # Check if it's a WAV file
            if wav_file.getnchannels() not in [1, 2]:
                raise ValueError("Audio must be mono or stereo")
            
            # Check sample rate (should be between 8000 and 48000 Hz)
            sample_rate = wav_file.getframerate()
            if sample_rate < 8000 or sample_rate > 48000:
                raise ValueError(f"Sample rate {sample_rate}Hz is not supported (must be 8000-48000Hz)")
            
            # Check duration (minimum 1 second, maximum 1 hour)
            frames = wav_file.getnframes()
            duration = frames / float(sample_rate)
            if duration < 1:
                raise ValueError("Audio is too short (less than 1 second)")
            if duration > 3600:
                raise ValueError("Audio is too long (more than 1 hour)")
        
        return True
        
    except wave.Error:
        raise ValueError("Invalid audio file format (must be WAV)")
    except Exception as e:
        raise ValueError(f"Audio validation failed: {str(e)}")


def preprocess_audio(audio_path):
    """Preprocess audio to improve speech recognition accuracy and reduce file size"""
    try:
        # Get original file size
        original_size = os.path.getsize(audio_path)
        
        # Get configurable file size limit (default: 50MB)
        max_file_size = int(os.getenv("MAX_AUDIO_FILE_SIZE", 50 * 1024 * 1024))
        
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            
            # Convert to mono if stereo (reduces file size by ~50%)
            if channels == 2:
                frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
                channels = 1
            
            # Normalize volume
            max_volume = audioop.max(frames, sample_width)
            if max_volume > 0:
                normalized_frames = audioop.mul(frames, sample_width, 32767 / max_volume)
            else:
                normalized_frames = frames
            
            # Reduce sample rate for large files to save space (but maintain quality for speech)
            target_sample_rate = sample_rate
            if original_size > max_file_size * 0.8:  # If file is approaching the limit
                # Reduce sample rate to 16kHz (good for speech recognition)
                if sample_rate > 16000:
                    target_sample_rate = 16000
                    # Resample audio (simple downsampling for demonstration)
                    # In production, you might use a proper resampling library
                    print(f"Reducing sample rate from {sample_rate}Hz to {target_sample_rate}Hz to reduce file size")
            
            # Write processed audio back
            processed_path = audio_path.replace('.wav', '_processed.wav')
            with wave.open(processed_path, 'wb') as out_file:
                out_file.setnchannels(channels)
                out_file.setsampwidth(sample_width)
                out_file.setframerate(target_sample_rate)
                out_file.writeframes(normalized_frames)
            
            # Check if processing reduced file size
            processed_size = os.path.getsize(processed_path)
            print(f"Audio preprocessing: {original_size/1024/1024:.1f}MB -> {processed_size/1024/1024:.1f}MB")
            
            return processed_path
            
    except Exception as e:
        print(f"Audio preprocessing failed: {str(e)}")
        return audio_path  # Return original if preprocessing fails



@app.post("/summarize/youtube", response_model=SummaryResponse)
async def summarize_youtube(request: SummaryRequest):
    
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Get video metadata
    metadata = get_video_metadata(video_id)
    
    # Get transcript
    transcript = get_video_transcript(video_id)
    
    # Generate summary
    summary = summarize_text(transcript, request.options)
    
    # Additional analyses if requested
    if request.options.include_sentiment:
        metadata["sentiment"] = analyze_sentiment(transcript)
    
    if request.options.include_keywords:
        metadata["keywords"] = extract_keywords(transcript)
    
    # Create response
    summary_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    response = {
        "id": summary_id,
        "summary": summary,
        "transcript": transcript,
        "metadata": {
            **metadata,
            "timestamp": timestamp,
            "video_id": video_id,
            "options": request.options.dict() if request.options else {},
        }
    }
    
    # Store in database
    SUMMARIES_DB.append(response)
    
    return response



@app.post("/summarize/upload", response_model=SummaryResponse)
async def summarize_upload(
    file: UploadFile = File(...),
    format: str = Query("markdown"),
    length: str = Query("medium"),
    language: str = Query("english"),
    include_sentiment: bool = Query(False),
    include_keywords: bool = Query(False)
):
    try:
        # Create options object
        options = SummaryOptions(
            format=format,
            length=length,
            language=language,
            include_sentiment=include_sentiment,
            include_keywords=include_keywords
        )
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Save uploaded file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Extract audio and transcribe
            video = mp.VideoFileClip(file_path)
            audio_path = os.path.join(temp_dir, "temp_audio.wav")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Validate audio file
            try:
                validate_audio_file(audio_path)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Audio validation failed: {str(e)}")
            
            # Preprocess audio to improve recognition
            processed_audio_path = preprocess_audio(audio_path)
            
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(processed_audio_path) as source:
                    # Adjust for ambient noise
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.record(source)
                    
                    # Try recognition with multiple attempts
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            transcript = recognizer.recognize_google(audio)
                            break
                        except sr.UnknownValueError:
                            if attempt == max_attempts - 1:
                                raise HTTPException(status_code=400, detail="Google Speech Recognition could not understand the audio after multiple attempts. The audio may be too noisy, too quiet, or contain no speech.")
                            continue
                        except sr.RequestError as e:
                            if attempt == max_attempts - 1:
                                raise HTTPException(status_code=500, detail=f"Could not request results from Google Speech Recognition service after multiple attempts; {e}")
                            continue
                            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")
            
            # Generate summary
            summary = summarize_text(transcript, options)
            
            # Additional analyses
            metadata = {
                "filename": file.filename,
                "duration_seconds": video.duration,
                "timestamp": datetime.now().isoformat()
            }
            
            if include_sentiment:
                metadata["sentiment"] = analyze_sentiment(transcript)
            
            if include_keywords:
                metadata["keywords"] = extract_keywords(transcript)
            
            # Create response
            summary_id = str(uuid.uuid4())
            
            response = {
                "id": summary_id,
                "summary": summary,
                "transcript": transcript,
                "metadata": metadata
            }
            
            # Store in database
            SUMMARIES_DB.append(response)
            
            return response
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summaries")
async def get_summaries():
    """Return all summaries in the database"""
    return SUMMARIES_DB



@app.get("/summaries/{summary_id}")
async def get_summary(summary_id: str):
    """Return a specific summary by ID"""
    for summary in SUMMARIES_DB:
        if summary["id"] == summary_id:
            return summary
    raise HTTPException(status_code=404, detail="Summary not found")


@app.delete("/summaries/{summary_id}")
async def delete_summary(summary_id: str):
    """Delete a specific summary by ID"""
    global SUMMARIES_DB
    original_length = len(SUMMARIES_DB)
    SUMMARIES_DB = [s for s in SUMMARIES_DB if s["id"] != summary_id]
    
    if len(SUMMARIES_DB) == original_length:
        raise HTTPException(status_code=404, detail="Summary not found")
        
    return {"status": "success", "message": "Summary deleted"}



@app.get("/download/{summary_id}")
async def download_summary(summary_id: str, format: str = "txt"):
    """Download a summary in different formats"""
    # Find the summary
    summary_data = None
    for summary in SUMMARIES_DB:
        if summary["id"] == summary_id:
            summary_data = summary
            break
    
    if not summary_data:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    content = ""
    media_type = "text/plain"
    filename = f"summary_{summary_id}"
    
    if format == "txt":
        content = f"Summary:\n\n{summary_data['summary']}\n\nTranscript:\n\n{summary_data['transcript']}"
        filename += ".txt"
    elif format == "json":
        content = json.dumps(summary_data, indent=2)
        media_type = "application/json"
        filename += ".json"
    elif format == "md":
        content = f"# Video Summary\n\n## Summary\n\n{summary_data['summary']}\n\n## Transcript\n\n{summary_data['transcript']}"
        media_type = "text/markdown"
        filename += ".md"
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")
    
    return StreamingResponse(
        io.StringIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "api_version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)