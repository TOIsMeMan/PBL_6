"""
DeepSpeech2 Transcription Service
Tích hợp model DeepSpeech2 để tách text từ video
"""

import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2CTCTokenizer
import tempfile
import json
import traceback
from typing import Dict, List, Optional

# Thêm path để import từ simple_test.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

try:
    import moviepy.editor as mp
    VIDEO_SUPPORT = True
except:
    VIDEO_SUPPORT = False
    print("⚠️ MoviePy not available - video extraction may fail")


class TranscriptionService:
    """Service để xử lý transcription từ video/audio"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.audio2mels = None
        self.amp2db = None
        self._load_model()
    
    def _load_model(self):
        """Load DeepSpeech2 model"""
        try:
            # Import model classes từ simple_test.py
            from simple_test import DeepSpeech2
            
            model_path = os.path.join(os.path.dirname(__file__), "../../../best_weights_finetuned1.pt")
            
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
            self.model = DeepSpeech2()
            self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.audio2mels = T.MelSpectrogram(sample_rate=16000, n_mels=80)
            self.amp2db = T.AmplitudeToDB(top_db=80.0)
            
            print(f"✅ DeepSpeech2 model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            traceback.print_exc()
            raise e
    
    def _extract_audio_from_video(self, video_path: str) -> tuple:
        """Extract audio từ video file"""
        if not VIDEO_SUPPORT:
            raise ImportError("MoviePy required for video processing")
        
        try:
            video = mp.VideoFileClip(video_path)
            audio_clip = video.audio
            
            if audio_clip is None:
                raise ValueError("No audio track found in video")
            
            # Save to temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_audio_path = tmp_file.name
            
            audio_clip.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Load audio
            audio, orig_sr = torchaudio.load(temp_audio_path, normalize=True)
            
            # Cleanup
            video.close()
            audio_clip.close()
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            return audio, orig_sr
        except Exception as e:
            print(f"❌ Error extracting audio: {str(e)}")
            raise e
    
    def _preprocess_audio(self, audio_path: str) -> tuple:
        """Preprocess audio file"""
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        # Load audio based on file type
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            audio, orig_sr = self._extract_audio_from_video(audio_path)
        else:
            audio, orig_sr = torchaudio.load(audio_path, normalize=True)
        
        # Resample to 16kHz
        if orig_sr != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=16000)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        audio_duration = audio.shape[1] / 16000
        return audio, audio_duration
    
    def _get_word_timestamps(self, decoded_text: str, audio_duration: float) -> List[Dict]:
        """Tính timestamps cho mỗi từ"""
        words = decoded_text.split()
        if not words:
            return []
        
        total_chars = sum(len(word) for word in words)
        word_timestamps = []
        current_time = 0.0
        
        for word in words:
            word_duration = (len(word) / total_chars) * audio_duration if total_chars > 0 else 0
            word_timestamps.append({
                'word': word,
                'start': current_time,
                'end': current_time + word_duration
            })
            current_time += word_duration
        
        return word_timestamps
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Tách text thành các câu"""
        import re
        sentences = re.split(r'([.!?])\s+', text)
        
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                sentence = (sentences[i] + sentences[i+1]).strip()
                if sentence:
                    result.append(sentence)
        
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result if result else [text]
    
    def _align_sentences_with_timestamps(
        self, 
        original_text: str, 
        corrected_text: str, 
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """Ánh xạ câu đã sửa với timestamps"""
        sentences = self._split_into_sentences(corrected_text)
        
        if not sentences or not word_timestamps:
            return [{
                'text': corrected_text,
                'start': word_timestamps[0]['start'] if word_timestamps else 0,
                'end': word_timestamps[-1]['end'] if word_timestamps else 0
            }]
        
        result = []
        word_index = 0
        total_words = len(word_timestamps)
        
        for sentence in sentences:
            sentence_word_count = len(sentence.replace(',', '').replace('"', '').replace(':', '').split())
            start_idx = word_index
            end_idx = min(word_index + sentence_word_count, total_words)
            
            if start_idx < total_words:
                start_time = word_timestamps[start_idx]['start']
                end_time = word_timestamps[min(end_idx - 1, total_words - 1)]['end']
                
                result.append({
                    'text': sentence,
                    'start': round(start_time, 3),
                    'end': round(end_time, 3)
                })
                
                word_index = end_idx
        
        return result
    
    def transcribe_video(
        self, 
        video_path: str,
        use_correction: bool = True,
        gemini_api_key: Optional[str] = None
    ) -> Dict:
        """
        Transcribe video/audio thành text với timestamps
        
        Returns:
            {
                'transcript': str,  # Full text
                'timestamps': List[Dict],  # [{text, start, end}, ...]
                'confidence': float,
                'duration': float
            }
        """
        try:
            # Preprocess audio
            audio, audio_duration = self._preprocess_audio(video_path)
            
            # Convert to mel spectrogram
            mel = self.audio2mels(audio)
            mel = self.amp2db(mel)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            mel = mel.unsqueeze(0)
            src_len = torch.tensor([mel.shape[-1]])
            
            # AI inference
            with torch.no_grad():
                pred_logits, _ = self.model(mel.to(self.device), src_len)
                probabilities = torch.softmax(pred_logits, dim=-1)
                confidence = probabilities.max(dim=-1)[0].mean().item()
            
            # Decode prediction
            pred_tokens = pred_logits.squeeze().argmax(axis=-1).tolist()
            transcript = self.tokenizer.decode(pred_tokens)
            
            # Get word timestamps
            word_timestamps = self._get_word_timestamps(transcript, audio_duration)
            
            # Optional: Gemini correction
            corrected_text = transcript
            if use_correction and gemini_api_key:
                try:
                    from simple_test import correct_with_gemini
                    corrected_text = correct_with_gemini(transcript, gemini_api_key)
                except Exception as e:
                    print(f"⚠️ Gemini correction failed: {str(e)}")
            
            # Align sentences with timestamps
            sentence_timestamps = self._align_sentences_with_timestamps(
                transcript, 
                corrected_text, 
                word_timestamps
            )
            
            return {
                'transcript': corrected_text,
                'timestamps': sentence_timestamps,
                'confidence': round(confidence, 3),
                'duration': round(audio_duration, 2)
            }
            
        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            traceback.print_exc()
            raise e


# Singleton instance
_transcription_service = None

def get_transcription_service() -> TranscriptionService:
    """Get or create transcription service instance"""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service
