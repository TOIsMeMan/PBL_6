import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2CTCTokenizer
from jiwer import wer, cer
import time
import requests
import json
import traceback

try:
    import sys
    if 'moviepy' in sys.modules:
        del sys.modules['moviepy']
    if 'moviepy.editor' in sys.modules:
        del sys.modules['moviepy.editor']

    import moviepy.editor as mp
    test_clip = mp.VideoFileClip.__name__
    VIDEO_SUPPORT = True
    print("✅ MoviePy loaded - Video files supported")

except Exception as e:
    VIDEO_SUPPORT = False
    print("⚠️ MoviePy import failed:", e)
    traceback.print_exc()
    print("   Try: python -m pip uninstall moviepy && python -m pip install moviepy")

# Model classes (copy từ deepspeech2.ipynb)
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True, **kwargs):
        super(MaskedConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, 
                                           kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, x, seq_lens):
        batch_size, channels, height, width = x.shape
        output_seq_lens = self._compute_output_seq_len(seq_lens)
        conv_out = super().forward(x)
        
        mask = torch.zeros(batch_size, output_seq_lens.max(), device=x.device)
        for i, length in enumerate(output_seq_lens):
            mask[i, :length] = 1
        mask = mask.unsqueeze(1).unsqueeze(1)
        conv_out = conv_out * mask
        
        return conv_out, output_seq_lens

    def _compute_output_seq_len(self, seq_lens):
        return torch.floor((seq_lens + (2 * self.padding[1]) - (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1

class ConvolutionFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(ConvolutionFeatureExtractor, self).__init__()
        self.in_channels = in_channels, 
        self.out_channels = out_channels
        
        self.conv1 = MaskedConv2d(in_channels, out_channels, kernel_size=(11, 41), stride=(2,2), padding=(5,20), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = MaskedConv2d(out_channels, out_channels, kernel_size=(11, 21), stride=(2,1), padding=(5,10), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.output_feature_dim = 20
        self.conv_output_features = self.output_feature_dim * self.out_channels
        
    def forward(self, x, seq_lens):
        x, seq_lens = self.conv1(x, seq_lens)
        x = self.bn1(x)
        x = torch.nn.functional.hardtanh(x)
        
        x, seq_lens = self.conv2(x, seq_lens)
        x = self.bn2(x)
        x = torch.nn.functional.hardtanh(x)
        
        x = x.permute(0,3,1,2).flatten(2)
        return x, seq_lens 

class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size = 512):
        super(RNNLayer, self).__init__()
        self.hidden_dim = hidden_size
        self.input_size = input_size
        
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                          batch_first=True, bidirectional=True)
        self.layernorm = nn.LayerNorm(2 * hidden_size)

    def forward(self, x, seq_lens):
        batch, seq_len, embed_dim = x.shape 
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)
        out, _ = self.rnn(packed_x)
        x, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=seq_len, batch_first=True)
        x = self.layernorm(x)
        return x

class DeepSpeech2(nn.Module):
    def __init__(self, conv_in_channels=1, conv_out_channels=32, rnn_hidden_size=512, rnn_depth=5):
        super(DeepSpeech2, self).__init__()
        
        self.feature_extractor = ConvolutionFeatureExtractor(conv_in_channels, conv_out_channels)
        self.output_hidden_features = self.feature_extractor.conv_output_features
        
        self.rnns = nn.ModuleList([
            RNNLayer(input_size=self.output_hidden_features if i==0 else 2 * rnn_hidden_size,
                     hidden_size=rnn_hidden_size)
            for i in range(rnn_depth)
        ])
        
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
        self.head = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, rnn_hidden_size), 
            nn.Hardtanh(), 
            nn.Linear(rnn_hidden_size, tokenizer.vocab_size)
        )

    def forward(self, x, seq_lens):
        x, final_seq_lens = self.feature_extractor(x, seq_lens)
        for rnn in self.rnns:
            x = rnn(x, final_seq_lens)
        x = self.head(x)
        return x, final_seq_lens

# ==========================================
# SIMPLE TESTER
# ==========================================

def load_audio_from_any_format(file_path, target_sr=16000):
    """Load audio từ bất kỳ format nào (mp4, mp3, wav, flac, etc.)"""
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Thử load trực tiếp bằng torchaudio trước (cho audio files)
        if file_ext in ['.wav', '.flac', '.mp3', '.ogg']:
            print(f"📁 Loading audio file: {file_ext}")
            audio, orig_sr = torchaudio.load(file_path, normalize=True)
            
        # Nếu là video file hoặc torchaudio fail
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm'] or not VIDEO_SUPPORT:
            if not VIDEO_SUPPORT:
                raise ImportError("MoviePy required for video files")
                
            print(f"🎬 Extracting audio from video: {file_ext}")
            # Extract audio using moviepy
            video = mp.VideoFileClip(file_path)
            
            # Extract audio
            audio_clip = video.audio
            if audio_clip is None:
                raise ValueError("No audio track found in video file")
            
            # Save to temporary wav file
            temp_audio_path = "temp_extracted_audio.wav"
            audio_clip.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Load the temporary file
            audio, orig_sr = torchaudio.load(temp_audio_path, normalize=True)
            
            # Cleanup
            video.close()
            audio_clip.close()
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        else:
            # Fallback: thử load bằng torchaudio anyway
            print(f"🔄 Trying to load unknown format: {file_ext}")
            audio, orig_sr = torchaudio.load(file_path, normalize=True)
    
    except Exception as e:
        print(f"❌ Error loading {file_ext} file: {str(e)}")
        if not VIDEO_SUPPORT and file_ext in ['.mp4', '.avi', '.mov']:
            print("💡 Tip: Install moviepy to handle video files:")
            print("   pip install moviepy")
        raise e
    
    # Resample if needed
    if orig_sr != target_sr:
        print(f"🔄 Resampling from {orig_sr}Hz to {target_sr}Hz")
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=target_sr)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        print("🔄 Converting stereo to mono")
        audio = audio.mean(dim=0, keepdim=True)
    
    print(f"✅ Audio loaded: {audio.shape[1]/target_sr:.2f}s duration")
    return audio, target_sr

def load_model():
    """Load trained model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    model = DeepSpeech2()
    model.load_state_dict(torch.load("best_weights1.pt", weights_only=True))
    model = model.to(device)
    model.eval()
    
    audio2mels = T.MelSpectrogram(sample_rate=16000, n_mels=80)
    amp2db = T.AmplitudeToDB(top_db=80.0)
    
    print(f"✅ Model loaded on {device}")
    return model, tokenizer, audio2mels, amp2db, device

def get_ground_truth(audio_path):
    """Tự động tìm ground truth từ file transcript"""
    audio_dir = os.path.dirname(audio_path)
    audio_filename = os.path.basename(audio_path)
    audio_id = os.path.splitext(audio_filename)[0]
    
    # Tìm file .txt trong cùng thư mục
    transcript_files = [f for f in os.listdir(audio_dir) if f.endswith('.txt')]
    
    for transcript_file in transcript_files:
        transcript_path = os.path.join(audio_dir, transcript_file)
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2 and parts[0] == audio_id:
                    return parts[1]
        except:
            continue
    
    return None

def correct_with_gemini(prediction, api_key):
    """Sửa prediction bằng Google Gemini API (Updated 2024)"""
    
    if not api_key:
        print("⚠️ No Gemini API key provided")
        return prediction
    
    try:
        # Sử dụng model mới gemini-2.0-flash
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": "AIzaSyDcqE4-ECdIbYChPXwu7Mg0KpgCaATab44"  # Sử dụng header mới
        }
        
        prompt = f"Fix grammar, spelling, and punctuation errors in this speech-to-text transcription while preserving the original meaning. Only return the corrected text: {prediction}"
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    corrected = candidate['content']['parts'][0]['text'].strip()
                    
                    # Clean up response
                    if corrected and corrected != prediction:
                        print(f"✅ Gemini 2.0 Flash success")
                        return corrected
            
            print("⚠️ No valid response from Gemini")
            return prediction
        else:
            print(f"❌ Gemini API error: {response.status_code}")
            if response.text:
                print(f"   Response: {response.text[:200]}")
            return prediction
            
    except Exception as e:
        print(f"❌ Error calling Gemini API: {str(e)}")
        return prediction

def translate_with_gemini(text, api_key, target_language="Vietnamese"):
    """Dịch text sang ngôn ngữ khác bằng Gemini API"""
    
    if not api_key:
        print("⚠️ No Gemini API key provided")
        return text
    
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": "AIzaSyDcqE4-ECdIbYChPXwu7Mg0KpgCaATab44"
        }
        
        prompt = f"""
        Translate the following English text to {target_language}. 
        Keep the same tone, context, and sentence length. Make it natural and fluent.
        
        Text to translate: "{text}"
        
        Translation:
        """
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    translated = candidate['content']['parts'][0]['text'].strip()
                    
                    # Clean up response
                    if "Translation:" in translated:
                        translated = translated.split("Translation:")[-1].strip()
                    
                    if translated and translated != text:
                        print(f"✅ Gemini translation success")
                        return translated
            
            print("⚠️ No valid translation from Gemini")
            return text
        else:
            print(f"❌ Gemini translation error: {response.status_code}")
            return text
            
    except Exception as e:
        print(f"❌ Error calling Gemini translation: {str(e)}")
        return text

def test_audio(audio_path, model, tokenizer, audio2mels, amp2db, device, 
               use_gemini=False, gemini_api_key=None, translate_to="Vietnamese"):
    """Test 1 file audio/video với tùy chọn sửa và dịch bằng Gemini"""
    
    print(f"🎵 Testing: {os.path.basename(audio_path)}")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    # ✨ LOAD AUDIO FROM ANY FORMAT
    try:
        start_time = time.time()
        audio, sample_rate = load_audio_from_any_format(audio_path, target_sr=16000)
        loading_time = time.time() - start_time
        print(f"⏱️  File loading time: {loading_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Failed to load audio: {str(e)}")
        return
    
    # Preprocess audio
    start_time = time.time()
    mel = audio2mels(audio)
    mel = amp2db(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = mel.unsqueeze(0)
    src_len = torch.tensor([mel.shape[-1]])
    
    # AI inference
    print("🤖 AI đang phân tích...")
    with torch.no_grad():
        pred_logits, _ = model(mel.to(device), src_len)
        probabilities = torch.softmax(pred_logits, dim=-1)
        confidence = probabilities.max(dim=-1)[0].mean().item()
    
    # Decode prediction
    pred_tokens = pred_logits.squeeze().argmax(axis=-1).tolist()
    ai_prediction = tokenizer.decode(pred_tokens)
    
    processing_time = time.time() - start_time
    audio_duration = audio.shape[1] / 16000
    
    # Tìm ground truth (chỉ cho LibriSpeech files)
    ground_truth = None
    if "LibriSpeech" in audio_path:
        ground_truth = get_ground_truth(audio_path)
    
    # Hiển thị kết quả gốc
    print(f"⏱️  Processing time: {processing_time:.3f}s")
    print(f"🎵 Audio duration: {audio_duration:.2f}s")
    print(f"📊 AI Confidence: {confidence:.3f}")
    print(f"")
    print(f"🎯 ORIGINAL PREDICTION: '{ai_prediction}'")
    
    # Gemini Correction
    corrected_prediction = ai_prediction
    if use_gemini and gemini_api_key:
        print("💎 Gemini đang sửa lỗi...")
        corrected_prediction = correct_with_gemini(ai_prediction, gemini_api_key)
        
        if corrected_prediction != ai_prediction:
            print(f"✨ GEMINI CORRECTED:    '{corrected_prediction}'")
            
            # Dịch sau khi correct
            if translate_to and translate_to.lower() != "english":
                print(f"🌐 Gemini đang dịch sang {translate_to}...")
                translated_text = translate_with_gemini(corrected_prediction, gemini_api_key, translate_to)
                print(f"🗣️ TRANSLATED TO {translate_to.upper()}: '{translated_text}'")
        else:
            print(f"⚠️ Gemini correction failed, keeping original")
    
    # So sánh với ground truth nếu có
    if ground_truth:
        print(f"")
        print(f"📝 GROUND TRUTH:       '{ground_truth}'")
        
        prediction_wer = wer(ground_truth, ai_prediction)
        prediction_cer = cer(ground_truth, ai_prediction)
        
        print(f"")
        print(f"📈 Prediction WER/CER: {prediction_wer:.3f} / {prediction_cer:.3f}")
        
        if prediction_wer == 0:
            print("🏆 PERFECT! Prediction is 100% accurate!")
        elif prediction_wer < 0.1:
            print("✅ EXCELLENT! Prediction is very good!")
        elif prediction_wer < 0.3:
            print("👍 GOOD! Prediction is quite good!")
        else:
            print("❌ POOR! Prediction needs improvement!")
    else:
        print("📝 No ground truth available (not a LibriSpeech file)")
        print("🎯 Transcription completed successfully!")
    
    print("=" * 60)
    return ai_prediction

def suggest_test_files():
    """Suggest available test files"""
    print("\n🎯 SUGGESTED TEST FILES:")
    print("=" * 50)
    
    # Check for LibriSpeech test files
    librispeech_path = "E:/PBL6/LibriSpeech/test-clean"
    if os.path.exists(librispeech_path):
        print("📁 LibriSpeech test files (with ground truth):")
        speakers = [d for d in os.listdir(librispeech_path) if os.path.isdir(os.path.join(librispeech_path, d))][:3]
        for speaker in speakers:
            speaker_path = os.path.join(librispeech_path, speaker)
            chapters = [d for d in os.listdir(speaker_path) if os.path.isdir(os.path.join(speaker_path, d))][:1]
            for chapter in chapters:
                chapter_path = os.path.join(speaker_path, chapter)
                audio_files = [f for f in os.listdir(chapter_path) if f.endswith('.flac')][:1]
                for audio_file in audio_files:
                    full_path = os.path.join(chapter_path, audio_file)
                    print(f"   {full_path}")
    
    # Check Downloads folder for common video/audio files
    downloads_path = "C:/Users/LENOVO/Downloads"
    if os.path.exists(downloads_path):
        print("\n🎬 Video/Audio files in Downloads:")
        extensions = ['.mp4', '.mp3', '.wav', '.avi', '.mov', '.flac']
        files = []
        for file in os.listdir(downloads_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(downloads_path, file))
        
        for file in files[:5]:  # Show first 5
            print(f"   {file}")
        
        if len(files) > 5:
            print(f"   ... and {len(files)-5} more files")

def show_available_speakers():
    """Hiển thị các speakers có sẵn"""
    print("\n📁 Available speakers in test-clean:")
    test_clean_path = "E:/PBL6/LibriSpeech/test-clean"
    if os.path.exists(test_clean_path):
        speakers = [d for d in os.listdir(test_clean_path) if os.path.isdir(os.path.join(test_clean_path, d))]
        print(f"   {speakers[:10]}... ({len(speakers)} total)")
    
    print("\n🎯 Quick examples to copy-paste:")
    print("audio_file = 'E:/PBL6/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac'")
    print("audio_file = 'E:/PBL6/LibriSpeech/test-clean/1188/133604/1188-133604-0001.flac'")
    print("audio_file = 'E:/PBL6/LibriSpeech/test-clean/260/123286/260-123286-0000.flac'")
    print("audio_file = 'E:/PBL6/LibriSpeech/test-clean/1221/135766/1221-135766-0000.flac'")

# ==========================================
# MAIN - CHỈ CẦN SỬA ĐƯỜNG DẪN VÀ API KEY
# ==========================================

if __name__ == "__main__":
    print("🚀 DeepSpeech2 with Video Support + Gemini Enhancement")
    print("=" * 60)
    
    # Check dependencies
    if not VIDEO_SUPPORT:
        print("📦 To enable video file support:")
        print("   pip install moviepy")
        print()
    
    # Load model
    model, tokenizer, audio2mels, amp2db, device = load_model()
    
    # ✨ E:/PBL6/LibriSpeech/test-clean/1320/122617/1320-122617-0010.flacY ✨
    audio_file = "E:/PBL6/LibriSpeech/test-clean/1320/122617/1320-122617-0020.flac"

    # ✨ CẤU HÌNH GEMINI API ✨
    USE_GEMINI = True
    GEMINI_API_KEY = "AIzaSyDcqE4-ECdIbYChPXwu7Mg0KpgCaATab44"
    TRANSLATE_TO = "Vietnamese"
    
    # Test file
    test_audio(audio_file, model, tokenizer, audio2mels, amp2db, device,
               use_gemini=USE_GEMINI, 
               gemini_api_key=GEMINI_API_KEY,
               translate_to=TRANSLATE_TO)
    
    # Show suggestions for other files
    suggest_test_files()
    
    print("\n💎 Gemini Enhancement Pipeline:")
    print(f"- Video Support: {'✅' if VIDEO_SUPPORT else '❌ (install moviepy)'}")
    print(f"- USE_GEMINI = {USE_GEMINI}")
    print(f"- TRANSLATE_TO = {TRANSLATE_TO}")
    
    print("\n" + "=" * 60)
    print("✅ Script completed!")