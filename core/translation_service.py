import asyncio
from typing import Dict, List, Optional, Tuple
import numpy as np
from googletrans import Translator
import azure.cognitiveservices.speech as speechsdk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import redis
import json
import hashlib
from datetime import datetime, timedelta
import logging


class MultiLanguageTranslator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.google_translator = Translator()
        
        self.cache = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        self.nlp_models = {}
        self._load_nlp_models()
        
        self.sign_vocabularies = self._load_sign_vocabularies()
        
    def _load_nlp_models(self):
        model_configs = {
            'en-es': 'Helsinki-NLP/opus-mt-en-es',
            'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
            'en-de': 'Helsinki-NLP/opus-mt-en-de',
            'en-ja': 'Helsinki-NLP/opus-mt-en-ja',
            'en-zh': 'Helsinki-NLP/opus-mt-en-zh'
        }
        
        for pair, model_name in model_configs.items():
            try:
                self.nlp_models[pair] = pipeline(
                    'translation',
                    model=model_name,
                    device=0 if tf.config.list_physical_devices('GPU') else -1
                )
            except Exception as e:
                self.logger.warning(f"Failed to load {pair} model: {e}")
    
    def _load_sign_vocabularies(self) -> Dict[str, Dict]:
        vocabularies = {}
        
        for lang_config in self.config['languages']['sign_languages']:
            vocab_path = f"data/{lang_config['code']}/vocabulary.json"
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocabularies[lang_config['code']] = json.load(f)
            except:
                vocabularies[lang_config['code']] = {}
        
        return vocabularies
    
    async def translate_sign_sequence(self, 
                                    predictions: List[int],
                                    source_language: str,
                                    target_language: str,
                                    context: Optional[Dict] = None) -> Dict:
        cache_key = self._get_cache_key(predictions, source_language, target_language)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        gloss_sequence = self._predictions_to_gloss(predictions, source_language)
        
        intermediate_text = self._gloss_to_text(gloss_sequence, source_language, context)
        
        if target_language == self._get_base_language(source_language):
            final_translation = intermediate_text
        else:
            final_translation = await self._translate_text(
                intermediate_text, 
                self._get_base_language(source_language),
                target_language
            )
        
        result = {
            'source_language': source_language,
            'target_language': target_language,
            'gloss_sequence': gloss_sequence,
            'intermediate_text': intermediate_text,
            'final_translation': final_translation,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.cache.setex(
            cache_key,
            self.config.get('cache_ttl', 3600),
            json.dumps(result)
        )
        
        return result
    
    def _predictions_to_gloss(self, predictions: List[int], 
                            language: str) -> List[str]:
        vocabulary = self.sign_vocabularies.get(language, {})
        gloss_sequence = []
        
        for pred in predictions:
            gloss_info = vocabulary.get(str(pred), {})
            if isinstance(gloss_info, dict):
                gloss = gloss_info.get('gloss', f'UNKNOWN_{pred}')
            else:
                gloss = str(gloss_info)
            
            gloss_sequence.append(gloss)
        
        return gloss_sequence
    
    def _gloss_to_text(self, gloss_sequence: List[str], 
                      language: str,
                      context: Optional[Dict] = None) -> str:
        rules = self._get_grammar_rules(language)
        
        processed_glosses = []
        i = 0
        
        while i < len(gloss_sequence):
            current = gloss_sequence[i]
            
            if i + 1 < len(gloss_sequence):
                bigram = (current, gloss_sequence[i + 1])
                if bigram in rules.get('compounds', {}):
                    processed_glosses.append(rules['compounds'][bigram])
                    i += 2
                    continue
            
            if current in rules.get('replacements', {}):
                processed_glosses.append(rules['replacements'][current])
            else:
                processed_glosses.append(current.lower())
            
            i += 1
        
        text = ' '.join(processed_glosses)
        
        text = self._apply_grammar_corrections(text, language)
        
        if context:
            text = self._apply_context(text, context)
        
        return text
    
    def _get_grammar_rules(self, language: str) -> Dict:
        rules = {
            'ASL': {
                'compounds': {
                    ('THANK', 'YOU'): 'thank you',
                    ('GOOD', 'MORNING'): 'good morning',
                    ('HOW', 'ARE', 'YOU'): 'how are you'
                },
                'replacements': {
                    'ME': 'I',
                    'YOU': 'you',
                    'QUESTION': '?',
                    'PERIOD': '.'
                },
                'word_order': 'TIME_TOPIC_COMMENT'
            },
            'BSL': {
                'compounds': {},
                'replacements': {},
                'word_order': 'TOPIC_COMMENT'
            }
        }
        
        return rules.get(language, {})
    
    def _apply_grammar_corrections(self, text: str, language: str) -> str:
        if language == 'ASL':
            text = text.replace(' ?', '?')
            text = text.replace(' .', '.')
            text = text.replace(' ,', ',')
            
            words = text.split()
            if len(words) > 1 and words[-1] in ['what', 'where', 'when', 'who', 'why', 'how']:
                words = [words[-1]] + words[:-1]
                text = ' '.join(words) + '?'
        
        text = text[0].upper() + text[1:] if text else text
        
        return text
    
    def _apply_context(self, text: str, context: Dict) -> str:
        if 'previous_sentence' in context:
            pass
        
        if 'user_preferences' in context:
            formality = context['user_preferences'].get('formality', 'neutral')
            if formality == 'formal':
                text = text.replace("hi", "hello")
                text = text.replace("thanks", "thank you")
        
        return text
    
    async def _translate_text(self, text: str, 
                            source_lang: str, 
                            target_lang: str) -> str:
        lang_pair = f"{source_lang}-{target_lang}"
        
        if lang_pair in self.nlp_models:
            try:
                result = self.nlp_models[lang_pair](text, max_length=512)
                return result[0]['translation_text']
            except Exception as e:
                self.logger.error(f"NLP model translation failed: {e}")
        
        try:
            translation = self.google_translator.translate(
                text, 
                src=source_lang, 
                dest=target_lang
            )
            return translation.text
        except Exception as e:
            self.logger.error(f"Google translation failed: {e}")
            return text
    
    def _get_base_language(self, sign_language: str) -> str:
        mappings = {
            'ASL': 'en',
            'BSL': 'en',
            'JSL': 'ja',
            'CSL': 'zh-cn',
            'FSL': 'fr',
            'GSL': 'de'
        }
        return mappings.get(sign_language, 'en')
    
    def _get_cache_key(self, predictions: List[int], 
                      source_lang: str, 
                      target_lang: str) -> str:
        content = f"{predictions}{source_lang}{target_lang}"
        return f"translation:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def translate_continuous_stream(self, 
                                        prediction_stream: asyncio.Queue,
                                        source_language: str,
                                        target_language: str) -> asyncio.Queue:
        output_queue = asyncio.Queue()
        buffer = []
        last_translation_time = datetime.now()
        
        async def process_stream():
            nonlocal buffer, last_translation_time
            
            while True:
                try:
                    prediction = await asyncio.wait_for(
                        prediction_stream.get(), 
                        timeout=2.0
                    )
                    
                    if prediction is None:
                        if buffer:
                            translation = await self.translate_sign_sequence(
                                buffer, source_language, target_language
                            )
                            await output_queue.put(translation)
                        break
                    
                    buffer.append(prediction)
                    
                    if (datetime.now() - last_translation_time).seconds > 3:
                        if buffer:
                            translation = await self.translate_sign_sequence(
                                buffer, source_language, target_language
                            )
                            await output_queue.put(translation)
                            buffer = []
                            last_translation_time = datetime.now()
                
                except asyncio.TimeoutError:
                    if buffer:
                        translation = await self.translate_sign_sequence(
                            buffer, source_language, target_language
                        )
                        await output_queue.put(translation)
                        buffer = []
                        last_translation_time = datetime.now()
        
        asyncio.create_task(process_stream())
        return output_queue


class TextToSpeech:
    def __init__(self, config: Dict):
        self.config = config
        
        if config.get('tts_backend') == 'azure':
            self.speech_config = speechsdk.SpeechConfig(
                subscription=config['azure_key'],
                region=config['azure_region']
            )
    
    async def synthesize_speech(self, text: str, 
                              language: str = 'en-US',
                              voice: Optional[str] = None) -> bytes:
        if self.config.get('tts_backend') == 'azure':
            return await self._azure_tts(text, language, voice)
        else:
            return await self._google_tts(text, language)
    
    async def _azure_tts(self, text: str, language: str, 
                        voice: Optional[str] = None) -> bytes:
        if voice:
            self.speech_config.speech_synthesis_voice_name = voice
        else:
            self.speech_config.speech_synthesis_language = language
        
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None
        )
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        else:
            raise Exception(f"Speech synthesis failed: {result.reason}")
    
    async def _google_tts(self, text: str, language: str) -> bytes:
        from gtts import gTTS
        import io
        
        tts = gTTS(text=text, lang=language[:2])
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read()