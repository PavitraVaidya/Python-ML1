# Python-ML1

# Multilingual Translators — Project README

## Overview

This repository collects *six related translator features / experiments* implemented as part of a single training project (NULLCLASS). Each feature is implemented as a separate Python module and — where requested — a GUI. The goal was to extend your original training project with internship tasks (extra features, models, or experiments) while re-using the same project/dataset where possible.

The six tasks covered here are:

1. *French → Tamil (5-letter-only translator)*
2. *Dual Language Translator — English → French + Hindi (>=10 letters only)*
3. *Voice Translator — English audio → Hindi (time-windowed, 9:30 PM–10:00 PM)*
4. *Logic Building Task — English → Hindi with vowel rules and time-window*
5. *OCR → Translator (image OCR + many target languages)*
6. *Realtime Voice Conversation — English ↔ Spanish (bidirectional voice translation)*

Each task section below contains: Problem Statement, Dataset, Methodology, How to run, Results & Evaluation notes, Limitations and Next steps.

---

# 1) French → Tamil (5-letter-only translator)

## Problem statement

Build a translator feature that accepts *French words* and returns a *Tamil* translation *only* when the French word has *exactly five letters*. For any other length the system should refuse to translate and return a clear message.

GUI requirement: a simple input (French word) and an output (Tamil translation), plus messages for invalid inputs.

## Dataset

* translator.csv — a plain text / CSV file in the format french → tamil (one pair per line). The provided code reads this file and filters entries whose French word length is exactly 5.
* We used the provided file path in your code: C:\Users\pavit\OneDrive\Desktop\NULLCLASS2\translator.csv.

## Methodology

* *Preprocessing*: read the CSV / text file, split lines on →, normalize to lowercase and filter to 5-character French words.
* *Model*: used a pretrained multilingual model — facebook/mbart-large-50-many-to-many-mmt — for inference. The tokenizer used is MBart50TokenizerFast (fallback to MBart50Tokenizer if needed).
* *Inference logic*:

  * Check input length. If len(word) != 5, return Only 5-letter French words are supported!.
  * Set tokenizer.src_lang = 'fr_XX' and generate tokens forcing forced_bos_token_id to the Tamil code ta_IN.
  * Decode and return the result.
* *GUI*: implemented with Gradio in the sample code (gr.Interface), one textbox input and one textbox output.

## How to run

* Install dependencies: pip install gradio transformers torch sentencepiece.
* Ensure translator.csv exists and contains french → tamil lines.
* Run the Python script; Gradio will open a local web UI.

## Results & evaluation

* *Qualitative*: the model produces Tamil text for 5-letter French inputs. Example usage (replace with your concrete data):

  * Input: amour → Output: காதல் (example; actual output depends on model and tokenizer)
* *Quantitative*: to properly evaluate, fine-grained evaluation (BLEU / chrF) is recommended on a held-out test set derived from translator.csv. Since the current setup uses a pretrained multilingual model without fine-tuning, results will vary.

## Limitations & next steps

* The code does not fine-tune mBART on your parallel pairs. Fine-tuning on the filtered dataset would generally improve accuracy for single-word translations.
* Short single-word translation quality depends heavily on the model’s vocabulary and training; consider augmenting the dataset or using a word-level dictionary-based fallback for small vocabularies.

---

# 2) Dual Language Translator — English → French + Hindi (10+ letters only)

## Problem statement

Create a translator feature that accepts English words or lines, and translates them into *both French and Hindi simultaneously. The feature must only accept inputs that have **10 or more letters* (character count). For inputs shorter than 10 letters, the system must prompt the user to “upload again.” A GUI is required with input and two output areas.

## Dataset

* english-hindi.csv (if present) — a CSV file expected to have english and hindi columns.
* For English→French we can use a pretrained translation model (e.g., Helsinki/opus or mBART). For Hindi, your code used Helsinki-NLP/opus-mt-en-hi and MarianMT variants.

## Methodology

* *Filtering*: check len(input_text.replace(' ', '')) or len(input_text) depending on whether spaces count; the original spec indicates letter count — implement as len(alphabets_only) (strip punctuation and spaces) or keep simple len(input_text). The sample code used character length checks in other tasks — pick the behavior consistently (README suggests counting letters only: strip whitespace).
* *Modeling*:

  * English→French: either Helsinki-NLP/opus-mt-en-fr or facebook/mbart-large-50 with forced_bos_token_id for fr_XX.
  * English→Hindi: Helsinki-NLP/opus-mt-en-hi or Helsinki/MarianMT as in your scripts.
* *Training*: your provided code uses pretrained models for inference. If you need to "train your own model" (as the guideline requested), follow these steps:

  * Fine-tune a seq2seq model (Marian/mbart) using a parallel corpus (English→French and English→Hindi). Use the same base model and fine-tune for each target language separately.
  * Scripts: provide a transformers training script with Seq2SeqTrainer and dataset split into train/val/test.
* *GUI*: a simple Tkinter or Gradio interface with an input textbox and two outputs (French, Hindi). If input length < 10, show Please upload again or upload again per spec.

## How to run

* Install: pip install transformers torch gradio sentencepiece sacrebleu.
* Place english-hindi.csv in the working directory (optional — used as fallback/lookup table).
* Start the GUI script.

## Results & evaluation

* *Qualitative*: for inputs >= 10 letters, the model outputs two translations. Example (hypothetical):

  * Input: development (11 letters) → French: développement, Hindi: विकास.
* *Quantitative evaluation*: evaluate using BLEU/chrF for both output languages on a held-out test set.

## Limitations & next steps

* Ambiguity in the spec: whether to count spaces/punctuation toward the 10-letter rule — choose and document one interpretation.
* Fine-tuning on a dedicated parallel corpus (English→French, English→Hindi) improves domain-level performance.

---

# 3) Voice Translator — English audio → Hindi (time-windowed 9:30 PM–10:00 PM)

## Problem statement

Accept English audio input from a user, transcribe it to English text (ASR), translate it into Hindi, and display the Hindi text. If the system doesn’t understand the audio (low confidence), prompt the user to repeat. This feature should be *active only between 9:30 PM and 10:00 PM (Asia/Kolkata)*; outside that window the GUI should display: Taking rest, see you tomorrow!.

GUI: Tkinter-based (VoiceTranslatorApp), record button, English transcript area, Hindi translation area, confidence display, and time-window enforcement.

## Dataset

* No external dataset is strictly required for the provided demo; the system uses pretrained ASR models (facebook/wav2vec2-base-960h or a local ASR path) and pretrained translation (Helsinki-NLP/opus-mt-en-hi) models.
* Optional: local human-transcribed English→Hindi parallel dataset for fine-tuning the translator.

## Methodology

* *ASR*: use Wav2Vec2 (or any other ASR) for speech-to-text. Your code loads local models if available (./models/asr) else falls back to facebook/wav2vec2-base-960h.
* *Confidence*: compute a proxy confidence using softmax over logit outputs and take the mean or max as a heuristic (your code computes a mean of the maximum token probabilities).
* *Translation*: use AutoTokenizer + AutoModelForSeq2SeqLM from Helsinki-NLP/opus-mt-en-hi.
* *Time-window enforcement*: check current IST time with pytz and disable UI outside 21:30–22:00.

## How to run

* Install dependencies: pip install torch transformers librosa sounddevice soundfile pytz (plus GUI libs which come with Python).
* Make sure microphone permissions are enabled.
* Run the Tkinter script — it will record 6 seconds of audio by default, transcribe and translate during the active time window.

## Results & evaluation

* *Qualitative*: The sample code will display the English transcript and Hindi translation when confidence ≥ 0.25. For lower confidence it will ask the user to repeat.
* *Quantitative*: evaluate ASR using WER on a labeled test set; evaluate translation using BLEU/chrF.

## Limitations & next steps

* No explicit noise-robust pre-processing (denoising) included.
* The confidence estimation is heuristic; consider using lattice-based posterior scoring or ASR-provided confidence scores.
* For production you'd want streaming ASR and lower-latency translation.

---

# 4) Logic Building Task — English → Hindi with vowel & time rules

## Problem statement

Translate English words into Hindi *but do not translate words that start with vowels*. If an English word starts with a vowel, display This word starts with a vowel. Please provide another word. Additionally, *the model should only translate English words that start with vowels between 9 PM and 10 PM* (i.e., a special override window where vowel-starting words are allowed to be translated). GUI mandatory.

*Interpretation used in this project*: By default the system refuses to translate vowel-starting words. Between 21:00–22:00 IST, vowel-starting words are allowed and will be translated. All non-vowel-starting words are translated normally whenever the GUI is active.

## Dataset

* english-hindi.csv is used as an optional lookup table. For unseen words the system falls back to a model (e.g., Helsinki-NLP/opus-mt-en-hi or MarianMT).

## Methodology

* *Rules engine*: implement a simple rule to detect vowel-starting tokens. Vowels considered: a, e, i, o, u (case-insensitive). If the word starts with a vowel and current time is outside 21:00–22:00, show the error message.
* *Translation*: same as other English→Hindi flows: lookup in CSV first, then model inference.
* *GUI*: Tkinter minimal GUI (input field, output field, translate button); time display.

## How to run

* Install dependencies and place english-hindi.csv in the working directory if available.
* Run the Tkinter script.

## Results & evaluation

* Quick manual tests: verify that words beginning with a/e/i/o/u are rejected outside 21:00–22:00 and accepted inside that window.

## Limitations & next steps

* Rules are simple and operate on the first character only — consider tokenization for multi-word inputs.
* Expand vowel rules to handle diacritics and non-ASCII characters.

---

# 5) OCR → Translator (Image OCR + many targets)

## Problem statement

Provide an image OCR frontend that extracts text from images and translates into a chosen target language (Hindi, French, Spanish, German, etc.). The GUI should allow image selection, show extracted text, and show translated output.

## Dataset

* Not data-driven; uses EasyOCR for detection and googletrans for translation (API wrapper for Google Translate). Optionally a dataset of scanned images with ground truth could be used to evaluate OCR accuracy.

## Methodology

* *OCR*: easyocr.Reader initialized lazily; reader.readtext(image_path, detail=1) to extract text pieces; combine into a single string.
* *Translation*: googletrans.Translator used for quick demo translations to many languages.
* *UI*: Tkinter-based two-pane viewer showing OCR-extracted text and translated text; language selection using a Combobox.

## How to run

* pip install easyocr googletrans (plus torch required by EasyOCR). Launch the Tkinter script and select an image.

## Results & evaluation

* Qualitative: the output depends on image quality and script. For printed clear text the OCR + translate flow works well. For handwritten or low-resolution images, OCR may miss content.

## Limitations & next steps

* googletrans is an unofficial API and may break; consider using an official translation API for production.
* Consider language detection and region-specific tokenizers for better translation accuracy.

---

# 6) Realtime Conversation with Voice Translation — English ↔ Spanish

## Problem statement

Build a system that enables near real-time conversation between an English speaker and a Spanish speaker. Each side speaks in their language; the system transcribes, translates to the other language, and reads the translation aloud.

GUI: not mandatory — the provided code is a console/daemon-style threaded app that listens and speaks both sides.

## Dataset

* No dataset required for the basic demo; uses speech_recognition (Google STT) for listening, googletrans for translation, and pyttsx3 for TTS. For higher accuracy, replace recognizer with a dedicated ASR model and translation with a seq2seq model.

## Methodology

* *ASR*: speech_recognition.Recognizer() with Google recognizer for the demo.
* *Translation*: googletrans.Translator.
* *TTS*: pyttsx3 selects OS voices (attempts to pick language-appropriate voice if available).
* *Pipeline*: Runs two threads:

  * Spanish → English: listen in es-ES, translate es→en, speak en.
  * English → Spanish: listen in en-US, translate en→es, speak es.

## How to run

* pip install SpeechRecognition googletrans pyttsx3 and ensure microphone permissions.
* Run the script and let each participant speak in turn. Press Ctrl+C to stop.

## Results & evaluation

* The demo shows functional end-to-end translation and speech, adequate for constrained small-group testing. However, the demo has the following limitations that affect performance:

  * Dependency on Google online recognizer — network and API variability.
  * TTS languages/voices availability depends on OS.
  * No speaker-turn management (can overlap speech and cause errors).

## Limitations & next steps

* Move to dedicated local ASR (wav2vec2) and local NMT models for offline/robust performance.
* Add voice activity detection (VAD) and speaker diarization to manage turn-taking.

--

# Dependencies (core)

* Python 3.8+
* torch
* transformers
* sentencepiece
* gradio
* easyocr
* googletrans (demo)
* sounddevice, soundfile, librosa (for recording / audio processing)
* speech_recognition, pyttsx3
* pytz, pandas

Example requirements.txt line:


transformers
torch
sentencepiece
gradio
easyocr
googletrans
sounddevice
soundfile
librosa
speechrecognition
pyttsx3
pandas
pytz

