
# Python-ML1

Multilingual Translators — Project README

Overview

This repository collects *six related translator features / experiments* implemented as part of a single training project (NULLCLASS). Each feature is implemented as a separate Python module and — where requested — a GUI. The goal was to extend your original training project with internship tasks (extra features, models, or experiments) while re-using the same project/dataset where possible.

The six tasks covered here are:

1. *French → Tamil (5-letter-only translator)
2. *Dual Language Translator — English → French + Hindi (>=10 letters only)
3. *Voice Translator — English audio → Hindi; time-windowed, 9:30 PM–10:00 PM
4. Logical Building Task — English → Hindi with vowel rules and time-window
5. *OCR → Translator (image OCR + many target languages)*

6. *Realtime Voice Conversation — English ↔ Spanish (bidirectional voice translation)

Each of the task sections below contains: Problem Statement, Dataset, Methodology, How to run, Results & Evaluation notes, Limitations and Next steps.

---

# 1) French → Tamil (5-letter-only translator)

Problem statement

Develop a translator that accepts *French words* but only returns a *Tamil translation* when a French word has *exactly five letters*. For other lengths, the system should refuse to translate and return a clear message.

GUI Requirement: a simple input (French word) and an output (Tamil translation), plus messages for invalid inputs.

Dataset

* translator.csv: a plain text/ CSV file in the format french → tamil (one pair per line). The given code reads this file and filters entries whose French word length is exactly 5.

We used the exact file path found in your code: C:\\Users\\pavit\\OneDrive\\Desktop\\NULLCLASS2\\translator.csv.
## Methodology

* *Preprocessing*: read the CSV / text file, split lines on →, normalize to lowercase and filter to 5-character French words.
* *Model*: used a pretrained multilingual model — facebook/mbart-large-50-many-to-many-mmt — for inference. The tokenizer used is MBart50TokenizerFast (fallback to MBart50Tokenizer if needed).
* Inference logic:

* Check input length. If len(word) != 5, return Only 5-letter French words are supported!.

* Set tokenizer.src_lang = 'fr_XX' and generate tokens forcing forced_bos_token_id to the Tamil code ta_IN.
* Decode and return the result.

* *GUI*: In the sample code, this is done with Gradio (gr.Interface), with one textbox input, and one textbox output.

## How to run

* Install the dependencies: pip install gradio transformers torch sentencepiece.

* Ensure translator.csv exists and contains french → tamil lines.

* Execute the Python script and Gradio will open a local web UI.

Results & evaluation

* *Qualitative*: The model gives Tamil text for 5-letter French inputs. Example usage (replace with your concrete data):

* amour → காதல் (example; actual output depends on model and tokenizer)

* *Quantitative*: fine-grained evaluation (BLEU / chrF) on a held-out test set derived from translator.csv will be appropriate. Since the current setup uses a pre-trained multilingual model w/o fine-tuning, results will vary.

Limitations & next steps

* The code does not fine-tune the mBART on your parallel pairs. Fine-tuning on the filtered dataset would, in general, improve accuracy for single-word translations.

* Short single-word translations depend a lot on the model's vocabulary and training; consider augmenting the dataset or use a word-level dictionary-based fallback for small vocabularies.

---

2) Dual Language Translator — English → French + Hindi (10+ letters only)
Problem statement

Create a translator feature that takes English words or lines and translates them into *both French and Hindi simultaneously. It should accept only the input if it has **10 or more letters* in the character count. In cases where the input is less than 10 letters, the system must ask the user to “upload again.” A GUI with input and two output areas is needed as well.
Dataset

* triples.csv: A CSV file expected to have english and hindi columns.

* For English→French we can use a pre-trained translation model, e.g. Helsinki/opus or mBART. For Hindi, your code used Helsinki-NLP/opus-mt-en-hi and MarianMT variants.
Methodology

*Filtering*: check either len(input_text.replace(' ', '')) or len(input_text) depending on whether spaces count. The original specification indicates letter count; implement as len(alphabets_only) where stripping punctuation and spaces are done or keep simple len(input_text). Sample code used character length checks elsewhere in these tasks. Pick the behavior to be implemented consistently. README suggests counting letters only and to strip whitespace.

* Modeling:

English→French: either Helsinki-NLP/opus-mt-en-fr or facebook/mbart-large-50 with forced_bos_token_id for fr_XX.

*: English→Hindi: Helsinki-NLP/opus-mt-en-hi or Helsinki/MarianMT as in your scripts.

* *Training*: Your code uses pre-trained models for inference. If you need to "train your own model", perform these steps:

* Fine-tune a seq2seq model (Marian/mbart) using a parallel corpus (English→French and English→Hindi). Using the same base model, fine tune for each target language individually.

* Scripts: provide a transformers training script with Seq2SeqTrainer and dataset split to train/val/test.

* GUI: a simple Tkinter or Gradio interface with an input textbox and two outputs each. French and Hindi. If input length < 10 then show Please upload again or upload again as per spec

How to run

* Install: pip install transformers torch gradio sentencepiece sacrebleu

* Put triples.csv in the working directory.

* Start the GUI script.

Results & evaluation

* *Qualitative*: for inputs >= 10 letters the model produces two translations. Example (hypothetical):
* Development, 11 letters → French: développement, Hindi: विकास.
* *Quantitative evaluation*: evaluate using BLEU/chrF for both output languages on a held-out test set.

Limitations & next steps

* Ambiguity in the spec: whether to count spaces/punctuation toward the 10-letter rule — choose and document one interpretation.
• Fine-tuning on the English→French and English→Hindi dedicated parallel corpus improves domain-level performance.

---

3) Voice Translator - English audio → Hindi (time-windowed 9:30 PM–10:00 PM)

Problem statement

Accept English audio input from a user, transcribe it into English text (ASR), translate it into Hindi, and display the Hindi text. If the system doesn't understand the audio (low confidence), prompt the user to repeat. This feature should be *active only between 9:30 PM and 10:00 PM (Asia/Kolkata)*; outside that window, the GUI should say: Taking rest, see you tomorrow!
GUI: Tkinter-based, featuring the application called VoiceTranslatorApp; record button; English transcript area; Hindi translation area; confidence; time-window enforcement.

Dataset

* No external dataset is strictly required for the given demo; pretrained ASR models are used: facebook/wav2vec2-base-960h or a local ASR path, while for translation: Helsinki-NLP/opus-mt-en-hi.

* Optional: local human-transcribed English→Hindi parallel dataset for fine-tuning the translator.

## Methodology

* *ASR*: use Wav2Vec2 (or any other ASR) for speech-to-text. Your code loads local models if available (./models/asr) else falls back to facebook/wav2vec2-base-960h.

* *Confidence*: calculate a proxy confidence using softmax over logit outputs and take the mean or max as a heuristic--your code computes a mean of the maximum token probabilities.

* *Translation*: use AutoTokenizer + AutoModelForSeq2SeqLM from Helsinki-NLP/opus-mt-en-hi.

* *Time-window enforcement*: check current IST time with pytz and disable UI outside 21:30–22:00.

## How to run
* Install the dependencies: pip install torch transformers librosa sounddevice soundfile pytz (plus GUI libs which come with Python).

* Microphone permissions should be enabled.

* Run the Tkinter script — it will record 6 seconds of audio by default, transcribe and translate during the active time window.

Results & evaluation

* *Qualitative*: Sample code will show the transcript in English and translation in Hindi for confidence ≥ 0.25. When it is lower than that, it will just ask the user to repeat.

• *Quantitative*: evaluate ASR using WER on a labelled test set; evaluate translation using BLEU/chrF.

Limitations & next steps

* No explicit noise-robust pre-processing (denoising) included.

* Confidence estimation is heuristic; use lattice-based posterior scoring or ASR-provided confidence scores if available.

* In production, you would want streaming ASR and lower-latency translation.

---

# 4) Logic Building Task — English → Hindi with vowel & time rules

## Problem statement

Translate the English words to Hindi, *except words that start with vowels*. In case of an English word that starts with a vowel, show This word starts with a vowel. Please provide another word. Also, *allow the model to translate only English words that start with vowels between 9 PM and 10 PM*; that is, a special override window where vowel-starting words are allowed to be translated. GUI is mandatory.

*Interpretation used in this project*: By default, the system refuses to translate vowel-starting words. Between 21:00–22:00 IST, vowel-starting words are allowed and will be translated. All non-vowel-starting words are translated normally whenever the GUI is active.
Dataset

* english-hindi.csv is used as an optional lookup table. For unseen words the system falls back to a model (e.g., Helsinki-NLP/opus-mt-en-hi or MarianMT).

Methodology

* *Rules engine*: implement a simple rule to detect vowel-starting tokens. Vowels considered: a, e, i, o, u (case-insensitive). If the word starts with a vowel and current time is outside 21:00–22:00, show the error message.

* Translation: same as other English→Hindi flows: lookup in CSV first, then model inference.

* *GUI*: Tkinter minimal GUI - input field, output field, translate button; time display.

## How to run

Install dependencies and put english-hindi.csv in the working directory if available.

* Run the Tkinter script.

Results & evaluation

* Quick manual tests: check that words starting with a/e/i/o/u are rejected outside 21:00–22:00 and accepted within it.

Limitations & next steps

* Rules are simple, and operate on first character only — consider tokenization for multi-word inputs.

* Extend vowel rules to handle diacritics and non-ASCII characters.

---

# 5) OCR → Translator (Image OCR + many targets)
## Problem statement
An OCR image frontend extracts the text from images and translates it into the target language selected by the user (Hindi, French, Spanish, German, etc.). The GUI will be able to pick up an image, show extracted text, and show translated output.

Dataset

It is not data-driven: EasyOCR performs the detection, and googletrans performs the translation with an API wrapper for Google Translate. Alternatively, a dataset of scanned images and their ground truth can be used to evaluate OCR accuracy.

Methodology

* OCR: lazy initialization of easyocr.Reader; read text pieces with reader.readtext(image_path, detail=1); combine to string.

* *Translation*: googletrans.Translator used for quick demo translations to many languages.

* *UI*: Tkinter-based two-pane viewer for OCR-extracted text and translated text with a Combobox for selecting the target language.
## How to run

* pip install easyocr googletrans (with torch required by EasyOCR). Run the Tkinter script and select an image.

## Results & evaluation

* Qualitative: output depends on image quality and script. For printed clear text, the flow of OCR + translation works well. For handwritten or low-resolution images, it's possible that some content gets missed by OCR.

Limitations & next steps

* googletrans is an unofficial API and might break; consider using an official translation API for production.
Consider language detection and area-specific tokenizers for the improvement of translation accuracy.
---
# 6) Realtime Conversation with Voice Translation — English ↔ Spanish
## Problem statement
Design a system to enable near real-time conversation between an English speaker and a Spanish speaker, where each side speaks in their own language; the system transcribes, translates to the partner's language, and reads the translation aloud.
GUI: not mandatory — the provided code is a console/daemon-style threaded app that listens and speaks both sides.
## Dataset
* The basic demo does not need a dataset; it uses speech_recognition for listening (Google STT), googletrans for translation, and pyttsx3 for TTS. For higher accuracy, replace recognizer with a dedicated ASR model and translation with a seq2seq model.

Methodology


*: ASR: speech_recognition.Recognizer() with Google recognizer for the demo.
* *Translation*: googletrans.Translator.
* *TTS*: pyttsx3 chooses OS voices (tries to select language-appropriate voice if it exists).
* *Pipeline*: runs two threads:
* Spanish → English : listen in es-ES, translate es→en, speak en
* English → Spanish: listen in en-US, translate en→es, speak es.


## How to run
pip install SpeechRecognition googletrans pyttsx3 and also make sure to allow microphone permissions.

* Run the script and let each participant speak in turn. To stop, press Ctrl+C.
* 
### Results & evaluation
* The demo shows functional end-to-end translation and speech, adequate for constrained small-group testing. However, the demo has the following limitations that affect performance:
  
* Dependency on Google online recognizer, network and API variability.

* TTS languages/voices availability depends on OS.
* No speaker-turn management (can talk over each other, causing misunderstandings/error).
  ### Limitations & next steps
  * Move to dedicated local ASR and local NMT models for offline/robust performance using wav2vec2.
    
  * Add voice activity detection and speaker diarization to manage turn-taking.

     --
    # Dependencies (core)
    * Python 3.8+
    * torch
    * transformers
    * sentencepiece
    * gradio
    * easyocr
    * googletrans (demo)
    * sounddevice, soundfile, librosa (for recording / audio processing) they are:
    * speech_recognition
    * pyttsx3
    * pytz, pandas Example
      requirements.txt line: transformers torch sentencepiece gradio easyocr googletrans sounddevice soundfile librosa speechrecognition pyttsx3 pandas pytz
