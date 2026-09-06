# Dataset metadata observations

Inspected 2026-09-06. Only public cards/previews were inspected; no audio acquired.
Pin release revisions when exporting metadata. These observations are not a
full-corpus distribution audit.

| Source | Observed fields | Mapping / remaining uncertainty |
| --- | --- | --- |
| [IndicVoices-R card](https://huggingface.co/datasets/ai4bharat/indicvoices_r/blob/main/README.md) | lang, speaker_id, gender, age_group, text, duration, audio | Map lang to language; keep original value. Card declares 48 kHz audio, but per-row source rates and audio locators still require verification. |
| [IndicSynth publisher preview](https://huggingface.co/datasets/vdivyasharma/IndicSynth) | Generative Model, Target Speaker ID, Source Speaker_ID, Gender, Source Reference Audio, Target Reference Audio, TTS Transcript, audio | Preserve target/source roles. Language can come from the verified subset. Visible preview lacks standalone duration; do not infer it from reference audio. |
| [ASVspoof mirror](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF) | path, audio, label, notes | Parse notes for speaker, codec, attack ID and vocoder family. Numeric label mapping must be explicitly verified. Always external. |

The supplied [AIKosh URL](https://aikosh.indiaai.gov.in/home/datasets/details/indicsynth.html)
returned “No data found” through public retrieval. The adapter uses the publisher's
Hugging Face preview as evidence, not an assumption that AIKosh exports match it.
The publisher is linked to the [IndicSynth paper](https://aclanthology.org/2025.acl-long.1070/).

IndicSynth's card identifies the genuine source/reference recordings as IndicSUPERB.
This is a meaningful provenance difference from the proposed genuine corpus,
IndicVoices-R. Evaluate whether a matched genuine comparison source is needed;
this milestone does not change the requested dataset strategy.

Before real acquisition, manually verify:

1. Dataset release revisions, export field names, duration units, language values,
   original sample rates and codec/container meanings.
2. That IndicVoices-R's restoration/enhancement process is appropriate for a
   bona fide anti-spoofing reference and which processing signatures remain.
3. A stable generated-audio locator per IndicSynth row. Source/target reference
   files are not the generated clip. Do not copy duration from a reference file.
4. Whether speaker IDs are global or language/subset-local, shared across corpora,
   and whether repeated source/target speakers and recordings need joint grouping.
5. Actual available generators and attack annotations in each selected language,
   including missing metadata and any release-specific spelling differences.
6. Row-level common languages and hours; the fixture intersection is not evidence
   of coverage in the real exports. Confirm desired six languages individually.
7. Metadata-only export availability, transfer size and permission. Streaming audio
   datasets can still transfer embedded audio bytes; do not run card download
   examples or full load_dataset calls to obtain metadata casually.
8. Applicable source terms and attribution. The publisher cards list CC BY 4.0
   for IndicVoices-R and CC BY-NC 4.0 for IndicSynth; verify the selected release
   and intended use before acquisition.
9. Small approved local audio checks for RMS, silence, channels, format, duration,
   and decode failures; obtain explicit approval for any large audio acquisition.
10. Freeze ASVspoof away from all training, threshold and model-selection decisions.
