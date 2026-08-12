# Covid-RARS Plain-Language Explanation Guide

Use this file when you need to explain the project without sounding like you are hiding behind technical words. The goal is to answer "what did you actually do?" in simple language.

## 1. What Features Did We Extract From Audio?

Do not say only "measured audio-summary features." That is too vague.

Say this:

```text
We did not give only the raw sound file to the model. We first measured many properties of each cough, breath, or speech recording: how long it is, how loud it is, how energy changes, which frequency ranges are strong, how rough or noisy the sound is, and how the sound changes over time. These measured audio summaries became the input features for classical models.
```

Plain examples:

| Feature type | Easy meaning | Why it might matter |
|---|---|---|
| Duration | How long the usable sound is | Short, incomplete, or unusual clips can behave differently |
| Loudness / energy | How strong the sound is | Cough strength and breathing effort may change energy patterns |
| RMS energy | Average signal power | Summarizes how intense the sound is |
| Zero-crossing rate | How often the waveform jumps from positive to negative | Higher values can indicate noisier or sharper sounds |
| Spectral centroid | Where the "center of mass" of frequency energy lies | Higher centroid means a brighter/sharper sound |
| Spectral bandwidth | How spread out the frequencies are | Noisy or complex sounds spread energy across frequencies |
| Spectral rolloff | Frequency below which most energy lies | Captures whether energy is concentrated low or high |
| Spectral flatness | How noise-like vs tone-like the signal is | Helps separate structured voice/breath from noisy audio |
| MFCCs | Compact summary of the sound's frequency shape | Standard speech/audio fingerprint; useful for cough/speech tone |
| Delta MFCCs | How MFCCs change over time | Captures movement/change in the cough or speech |
| Delta-delta MFCCs | How the rate of change changes | Captures acceleration-like change in audio pattern |
| Mel-band summaries | Energy in human-hearing-inspired frequency bands | Captures low/mid/high frequency distribution |
| Chroma / tonnetz | Pitch and harmonic summaries | More useful for speech-like segments than pure cough |
| Spectral contrast | Difference between peaks and valleys in frequency bands | Captures sharpness and texture |
| Tempogram | Rhythm/change pattern over time | Captures repeated temporal structure |

We also used OpenSMILE ComParE 2016 and IS10. Explain them like this:

```text
OpenSMILE is a standard audio-feature toolkit used in speech and paralinguistic research. ComParE 2016 and IS10 are predefined collections of thousands of audio measurements. They include pitch, loudness, spectral shape, MFCCs, jitter/shimmer-like voice-quality measures, and summary statistics such as mean, standard deviation, percentiles, ranges, and slopes.
```

The simple version:

```text
Our audio features are measured summaries of sound, not magic. They describe duration, loudness, pitch/frequency shape, noisiness, rhythm, and how these change over time.
```

## 2. From So Many Features, What Did We Choose And Why?

After combining our measured audio summaries with OpenSMILE ComParE 2016 and IS10, the table had `10,147` columns in total. Seven were identifier or label fields, leaving `10,140` numeric feature candidates.

That is too many for the number of participants. If we train directly on all of them, the model can memorize noise.

So we selected features in this way:

1. Use only the training split.
2. Train a LightGBM ranking model on the training rows.
3. Ask which features helped separate positive and negative labels in training.
4. Keep the top `800`.
5. Use the same selected `800` features for validation, test, temporal, and external checks.

Why top `800`?

```text
We tested top 500, 800, and 1200. Top 800 was a practical middle point: enough information to perform strongly, but not so many features that the model becomes unnecessarily high-dimensional.
```

Why LightGBM?

```text
LightGBM is good at ranking which features are useful in high-dimensional tabular data. It can capture non-linear relations and gives feature-importance scores.
```

Important defense:

```text
Feature selection was not done using test or COUGHVID labels. It was learned from training data only, so the test set was not used to choose the features.
```

## 3. What Were The Weights Used And Why?

There are different meanings of "weights" in this project.

### A. Model weights

For WavLM:

- We used pretrained `microsoft/wavlm-base-plus` weights.
- These weights come from large-scale self-supervised audio/speech pretraining.
- We fine-tuned only part of the model, including the top layers and classification head.

Easy explanation:

```text
WavLM already knows general audio/speech patterns from large pretraining. We adapted it to our COVID cough task instead of training a transformer from zero.
```

### B. Class weights / imbalance handling

COVID positive and negative examples are imbalanced. If the model sees many more negatives, it may learn to predict negative too often.

So some models use balanced class weighting or SMOTE-style imbalance handling.

Easy explanation:

```text
The weights tell the model not to ignore the smaller positive class. A positive mistake and a negative mistake should both matter.
```

### C. Fusion weights

When combining cough, breath, and speech predictions, we used different fusion strategies:

| Fusion type | Easy meaning |
|---|---|
| Uniform mean | Give each available model/modality equal weight |
| Validation-weighted fusion | Give more weight to models that performed better on validation data |
| Stacked logistic fusion | Learn a small second-stage model that combines validation predictions |

Why use validation weights?

```text
If speech is more reliable than breath on validation data, the fusion should be allowed to trust speech more. But this weighting is learned from validation data, not from the test set.
```

### D. IPW weights

IPW means inverse probability weighting. It is not a neural-network weight.

Easy explanation:

```text
IPW is a statistical balancing method. If one group is overrepresented, it gives that group lower weight and gives underrepresented groups higher weight, so the comparison becomes less biased.
```

Why used?

```text
We used it to check whether performance remains high after reducing imbalance in metadata/context variables.
```

## 4. What Is Time-Stratified Validation?

Simple explanation:

```text
Time-stratified validation means we split participants while also paying attention to when recordings were collected. The goal is to avoid a split where the train and test sets are accidentally too similar in collection time.
```

Why someone cares:

```text
COVID data was collected across different pandemic waves. Testing policy, symptoms, variants, devices, and participant behavior changed over time. A model may look good if train and test come from the same time period, but fail when time changes.
```

In our result:

- Existing participant split: `0.895` AUROC for the primary equal-weight fusion (`0.897` for the exploratory stack).
- Time-stratified participant split: `0.849` AUROC.

Meaning:

```text
The model is still strong, but performance drops when we make the time structure more careful.
```

## 5. What Is Early-To-Late Temporal Validation?

Simple explanation:

```text
Early-to-late validation means train the model on earlier recordings and test it on later recordings.
```

Why someone cares:

```text
This is closer to real deployment. In real life, we train a model on past data and use it on future patients. If it fails from early to late, it means the model learned patterns that changed over time.
```

In our result:

- Early-to-late AUROC drops to about `0.698`.
- Multi-seed temporal mean is about `0.691 +/- 0.006`.

Meaning:

```text
The model's learned audio patterns are not stable across the pandemic timeline.
```

## 6. What Is Shuffle-Label Sanity?

Simple explanation:

```text
Shuffle-label sanity means we randomly mix up the positive/negative labels and run the pipeline again.
```

Why do this?

```text
If the model still gets high AUROC after labels are randomly shuffled, then something is wrong: maybe leakage or a bug. If performance drops to around 0.5, the pipeline passes this sanity check.
```

Our result:

- Full metadata observed AUROC: `0.964`.
- Full metadata shuffled AUROC: about `0.499`.
- Symptoms-only observed AUROC: `0.932`.
- Symptoms-only shuffled AUROC: about `0.500`.

Meaning:

```text
The high metadata score is not a simple coding bug. The dataset itself contains real label-related shortcuts.
```

## 7. What Is Temporal Robustness?

Simple explanation:

```text
Temporal robustness means checking whether the model gives similar results across time-based splits and repeated random seeds.
```

Why someone cares:

```text
If a model only works for one lucky split, reviewers will not trust it. If the drop appears again and again across seeds, then the problem is real.
```

Our result:

- Internal equal-weight fusion across five random states: about `0.8945 +/- 0.0068`.
- Strict early-to-late temporal validation: about `0.691 +/- 0.006`.

Meaning:

```text
The internal result is stable, and the temporal drop is also stable. The drop is not random bad luck.
```

## 8. What Is Feature Stability?

Simple explanation:

```text
Feature stability asks whether the same audio measurements are selected as important in early data and late data.
```

Our result:

- Early top-800 and late top-800 shared only `110` features.
- Jaccard overlap was `0.074`.

Why someone cares:

```text
If the important features keep changing over time, then the model is probably not learning one stable disease sound. It is learning patterns tied to collection time or changing dataset conditions.
```

## 9. What Is Support Overlap?

Simple explanation:

```text
Support overlap asks whether the external dataset looks similar to the training dataset in feature space.
```

Imagine this:

```text
If all Coswara samples live in one region of the feature map, but many COUGHVID samples live outside that region, the model is guessing in unfamiliar territory.
```

Our result:

- Domain classifier AUROC: `0.750`.
- About `25.2%` of COUGHVID examples are probably outside the Coswara support band.

Why someone cares:

```text
This explains why external transfer fails. The external data is not just a new test set; a large part of it looks different from the training domain.
```

## 10. What Is Calibration?

Simple explanation:

```text
Calibration asks whether predicted probabilities are believable. If the model says 80% risk, then around 80 out of 100 similar cases should actually be positive.
```

Why someone cares:

```text
In screening, probability quality matters. A model can rank cases decently but still give unsafe probability values.
```

Metrics:

| Metric | Easy meaning |
|---|---|
| Brier score | Average probability error; lower is better |
| ECE | Expected calibration error; lower is better |

Reverse temporal result:

- AUROC `0.920`, but ECE `0.471`.

Meaning:

```text
The model can appear good by AUROC while its probabilities are badly wrong.
```

## 11. What Are Clinical Operating Points?

Simple explanation:

```text
An operating point is the threshold where we decide positive or negative.
```

For screening, we often want high sensitivity:

```text
We want to catch most positives, so we force sensitivity to at least 0.90.
```

Our COUGHVID result at high sensitivity:

- specificity about `0.11-0.16`.
- precision about `0.035-0.037`.
- COUGHVID prevalence about `0.034`.

Why someone cares:

```text
Precision is barely above base rate. That means if the model flags someone positive, it is almost no better than knowing the dataset prevalence.
```

## 12. What Is Decision Curve Analysis?

Simple explanation:

```text
Decision curve analysis asks whether using the model gives more clinical benefit than simple strategies like treating everyone or treating no one.
```

Why someone cares:

```text
AUROC is not enough for medicine. A model must improve decisions. If it creates many false positives with little benefit, it is not useful for screening.
```

Our result:

```text
Under external transfer, decision-curve evidence is weak or negative, meaning the model does not provide useful clinical net benefit.
```

## 13. What Is Recalibration?

Simple explanation:

```text
Recalibration means adjusting the model's probability outputs on a small target-domain sample without retraining the full model.
```

Why someone cares:

```text
If external failure is only a threshold/probability problem, recalibration might fix it. If AUROC stays weak, the model is failing to rank positives above negatives.
```

Our result:

```text
Recalibration did not rescue AUROC, so the COUGHVID problem is not only threshold mismatch.
```

## 14. What Is Incremental Audio + Metadata Value?

Simple explanation:

```text
This asks whether audio adds useful information after symptoms/metadata are already known.
```

Why someone cares:

```text
If a symptom checklist already performs well, then an audio model must prove it adds extra value. Otherwise, there is no reason to collect audio.
```

Our best symptoms-only aligned example:

| Branch | AUROC |
|---|---:|
| Symptoms-only metadata | `0.888` |
| Audio-only | `0.818` |
| Metadata + audio | `0.951` |

But:

- delta over metadata: `+0.063`.
- confidence interval: `[-0.005, 0.149]`.
- p-value: `0.104`.
- sample size: about `61`.

Meaning:

```text
Audio may add some value over symptoms-only in this small aligned subset, but the evidence is not statistically strong enough to make a big clinical claim.
```

## 15. What Is External Transfer?

Simple explanation:

```text
External transfer means train on one dataset and test on a different dataset.
```

Our setup:

- Train/source: Coswara cough.
- Test/external: COUGHVID cough.

Why someone cares:

```text
A real model should not only work on the dataset it was developed on. It should work on a new dataset collected by different people under different conditions.
```

Our result:

- ComParE+IS10 audio-summary models: `0.523-0.543` AUROC.
- WavLM transformer: `0.484` AUROC.
- CNN-BiGRU: `0.548` AUROC.

Meaning:

```text
The cough signal learned from Coswara did not transfer reliably to COUGHVID.
```

## 16. Simple Final Explanation For A Meeting

Use this if she asks for the entire logic in simple words:

```text
Ma'am, we first built a strong model using cough, breath, and speech. Instead of feeding only raw audio, we measured many sound properties such as duration, loudness, frequency shape, noisiness, and how these change over time. We selected the best 800 measurements using training data only, then trained several models and fusion methods.

The primary equal-weight model was strong inside Coswara, reaching 0.895 AUROC; an exploratory stack reached 0.897. But when we tested the broader workflow more strictly, performance dropped: 0.849 with time-aware participant splitting, about 0.698 from early-to-late time validation, and almost random on COUGHVID. We also tested WavLM and CNN-BiGRU, and they did not solve external transfer.

Then we checked why this happens. Metadata alone predicted labels at 0.964 AUROC, so the dataset has strong shortcuts. The important audio features also changed over time, and COUGHVID partly lies outside the Coswara feature space. So the main conclusion is not that we made a deployable COVID detector. The conclusion is that high internal COVID-audio scores are not enough unless temporal drift, metadata shortcuts, calibration, and external validation are tested.
```


