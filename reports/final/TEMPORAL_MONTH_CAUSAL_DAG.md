# Temporal Month Causal DAG

```text
Calendar month / collection period
  |
  |--> population prevalence and testing behavior --> observed COVID label
  |
  |--> symptom mix and healthcare-seeking behavior --> symptom metadata --> observed COVID label
  |
  |--> recruitment geography and demographics --> participant mix --> observed COVID label
  |
  |--> device, environment, prompt compliance, recording quality --> audio features
  |
  |--> shortcut variable used by source-domain models

COVID pathophysiology --> respiratory audio features --> desired COVID prediction signal
```

The month variable is not interpreted as a biological cause of COVID audio changes. It is a proxy for changing prevalence, recruitment, symptom mix, and recording conditions. A model can therefore exploit month-linked structure during random participant splits, while the same shortcut becomes harmful under early-to-late evaluation.
