# Task 1 qualitative-evidence review

The Task 1 report requires two distinct pieces of qualitative evidence.

## A. Preference/reward-versus-quality disagreement

Select a case where the numerical signal or preference ordering and
your manual quality assessment do not agree.

Acceptable patterns include:

- a high-reward response that is factually wrong, vague, repetitive,
  unsafe, irrelevant, or poorly follows the instruction;
- a lower-reward response that is manually more accurate or useful;
- a preferred response that is manually worse than the rejected
  response;
- a response that improves preference accuracy or reward while
  visibly degrading another important quality dimension.

Do not call something a disagreement merely because two scores differ.
State the metric or preference claim and the specific quality defect.

## B. Length bias or instruction compliance

Select a case demonstrating one of the following:

- a longer response being preferred without a corresponding quality
  advantage;
- standard DPO producing unnecessary verbosity;
- standard DPO violating a word limit while length-balanced DPO
  complies;
- length-balanced DPO becoming shorter but losing useful content;
- similar-quality responses receiving different treatment because
  their lengths differ.

## Review columns

For promising rows, fill:

- `manual_quality_assessment`
- `metric_or_preference_claim`
- `observed_disagreement`
- `length_or_compliance_finding`
- `include_in_report` with YES or NO
- `final_report_excerpt`
- `report_interpretation`

Quote only the minimum response text needed in the final report.
