# Task 3 normalization implementation note

The normalization runs stored with the `dr_grpo` identifier implement the
fixed-denominator component of Dr-GRPO:

- canonical GRPO divides each sequence objective by its realized valid
  response-token count;
- the fixed-denominator variant divides by `max_completion_length`.

Both saved conditions use the same within-group standardized reward advantage:

`(reward - group_mean) / (group_std + eps)`.

Therefore, the saved comparison isolates sequence-length normalization and
should be described in the report as a fixed-denominator GRPO variant retaining
group reward-standard-deviation normalization. It is not a full implementation
of every Dr-GRPO normalization change.

The saved results were not regenerated or relabelled after identifying this
implementation detail.
