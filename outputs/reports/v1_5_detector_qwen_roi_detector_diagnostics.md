# Detector-Qwen Diagnostic Report

This report checks whether detector-generated semantic tokens are reliable enough for VQA service control.
Ground-truth annotations are used only for evaluation; the `s=1` Qwen input is detector output.

## Detector Count Quality

- count diagnostic rows: `3080`
- raw detections: `3185`
- count MAE: `10.514`
- zero-detection rate: `0.880`
- under-count ratio: `0.958`
- over-count ratio: `0.021`

## Per-class Count Error

| class | samples | count MAE |
|---|---:|---:|
| car | 467 | 26.460 |
| pedestrian | 474 | 15.728 |
| people | 437 | 9.993 |
| motor | 441 | 9.283 |
| van | 384 | 4.740 |
| tricycle | 307 | 2.922 |
| truck | 250 | 2.788 |
| awning-tricycle | 197 | 2.320 |
| bus | 123 | 1.943 |

## Detector Error vs Qwen Correctness for Semantic-token Evidence

| detector count error bin | Qwen accuracy | samples |
|---|---:|---:|
| 0 | 0.818 | 693 |
| 1-2 | 0.773 | 1611 |
| 3-5 | 0.690 | 756 |
| >5 | 0.648 | 486 |

## Semantic-token Qwen Accuracy by Channel

| channel | accuracy | samples |
|---|---:|---:|
| bad | 0.723 | 1182 |
| medium | 0.764 | 1182 |
| good | 0.754 | 1182 |

## Counting Failure Examples

- `0000081_00000_d_0000001` channel=bad class=pedestrian GT=65 detector=96 Pred=`17` Q: How many pedestrian objects are in this area?
- `0000081_00000_d_0000001` channel=bad class=pedestrian GT=65 detector=96 Pred=`17` Q: How many pedestrian objects are in this area?
- `0000081_00000_d_0000001` channel=bad class=pedestrian GT=65 detector=96 Pred=`17` Q: How many pedestrian objects are in this area?
- `0000081_00000_d_0000001` channel=medium class=pedestrian GT=65 detector=96 Pred=`1` Q: How many pedestrian objects are in this area?
- `0000081_00000_d_0000001` channel=medium class=pedestrian GT=65 detector=96 Pred=`1` Q: How many pedestrian objects are in this area?
- `0000081_00000_d_0000001` channel=medium class=pedestrian GT=65 detector=96 Pred=`1` Q: How many pedestrian objects are in this area?
- `0000081_00000_d_0000001` channel=good class=pedestrian GT=65 detector=96 Pred=`100` Q: How many pedestrian objects are in this area?
- `0000081_00000_d_0000001` channel=good class=pedestrian GT=65 detector=96 Pred=`100` Q: How many pedestrian objects are in this area?
- `0000081_00000_d_0000001` channel=good class=pedestrian GT=65 detector=96 Pred=`100` Q: How many pedestrian objects are in this area?
- `0000001_05249_d_0000009` channel=bad class=car GT=76 detector=102 Pred=`yes
1` Q: How many car objects are in this area?
- `0000001_05249_d_0000009` channel=bad class=car GT=76 detector=102 Pred=`yes
1` Q: How many car objects are in this area?
- `0000001_05249_d_0000009` channel=bad class=car GT=76 detector=102 Pred=`yes
1` Q: How many car objects are in this area?
- `0000001_05249_d_0000009` channel=medium class=car GT=76 detector=102 Pred=`111` Q: How many car objects are in this area?
- `0000001_05249_d_0000009` channel=medium class=car GT=76 detector=102 Pred=`111` Q: How many car objects are in this area?
- `0000001_05249_d_0000009` channel=medium class=car GT=76 detector=102 Pred=`111` Q: How many car objects are in this area?
