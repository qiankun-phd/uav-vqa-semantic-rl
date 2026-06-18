# Detector-Qwen Diagnostic Report

This report checks whether detector-generated semantic tokens are reliable enough for VQA service control.
Ground-truth annotations are used only for evaluation; the `s=1` Qwen input is detector output.

## Detector Count Quality

- count diagnostic rows: `3377`
- raw detections: `22346`
- count MAE: `5.863`
- zero-detection rate: `0.338`
- under-count ratio: `0.789`
- over-count ratio: `0.119`

## Per-class Count Error

| class | samples | count MAE |
|---|---:|---:|
| pedestrian | 520 | 10.162 |
| people | 482 | 8.297 |
| car | 515 | 7.625 |
| motor | 485 | 6.573 |
| van | 421 | 3.285 |
| tricycle | 337 | 2.534 |
| awning-tricycle | 220 | 2.136 |
| truck | 266 | 2.023 |
| bus | 131 | 1.191 |

## Detector Error vs Qwen Correctness for Semantic-token Evidence

| detector count error bin | Qwen accuracy | samples |
|---|---:|---:|
| 0 | 0.958 | 9405 |
| 1-2 | 0.449 | 9360 |
| 3-5 | 0.368 | 4338 |
| >5 | 0.333 | 3654 |

## Counting Correctness by GT Count Bin

| GT count bin | semantic-token accuracy | samples |
|---|---:|---:|
| 1 | 0.982 | 1962 |
| 2-3 | 0.315 | 1476 |
| 4-9 | 0.178 | 1584 |
| >=10 | 0.311 | 4122 |

## Semantic-token Qwen Accuracy by Channel

| channel | accuracy | samples |
|---|---:|---:|
| bad | 0.543 | 8919 |
| medium | 0.621 | 8919 |
| good | 0.632 | 8919 |

## Counting Failure Examples

- `0000295_01000_d_0000026` channel=bad class=car GT=102 detector=35 transmitted=21 calibrated=49 Pred=`49` Q: How many car objects are in this area?
- `0000295_01000_d_0000026` channel=bad class=car GT=102 detector=35 transmitted=21 calibrated=49 Pred=`49` Q: How many car objects are in this area?
- `0000295_01000_d_0000026` channel=bad class=car GT=102 detector=35 transmitted=21 calibrated=49 Pred=`49` Q: How many car objects are in this area?
- `0000295_01000_d_0000026` channel=medium class=car GT=102 detector=35 transmitted=30 calibrated=30 Pred=`30` Q: How many car objects are in this area?
- `0000295_01000_d_0000026` channel=medium class=car GT=102 detector=35 transmitted=30 calibrated=30 Pred=`30` Q: How many car objects are in this area?
- `0000295_01000_d_0000026` channel=medium class=car GT=102 detector=35 transmitted=30 calibrated=30 Pred=`30` Q: How many car objects are in this area?
- `0000295_01000_d_0000026` channel=good class=car GT=102 detector=35 transmitted=35 calibrated=34 Pred=`34` Q: How many car objects are in this area?
- `0000295_01000_d_0000026` channel=good class=car GT=102 detector=35 transmitted=35 calibrated=34 Pred=`34` Q: How many car objects are in this area?
- `0000295_01000_d_0000026` channel=good class=car GT=102 detector=35 transmitted=35 calibrated=34 Pred=`34` Q: How many car objects are in this area?
- `0000291_05201_d_0000893` channel=bad class=car GT=69 detector=128 transmitted=47 calibrated=111 Pred=`111` Q: How many car objects are in this area?
- `0000291_05201_d_0000893` channel=bad class=car GT=69 detector=128 transmitted=47 calibrated=111 Pred=`111` Q: How many car objects are in this area?
- `0000291_05201_d_0000893` channel=bad class=car GT=69 detector=128 transmitted=47 calibrated=111 Pred=`111` Q: How many car objects are in this area?
- `0000291_05201_d_0000893` channel=medium class=car GT=69 detector=128 transmitted=115 calibrated=115 Pred=`115` Q: How many car objects are in this area?
- `0000291_05201_d_0000893` channel=medium class=car GT=69 detector=128 transmitted=115 calibrated=115 Pred=`115` Q: How many car objects are in this area?
- `0000291_05201_d_0000893` channel=medium class=car GT=69 detector=128 transmitted=115 calibrated=115 Pred=`115` Q: How many car objects are in this area?
