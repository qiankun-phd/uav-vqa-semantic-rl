# V0.5 VisDrone Q/A Examples

Each image shows VisDrone object boxes and the generated VQA-style questions derived from annotations.

## Low density example: `0000289_01001_d_0000816`

- annotated objects: `33`
- figure: `outputs/figures/0000289_01001_d_0000816_low_qa_annotated.jpg`

- **presence** Q: Are there awning-tricycle objects in this area? A: `yes`
- **counting** Q: How many awning-tricycle objects are in this area? A: `1`
- **presence** Q: Are there bicycle objects in this area? A: `yes`
- **presence** Q: Are there car objects in this area? A: `yes`
- **counting** Q: How many car objects are in this area? A: `16`
- **presence** Q: Are there motor objects in this area? A: `yes`
- **counting** Q: How many motor objects are in this area? A: `5`
- **presence** Q: Are there pedestrian objects in this area? A: `yes`
- ... 8 more generated Q/A tasks

## Medium density example: `0000289_05001_d_0000836`

- annotated objects: `65`
- figure: `outputs/figures/0000289_05001_d_0000836_medium_qa_annotated.jpg`

- **presence** Q: Are there awning-tricycle objects in this area? A: `yes`
- **counting** Q: How many awning-tricycle objects are in this area? A: `2`
- **presence** Q: Are there bicycle objects in this area? A: `yes`
- **presence** Q: Are there car objects in this area? A: `yes`
- **counting** Q: How many car objects are in this area? A: `19`
- **presence** Q: Are there motor objects in this area? A: `yes`
- **counting** Q: How many motor objects are in this area? A: `9`
- **presence** Q: Are there pedestrian objects in this area? A: `yes`
- ... 8 more generated Q/A tasks

## Dense density example: `0000289_03801_d_0000830`

- annotated objects: `100`
- figure: `outputs/figures/0000289_03801_d_0000830_dense_qa_annotated.jpg`

- **presence** Q: Are there awning-tricycle objects in this area? A: `yes`
- **counting** Q: How many awning-tricycle objects are in this area? A: `1`
- **presence** Q: Are there bicycle objects in this area? A: `yes`
- **presence** Q: Are there bus objects in this area? A: `yes`
- **counting** Q: How many bus objects are in this area? A: `1`
- **presence** Q: Are there car objects in this area? A: `yes`
- **counting** Q: How many car objects are in this area? A: `16`
- **presence** Q: Are there motor objects in this area? A: `yes`
- ... 10 more generated Q/A tasks
