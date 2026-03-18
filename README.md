# tkNodes
Custom nodes for ComfyUI

## Nodes

### LTXV Block Loop Patcher
This node allows you to loop through specified layers in the LTX 2.3 model to potentially improve output quality with an eye toward artifacts in high motion. Typically when sampling, each denoise step makes a full linear pass through the model blocks, and we may choose to use more steps to increase quality, at the cost of more time. Here, we can loop back through some blocks and continue as usual without doing full steps.
A value of 12,16 looks like this with one loop, passing output from block 16 back into block 12 like so. A second loop repeats that loop, and so on.
...10, 11, 12, 13, 14, 16, > 12, 13, 14, 15, 16, > 17, 18...
This works, but it's no magic fix-all solution, sometimes better, sometimes worse. So far, the 10 to 18 range seems to work best, and early in steps. Doing late steps with higher blocks never improved anything in my tests. The general idea behind video model blocks is that early blocks tend to work on rough structure, middle blocks for refined structure, and higher blocks for fine detail.