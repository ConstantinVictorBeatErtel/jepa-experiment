# Paper Assets

Suggested placement in the report:

1. Insert `table_main_results.png` in Section IV as the main quantitative table.
2. Insert `figure_quantitative_results.png` immediately after the quantitative paragraph.
3. Insert `figure_training_curves.png` in the implementation or appendix section.
4. Insert `figure_qualitative_best4.png` in the qualitative analysis subsection.
5. Insert `table_protocol_alignment.png` near the experimental setup subsection to tie the CIFAR-10 experiment back to Assran et al. (2023).

## Ready-to-paste caption text

Table 1. Quantitative comparison on CIFAR-10 under the I-JEPA-style protocol. The JEPA objective outperforms MAE on all downstream metrics, indicating that latent prediction yields stronger semantic representations than direct pixel reconstruction in this setting.

Table 2. Mapping between the original I-JEPA design and the present CIFAR-10 adaptation. The table makes clear that the experiment preserves the core teacher, masking, and predictor design choices from Assran et al. (2023) while changing only the data regime and model scale.

Figure 1. Quantitative comparison between the two self-supervised objectives. JEPA improves over MAE by 14.79 points on linear probe, 15.56 points on k-NN acc@1, and 5.54 points on k-NN acc@5.

Figure 2. Pretraining loss curves for JEPA and MAE. Both objectives optimize stably, but the curves should be read within-objective rather than across-objective because the loss definitions differ.

Figure 3. Best qualitative nearest-neighbor retrieval cases. The JEPA predictor retrieves training images whose object category is more often aligned with the query than the MAE encoder, and the retrieved neighborhoods more closely resemble the true JEPA target neighborhoods.

## Quantitative values

- JEPA linear probe: 71.08%
- MAE linear probe: 56.29%
- JEPA k-NN @1 / @5: 66.15% / 89.13%
- MAE k-NN @1 / @5: 50.59% / 83.59%
- Average JEPA semantic retrieval score: 0.502
