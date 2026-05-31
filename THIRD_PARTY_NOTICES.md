# Third-Party Notices

## IU X-Ray / Open-i Dataset

This repository contains code that can train and evaluate on the Indiana University Chest X-ray Collection distributed through Open-i and mirrored on Kaggle.

Dataset files, radiology reports, X-ray images, and patient-derived qualitative figures are not covered by this repository's Apache-2.0 code license. The IU/Open-i chest X-ray data is distributed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).

To avoid accidentally redistributing adapted patient images, this repository does not commit generated X-ray overlays or patient example figures. Use the scripts in this repository to regenerate qualitative examples locally after downloading the dataset under its original terms.

Recommended dataset citation:

```bibtex
@article{demner2016preparing,
  title={Preparing a collection of radiology examinations for distribution and retrieval},
  author={Demner-Fushman, Dina and Kohli, Marc D and Rosenman, Marc B and Shooshan, Sonya E and Rodriguez, Laritza and Antani, Sameer and Thoma, George R and McDonald, Clement J},
  journal={Journal of the American Medical Informatics Association},
  volume={23},
  number={2},
  pages={304--310},
  year={2016},
  publisher={Oxford University Press}
}
```
