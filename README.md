# equiflow

***Under construction!***

*equiflow* is a package designed to generate "Equity-focused Cohort Selection Flow Diagrams". We hope to facilitate research, increase its reproducibility, and improve the transparency of the process of cohort curation in machine learning studies.


## Vision
*equiflow* will provide tabular and visual representations of inclusion and exclusion criteria applied to a clinical dataset. Each patient exclusion step can depict the cohort composition across demographics and outcomes, to interrogate potential sampling selection bias.

This package is designed to enhance the transparency and reproducibility of research in the medical machine learning field. It complements other tools like tableone, which is used for generating summary statistics for patient populations.


## Citation
The concept was first introuced in our [position paper](https://www.sciencedirect.com/science/article/pii/S1532046424000492).

> Ellen JG, Matos J, Viola M, et al. Participant flow diagrams for health equity in AI. J Biomed Inform. 2024;152:104631. [https://doi.org/10.1016/j.jbi.2024.104631](https://doi.org/10.1016/j.jbi.2024.104631)


## Motivation

Selection bias can arise through many aspects of a study, including recruitment, inclusion/exclusion criteria, input-level exclusion and outcome-level exclusion, and often reflects the underrepresentation of populations historically disadvantaged in medical research. The effects of selection bias can be further amplified when non-representative samples are used in artificial intelligence (AI) and machine learning (ML) applications to construct clinical algorithms. Building on the “Data Cards” initiative for transparency in AI research, we advocate for the addition of a **participant flow diagram for AI studies detailing relevant sociodemographic** and/or clinical characteristics of excluded participants across study phases, with the goal of identifying potential algorithmic biases before their clinical implementation. We include both a model for this flow diagram as well as a brief case study explaining how it could be implemented in practice. Through standardized reporting of participant flow diagrams, we aim to better identify potential inequities embedded in AI applications, facilitating more reliable and equitable clinical algorithms.

