# Chapter 3 Theoretical Framework: Prose Audit, Diagram Plan, and Recent Citation Set

## Executive summary

Your revised Chapter 3 now has a much stronger theoretical progression: **clinical problem → predictive models → imbalance handling → TabNet/optimization → evaluation → probability calibration → XAI → calibrated and uncertainty-aware explanations**. The main work remaining is not to add more theory, but to **tighten several claims, reduce figure density in the algorithm section, strengthen 2021–present citations, and make the transition from calibration to XAI to Calibrated Explanations explicit**. The most important technical edits are to avoid treating **FBS \(\geq 100\) mg/dL as a diabetes diagnosis**, correct the wording around \(F_2\), qualify the claim that AUPRC is inherently superior under imbalance, and distinguish **Venn–Abers probabilistic validity from conformal coverage guarantees**. The uploaded Chapter 3 already contains the essential technical material; these recommendations are primarily editorial and evidentiary rather than an expansion into an RRL. fileciteturn0file0

## Structural and prose audit

### Recommended section-level edits

| Section | Assessment | Recommended change |
|---|---|---|
| **3.1 Diabetes and Prediabetes** | **Needs minor tightening** | Keep the medical background concise. Most importantly, call the study target **at-risk/not-at-risk**, not diabetic/non-diabetic. |
| **3.2 Machine Learning and Binary Classification** | **Sufficient** | Keep the supervised-learning and binary-classification introduction short and connect it directly to the operational FBS target. |
| **3.2.1 Traditional ML Algorithms** | **Technically sufficient, visually too dense** | Reduce textbook-level explanations and avoid one diagram for every algorithm. Keep only diagrams that explain genuinely different mechanisms. |
| **3.2.2 Ensemble Learning** | **Needs citations and one visual** | Current distinction between soft voting and stacking is good. Add a recent diabetes/healthcare ensemble citation and one original comparison diagram. |
| **3.2.3 Handling Class Imbalance** | **Strong** | Keep SMOTE equation and existing diagram. Avoid repeating the full F2 explanation here because Section 3.4 handles evaluation. |
| **3.3 Neural Networks and Optimization Strategies** | **Sufficient** | Keep the parent neural-network discussion very short; TabNet is the actual theoretical focus. |
| **3.3.1 TabNet Architecture** | **Strong** | Keep the encoder figure and sequential-attention explanation. Avoid explaining decoder/internal GLU details unless used later. |
| **3.3.2 Bayesian Optimization with Optuna** | **Strong with one terminology check** | Retain TPE explanation if your actual Optuna code uses `TPESampler`. Call it TPE-based sequential/Bayesian optimization rather than describing a Gaussian-process optimizer. |
| **3.4 Model Evaluation Metrics** | **Good, with two technical corrections** | Fix the F2 wording and qualify the AUPRC claim. One confusion-matrix diagram is enough. |
| **3.5 Probability Calibration** | **Strong and central to thesis** | Keep Platt, isotonic, temperature scaling, and calibration assessment. Replace ECE “accuracy” with observed positive frequency for binary risk calibration. |
| **3.6 XAI** | **Good but can be shorter** | SHAP and LIME mechanisms are enough. Do not enumerate every SHAP plot in the theoretical framework. |
| **3.7 Calibrated Explanations and Uncertainty** | **Strongest conceptual contribution, but needs precise terminology** | Tighten Venn–Abers validity language, avoid claiming CE counterfactuals are necessarily globally minimal, and describe Mondrian CE as group-conditional analysis rather than a fairness guarantee. |

### Diabetes and prediabetes

The most important revision in Section 3.1 concerns the meaning of your target.

The ADA's 2026 diagnostic criteria identify fasting plasma glucose of **100–125 mg/dL as prediabetes** and **\(\geq126\) mg/dL as diabetes**, with confirmatory testing generally required in the absence of unequivocal hyperglycemia. Therefore, a thesis target defined as **FBS \(\geq100\) mg/dL** should not be described as “diabetic”; it is better characterized as an operational **at-risk** class encompassing elevated fasting glucose. citeturn18search0

I recommend wording such as:

```latex
For the purposes of this study, respondents with fasting blood sugar (FBS)
values below 100 mg/dL are classified as not at risk, while respondents
with FBS values greater than or equal to 100 mg/dL are classified as at risk.
This operational threshold encompasses elevated fasting glucose associated
with prediabetes and diabetes risk and is used for binary prediction rather
than clinical diagnosis \cite{ada_standards_diagnosis_2026}.
```

Also soften statements such as **“prediabetes is reversible.”** A more defensible formulation is that lifestyle and other interventions can reduce progression to type 2 diabetes, and some individuals may return to normoglycemia. The ADA's current prevention guidance supports intervention to prevent or delay progression. citeturn18search1

### Machine learning, ensembles, and imbalance

Your machine-learning introduction does not need more theory. For recent applied support, Mao et al. used multiple models including Random Forest, LightGBM, XGBoost, AdaBoost, multilayer perceptron, and Gaussian Naive Bayes in diabetes-risk prediction, making it a useful recent study-aligned citation without turning this section into an RRL. citeturn23view0

For **3.2.1**, however, you currently risk over-illustrating the algorithms. Random Forest, kNN, generic gradient boosting, LightGBM, and CatBoost diagrams in succession can make the section look more like an introductory machine-learning textbook than a theoretical framework. I would keep at most:

1. the **kNN diagram**, because its spatial mechanism is difficult to communicate as efficiently in prose; and
2. one **gradient-boosting/tree-growth diagram**, preferably the LightGBM leaf-wise illustration if it genuinely depicts leaf-wise best-first growth.

Your existing generic XGBoost/boosting figure can be removed if the LightGBM figure already establishes sequential boosting. Likewise, CatBoost does not require a figure unless the diagram specifically illustrates ordered boosting or categorical-feature processing.

For **3.2.2 Ensemble Learning**, your voting-versus-stacking distinction is worth retaining. Lu and Uddin's 2022 diabetes-readmission work used a stacking architecture involving Random Forest, AdaBoost, XGBoost, CatBoost, and logistic regression, making it highly appropriate as a recent healthcare citation. citeturn11search0turn11search1 Gollapalli et al. similarly investigated stacking ensembles for diabetes classification. citeturn10search0

For **3.2.3**, your SMOTE discussion is already at the appropriate theoretical depth. Elreedy, Atiya, and Kamalov provide a modern theoretical treatment of SMOTE and its interpolation behavior, while Wah et al. provide the visual illustration you have been considering. citeturn19search0turn20view0

One useful conceptual bridge can be added at the end:

```latex
While SMOTE addresses imbalance during model training, the evaluation of
the resulting classifiers must also account for the importance of correctly
identifying the positive at-risk class. This motivates the recall-sensitive
evaluation measures discussed in Section~\ref{sec:model_evaluation}.
```

That is enough; do not explain \(F_2\) fully twice.

### TabNet and Optuna

Your TabNet subsection is well aligned with the theoretical framework. **Arik and Pfister (2021)** remains the correct primary source because the original TabNet article itself falls within your requested 2021–present range. Joseph, Joseph, and Prasad's 2022 diabetes application is especially valuable because it combines **TabNet, Bayesian optimization, SHAP, and LIME**, closely resembling your study context. citeturn11search5turn11search6

The current TabNet encoder diagram should therefore stay.

Your Optuna discussion is also substantially correct. Bischl et al. provide a contemporary methodological treatment of hyperparameter optimization, while Optuna's official `TPESampler` documentation explains the two density models conventionally denoted \(l(x)\) and \(g(x)\), with new configurations selected according to their relative densities. citeturn8search4turn8search5 Current Optuna documentation also identifies TPE as the default sampler when no sampler is supplied to `create_study`. citeturn9search0

I would only change:

> “Optuna's default Tree-structured Parzen Estimator…”

to:

> “Optuna's Tree-structured Parzen Estimator (TPE) sampler…”

unless your source code explicitly relies on the default. This keeps the theoretical framework tied to **what you actually configured**, rather than to a software default that could change.

### Evaluation metrics

Precision and recall are fine as written. Retain their equations.

For \(F_2\), change:

> “F2 weights recall twice as heavily as precision.”

That is an imprecise interpretation of \(\beta=2\). Use:

```latex
The F2-score is obtained by setting $\beta=2$ in the generalized
$F_\beta$ measure, giving greater emphasis to recall than precision.
This weighting is appropriate for the study's early-risk screening
objective, where failing to identify an at-risk individual is considered
more consequential than generating an additional false-positive flag.
```

Keep both the generalized \(F_\beta\) formula and the simplified \(F_2\) formula.

For **AUPRC**, avoid a categorical statement such as:

> “AUPRC is better than AUROC for imbalanced datasets.”

Recent work has emphasized that neither AUROC nor AUPRC is universally superior solely because a dataset is imbalanced; AUPRC is particularly useful when the precision–recall trade-off for the positive class is the substantive focus, and its baseline depends on prevalence. citeturn7academia51

A stronger formulation for your thesis is:

```latex
AUPRC summarizes the trade-off between precision and recall across
classification thresholds. In this study, it is particularly relevant because
evaluation focuses on the model's ability to identify the positive at-risk
class while controlling false-positive predictions. Unlike a fixed baseline
such as 0.5 for AUROC, the expected AUPRC of a random classifier is related
to the prevalence of the positive class.
```

### Probability calibration

This is now one of your best-organized sections. Ojeda et al.'s comprehensive 2023 comparison is an excellent recent source covering probability calibration methods and their evaluation. citeturn16search0turn16search2

Keep all three:

- Platt Scaling
- Isotonic Regression
- Temperature Scaling

Temperature scaling is particularly important because it makes the theoretical framework match the calibration methods actually compared in your study. Recent methodological work by Balanya, Maroñas, and Ramos directly addresses temperature-scaling-based neural-network calibration. citeturn17search0 A recent medical-imaging calibration study by Penso, Frenkel, and Goldberger also provides healthcare-specific support for post-hoc confidence calibration. citeturn16search1

For ECE, I would change notation from:

\[
\operatorname{acc}(B_m)
\]

to something such as:

\[
\operatorname{obs}(B_m)
\]

where \(\operatorname{obs}(B_m)\) is the **observed positive-class frequency** in bin \(B_m\). For a binary diabetes-risk probability, this is conceptually clearer than speaking of “accuracy” inside each bin.

So:

```latex
\begin{equation}
\mathrm{ECE} =
\sum_{m=1}^{M}
\frac{|B_m|}{N}
\left|
\mathrm{obs}(B_m)-\mathrm{conf}(B_m)
\right|,
\end{equation}
```

Also retain your distinction between **discrimination** and **calibration**. That distinction is central to why the next sections exist.

### XAI and Calibrated Explanations

For SHAP, you no longer need the long list of force plots, summary plots, dependence plots, TreeExplainer, and KernelExplainer in Chapter 3. Your theoretical framework should answer:

> What is SHAP? What does a SHAP value mean? What can it explain in this thesis?

Recent healthcare examples provide sufficient contemporary grounding. Du et al. used SHAP in an explainable clinical decision-support system for gestational diabetes, and Mao et al. used SHAP in diabetes-risk modeling. citeturn10search1turn23view0

For LIME, Wang, Chen, and Chiu's 2023 diabetes study is particularly suitable because LIME was explicitly incorporated into an explainable healthcare framework. citeturn12search0turn12search1 Lu and Uddin's diabetes-readmission stacking study provides another recent model-specific example. citeturn11search0

Your direct SHAP/LIME comparison is worth retaining:

> SHAP provides theoretically grounded feature attribution and supports both local and aggregated interpretation, whereas LIME approximates local behavior through a simpler surrogate.

That is more useful here than several paragraphs listing visualization types.

For Section 3.7, use **Löfström et al. (2024)** as the central citation. Their Calibrated Explanations work explicitly combines uncertainty information and counterfactual explanations. citeturn14search4turn14search5

Three technical wording changes are advisable:

**Venn–Abers.** Do not describe the pair \([p_0,p_1]\) as if it were a conventional conformal prediction interval with a coverage probability. Recent Venn–Abers work characterizes the method as a distribution-free probabilistic framework for calibrated predictive distributions/multiprobability predictions under weak assumptions. citeturn14search2turn15view2

Use:

```latex
Venn--Abers predictors produce multiprobability outputs that provide
calibrated probabilistic information under exchangeability assumptions.
The separation between the probability bounds can additionally be used
as an indication of predictive uncertainty.
```

rather than:

> “The interval has a finite-sample coverage guarantee.”

**Counterfactual CE.** Unless your implementation explicitly performs an optimization that proves minimality, replace:

> “identifies the minimal change required…”

with:

> “identifies feature changes or alternative conditions associated with a change in the calibrated prediction.”

The Calibrated Explanations framework supports counterfactual reasoning, but the word **minimal** implies a specific optimization property that should only be claimed if your implemented search enforces it. citeturn14search5

**Mondrian CE.** Do not say it “ensures fairness.” Conditional Calibrated Explanations can support subgroup-conditional uncertainty analysis and can be useful for investigating bias, but that is not equivalent to proving algorithmic fairness. Löfström and Löfström's 2024 work explicitly examines conditional explanations in relation to bias and uncertainty. citeturn14search6turn14search10

A better ending is:

```latex
In this study, biological sex is used as a conditioning category to examine
whether calibrated predictions, uncertainty estimates, and explanations behave
consistently across male and female respondents. This provides a
fairness-oriented subgroup analysis without treating group-conditional
calibration alone as evidence of overall algorithmic fairness.
```

## Diagram plan

The goal should be **fewer, more explanatory figures**, not a diagram for every named technique.

| Section | Diagram recommendation | Exact placement | Suggested caption | Width |
|---|---|---|---|---|
| **3.1** | **Add one original threshold diagram** | Immediately after FPG/prediabetes thresholds | “Fasting plasma glucose ranges used to contextualize the study's binary at-risk classification.” | `0.78\linewidth` |
| **3.2** | None | — | — | — |
| **3.2.1** | **Do not add more. Prune existing figures.** | Keep kNN and one boosting/tree-growth figure | Existing captions can remain after corrections | `0.60–0.75\linewidth` |
| **3.2.2** | **Add one original voting-vs-stacking schematic** | After stacking paragraph | “Conceptual comparison of soft voting and stacking ensemble strategies.” | `0.85\linewidth` |
| **3.2.3** | **Keep current SMOTE figure** | Immediately after SMOTE mechanism paragraph | “Generation of synthetic minority-class samples using SMOTE.” | `0.65–0.70\linewidth` |
| **3.3** | None | — | — | — |
| **3.3.1** | **Keep TabNet encoder** | After first TabNet mechanism paragraph | “TabNet encoder architecture showing sequential attentive feature selection across decision steps.” | `0.85–0.90\linewidth` |
| **3.3.2** | **Optional original TPE workflow** | After paragraph explaining TPE | “Sequential TPE-based hyperparameter optimization workflow used for BO-TabNet.” | `0.78\linewidth` |
| **3.4** | **Add one original confusion matrix** | Immediately before Precision/Recall formulas | “Confusion-matrix outcomes underlying precision and recall for the positive at-risk class.” | `0.58–0.65\linewidth` |
| **3.5** | **Add one original reliability diagram** | After calibration-vs-discrimination introduction | “Conceptual reliability diagram showing agreement between predicted probabilities and observed event frequencies.” | `0.68–0.75\linewidth` |
| **3.6.1** | Optional SHAP local/global schematic | After final SHAP paragraph | “Local and aggregated interpretation of feature contributions using SHAP.” | `0.80\linewidth` |
| **3.6.2** | Optional LIME schematic | After LIME mechanism paragraph | “Local surrogate approximation of a complex model using proximity-weighted perturbations.” | `0.75\linewidth` |
| **3.7** | **Strongly recommended original CE pipeline** | After 3.7.2 Calibrated Explanations | “Conceptual flow from model prediction and Venn–Abers calibration to uncertainty-aware factual and counterfactual explanations.” | `0.92\linewidth` |

### Most valuable new figures

If you only add **four** figures from this list, prioritize:

1. voting versus stacking;
2. confusion matrix;
3. reliability diagram;
4. Calibrated Explanations pipeline.

Those four explain relationships that prose alone does not communicate as efficiently.

Your SMOTE and TabNet figures already perform the same role in their respective sections.

### Drop-in figure blocks

For the **ensemble diagram**, create it yourself rather than copying another paper:

```latex
% Place immediately after the stacking ensemble paragraph.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\linewidth]
    {Diagrams/voting_stacking_comparison.pdf}
    \caption{Conceptual comparison of soft voting and stacking ensemble
    strategies. Source: Authors' illustration.}
    \label{fig:voting_stacking}
\end{figure}
```

For the **confusion matrix**:

```latex
% Place immediately before the Precision and Recall equations.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.62\linewidth]
    {Diagrams/confusion_matrix.pdf}
    \caption{Confusion-matrix outcomes underlying precision and recall for
    the positive at-risk class. Source: Authors' illustration.}
    \label{fig:confusion_matrix}
\end{figure}
```

For the **calibration diagram**:

```latex
% Place after the paragraph distinguishing calibration from discrimination.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.72\linewidth]
    {Diagrams/reliability_diagram.pdf}
    \caption{Conceptual reliability diagram showing agreement between
    predicted probabilities and observed event frequencies.
    Authors' illustration based on \cite{ojeda_calibration_2023}.}
    \label{fig:reliability_diagram}
\end{figure}
```

For the **TPE workflow**:

```latex
% Place after the paragraph describing the TPE sampler.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.78\linewidth]
    {Diagrams/optuna_tpe_workflow.pdf}
    \caption{Sequential TPE-based hyperparameter optimization workflow used
    for BO-TabNet. Authors' illustration based on the Optuna
    TPESampler documentation \cite{optuna_tpe}.}
    \label{fig:optuna_tpe}
\end{figure}
```

For the **Calibrated Explanations pipeline**:

```latex
% Place after the main Calibrated Explanations subsection,
% before Counterfactual Calibrated Explanations.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\linewidth]
    {Diagrams/calibrated_explanations_pipeline.pdf}
    \caption{Conceptual flow from model prediction and Venn--Abers
    calibration to uncertainty-aware factual and counterfactual
    explanations. Authors' illustration based on
    L{\"o}fstr{\"o}m et al. \cite{lofstrom_ce_2024}.}
    \label{fig:calibrated_explanations}
\end{figure}
```

For your **existing SMOTE figure**, Wah et al.'s 2023 paper explicitly presents the synthetic-instance generation mechanism through interpolation and nearest minority neighbors. citeturn20view0 If you redraw it:

```latex
\caption{Generation of synthetic minority-class samples using SMOTE.
Adapted from Wah et al. \cite{wah_smote_2023}.}
```

For TabNet, continue using the **encoder only**, rather than all four panels from the original TabNet architecture figure.

## Recent citation set

The following is the **minimal set** I would actually use in Chapter 3. It is deliberately much smaller than an RRL bibliography.

| Chapter 3 location | Recommended recent source(s) | Why it belongs here |
|---|---|---|
| **3.1 Diabetes and Prediabetes** | ADA Professional Practice Committee, 2026, *Diagnosis and Classification of Diabetes*; ADA, 2026, *Prevention or Delay of Diabetes* | Current clinical thresholds and prevention context. citeturn18search0turn18search1 |
| **3.2 / 3.2.1 ML algorithms** | Mao et al., 2023, *Journal of Diabetes Investigation*, DOI `10.1111/jdi.13937` | Recent diabetes application of RF, LightGBM, XGBoost, AdaBoost, NB and SHAP. citeturn23view0 |
| **3.2.2 Ensemble Learning** | Lu & Uddin, 2022, *Information*, 13(9):436, DOI `10.3390/info13090436`; Gollapalli et al., 2022, *Computers in Biology and Medicine*, 147:105757, DOI `10.1016/j.compbiomed.2022.105757` | Recent diabetes-specific stacking applications. citeturn11search0turn10search0 |
| **3.2.3 SMOTE** | Elreedy, Atiya & Kamalov, 2024, *Machine Learning*, 113:4903–4923, DOI `10.1007/s10994-022-06296-4`; Wah et al., 2023, *Computers, Materials & Continua*, 75(3):4821–4841, DOI `10.32604/cmc.2023.034470` | Modern theoretical SMOTE source plus source for the selected diagram. citeturn19search0turn20view0 |
| **3.3.1 TabNet** | Arik & Pfister, 2021, *AAAI*, 35(8):6679–6687; Joseph, Joseph & Prasad, 2022, *Computers in Biology and Medicine*, 151:106178, DOI `10.1016/j.compbiomed.2022.106178` | Original TabNet is already 2021; Joseph et al. is directly diabetes + BO-TabNet + XAI. citeturn11search5 |
| **3.3.2 Optuna / BO** | Bischl et al., 2023, *WIREs Data Mining and Knowledge Discovery*, 13(2):e1484, DOI `10.1002/widm.1484`; Optuna `TPESampler` documentation; Joseph et al., 2022 | Methodological HPO theory, actual TPE mechanics, and a directly aligned diabetes application. citeturn8search4turn11search5 |
| **3.4 Precision / Recall / F2** | Use one of the above recent diabetes-ML applications as context; no separate paper is required merely to justify the standard algebraic definitions | Avoid bibliography inflation. |
| **3.4 AUPRC** | McDermott et al., 2024, *A Closer Look at AUROC and AUPRC under Class Imbalance*, arXiv:2401.06091 | Useful for avoiding the oversimplified “AUPRC is always better for imbalance” claim. citeturn7academia51 |
| **3.5 Platt / Isotonic / Assessment** | Ojeda et al., 2023, *Statistics in Medicine*, 42(29):5451–5478, DOI `10.1002/sim.9921` | Contemporary comparison of probability calibration approaches. citeturn16search0 |
| **3.5.3 Temperature Scaling** | Balanya, Maroñas & Ramos, 2024, *Neural Computing and Applications*, 36:8073–8095, DOI `10.1007/s00521-024-09505-4`; Penso, Frenkel & Goldberger, 2024, *IEEE TMI*, 43(6):2050–2060, DOI `10.1109/TMI.2024.3353762` | Modern temperature calibration plus a healthcare application. citeturn17search0turn16search1 |
| **3.6.1 SHAP** | Du et al., 2022, *Scientific Reports*, 12:1170, DOI `10.1038/s41598-022-05112-2`; Mao et al., 2023 | Recent SHAP use in diabetes-related healthcare prediction. citeturn10search1turn23view0 |
| **3.6.2 LIME** | Wang, Chen & Chiu, 2023, *Healthcare Analytics*, 3:100183, DOI `10.1016/j.health.2023.100183`; Lu & Uddin, 2022 | Recent LIME use in diabetes/healthcare prediction. citeturn12search1turn11search0 |
| **3.7.1 Venn–Abers** | Nouretdinov & Gammerman, 2024, *PMLR*, 230:490–507 | Recent direct Venn–Abers methodology source. citeturn14search2turn15view2 |
| **3.7.2 CE** | Löfström et al., 2024, *Expert Systems with Applications*, 246:123154, DOI `10.1016/j.eswa.2024.123154` | Core Calibrated Explanations source. citeturn14search5 |
| **3.7.3 Counterfactual CE** | Löfström et al., 2024, same paper | Counterfactuals are part of the CE framework itself. citeturn14search4 |
| **3.7.4 Mondrian / Conditional CE** | Löfström & Löfström, 2024, *Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty*, CCIS 2153:332–355, DOI `10.1007/978-3-031-63787-2_17` | Best direct recent source for conditional/group-specific CE. citeturn14search6turn14search10 |

This means you can support nearly the entire Chapter 3 with roughly **a dozen core sources**, with individual references reused where they genuinely support multiple concepts.

## Drop-in BibTeX set

The following entries cover the high-priority additions. For items where the retrieved bibliographic record did not expose every author field, I have explicitly marked that metadata for verification rather than inventing it.

```bibtex
@article{ada_standards_diagnosis_2026,
  author  = {{American Diabetes Association Professional Practice Committee for Diabetes}},
  title   = {2. Diagnosis and Classification of Diabetes:
             Standards of Care in Diabetes---2026},
  journal = {Diabetes Care},
  year    = {2026},
  volume  = {49},
  number  = {Supplement_1},
  doi     = {10.2337/dc26-s002}
}

@article{ada_prevention_2026,
  author  = {{American Diabetes Association Professional Practice Committee for Diabetes}},
  title   = {3. Prevention or Delay of Diabetes and Associated Comorbidities:
             Standards of Care in Diabetes---2026},
  journal = {Diabetes Care},
  year    = {2026},
  volume  = {49},
  number  = {Supplement_1},
  pages   = {S50--S60},
  doi     = {10.2337/dc26-s003}
}

@article{mao_diabetes_ml_2023,
  author  = {Mao, Yaqian and Zhu, Zheng and Pan, Shuyao and Lin, Wei
             and Liang, Jixing and Huang, Huibin and Li, Liantao
             and Wen, Junping and Chen, Gang},
  title   = {Value of Machine Learning Algorithms for Predicting Diabetes Risk:
             A Subset Analysis from a Real-World Retrospective Cohort Study},
  journal = {Journal of Diabetes Investigation},
  year    = {2023},
  volume  = {14},
  number  = {2},
  pages   = {309--320},
  doi     = {10.1111/jdi.13937}
}

@article{lu_uddin_stacking_2022,
  author  = {Lu, Haohui and Uddin, Shahadat},
  title   = {Explainable Stacking-Based Model for Predicting Hospital
             Readmission for Diabetic Patients},
  journal = {Information},
  year    = {2022},
  volume  = {13},
  number  = {9},
  pages   = {436},
  doi     = {10.3390/info13090436}
}

% Verify the complete author list before final submission.
@article{gollapalli_stacking_diabetes_2022,
  author  = {Gollapalli, Mohammed and others},
  title   = {A Novel Stacking Ensemble for Detecting Three Types of Diabetes
             Mellitus Using a Saudi Arabian Dataset:
             Pre-Diabetes, T1DM, and T2DM},
  journal = {Computers in Biology and Medicine},
  year    = {2022},
  volume  = {147},
  pages   = {105757},
  doi     = {10.1016/j.compbiomed.2022.105757}
}

@article{elreedy_smote_2024,
  author  = {Elreedy, Dina and Atiya, Amir F. and Kamalov, Firuz},
  title   = {A Theoretical Distribution Analysis of Synthetic Minority
             Oversampling Technique (SMOTE) for Imbalanced Learning},
  journal = {Machine Learning},
  year    = {2024},
  volume  = {113},
  pages   = {4903--4923},
  doi     = {10.1007/s10994-022-06296-4}
}

@article{wah_smote_2023,
  author  = {Yap, Bee Wah and Ismail, Azlan and Azid, Nur Niswah Naslina
             and Jaafar, Jafreezal and Abdul Aziz, Izzatdin
             and Hasan, Mohd Hilmi and Mohamad Zain, Jasni},
  title   = {Machine Learning and Synthetic Minority Oversampling Techniques
             for Imbalanced Data: Improving Machine Failure Prediction},
  journal = {Computers, Materials \& Continua},
  year    = {2023},
  volume  = {75},
  number  = {3},
  pages   = {4821--4841},
  doi     = {10.32604/cmc.2023.034470}
}

@inproceedings{arik_pfister_tabnet_2021,
  author    = {Arik, Sercan {\"O}. and Pfister, Tomas},
  title     = {TabNet: Attentive Interpretable Tabular Learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2021},
  volume    = {35},
  number    = {8},
  pages     = {6679--6687},
  doi       = {10.1609/aaai.v35i8.16826}
}

@article{joseph_tabnet_diabetes_2022,
  author  = {Joseph, Lionel P. and Joseph, Erica A. and Prasad, Ramendra},
  title   = {Explainable Diabetes Classification Using Hybrid
             Bayesian-Optimized TabNet Architecture},
  journal = {Computers in Biology and Medicine},
  year    = {2022},
  volume  = {151},
  pages   = {106178},
  doi     = {10.1016/j.compbiomed.2022.106178}
}

@article{bischl_hpo_2023,
  author  = {Bischl, Bernd and Binder, Martin and Lang, Michel
             and Pielok, Tobias and Richter, Jakob and Coors, Stefan
             and Thomas, Janek and Ullmann, Theresa and Becker, Marc
             and Boulesteix, Anne-Laure and Deng, Difan
             and Lindauer, Marius},
  title   = {Hyperparameter Optimization: Foundations, Algorithms,
             Best Practices, and Open Challenges},
  journal = {WIREs Data Mining and Knowledge Discovery},
  year    = {2023},
  volume  = {13},
  number  = {2},
  pages   = {e1484},
  doi     = {10.1002/widm.1484}
}

@misc{optuna_tpe,
  author       = {{Optuna Developers}},
  title        = {TPESampler},
  organization = {Optuna},
  year         = {2026},
  howpublished = {\url{https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html}},
  note         = {Accessed 2026-10-07}
}

@article{mcdermott_auprc_2024,
  author  = {McDermott, Matthew B. A. and Zhang, Haoran
             and Hansen, Lasse Hyldig and Angelotti, Giovanni
             and Gallifant, Jack},
  title   = {A Closer Look at AUROC and AUPRC under Class Imbalance},
  journal = {arXiv preprint arXiv:2401.06091},
  year    = {2024}
}

@article{ojeda_calibration_2023,
  author  = {Ojeda, Francisco M. and Jansen, Max L. and Thi{\'e}ry, Alexandre
             and Blankenberg, Stefan and Weimar, Christian
             and Schmid, Matthias and Ziegler, Andreas},
  title   = {Calibrating Machine Learning Approaches for Probability
             Estimation: A Comprehensive Comparison},
  journal = {Statistics in Medicine},
  year    = {2023},
  volume  = {42},
  number  = {29},
  pages   = {5451--5478},
  doi     = {10.1002/sim.9921}
}

@article{balanya_temperature_2024,
  author  = {Balanya, Sergio A. and Maro{\~n}as, Juan and Ramos, Daniel},
  title   = {Adaptive Temperature Scaling for Robust Calibration of
             Deep Neural Networks},
  journal = {Neural Computing and Applications},
  year    = {2024},
  volume  = {36},
  pages   = {8073--8095},
  doi     = {10.1007/s00521-024-09505-4}
}

@article{penso_medical_calibration_2024,
  author  = {Penso, Coby and Frenkel, Lior and Goldberger, Jacob},
  title   = {Confidence Calibration of a Medical Imaging Classification
             System That Is Robust to Label Noise},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2024},
  volume  = {43},
  number  = {6},
  pages   = {2050--2060},
  doi     = {10.1109/TMI.2024.3353762}
}

@article{du_gdm_shap_2022,
  author  = {Du, Yuhan and Rafferty, Anthony R. and McAuliffe, Fionnuala M.
             and Wei, Lan and Mooney, Catherine},
  title   = {An Explainable Machine Learning-Based Clinical Decision
             Support System for Prediction of Gestational Diabetes Mellitus},
  journal = {Scientific Reports},
  year    = {2022},
  volume  = {12},
  pages   = {1170},
  doi     = {10.1038/s41598-022-05112-2}
}

@article{wang_diabetes_lime_2023,
  author  = {Wang, Yu-Cheng and Chen, Tin-Chih Toly and Chiu, Min-Chi},
  title   = {A Systematic Approach to Enhance the Explainability of
             Artificial Intelligence in Healthcare with Application
             to Diagnosis of Diabetes},
  journal = {Healthcare Analytics},
  year    = {2023},
  volume  = {3},
  pages   = {100183},
  doi     = {10.1016/j.health.2023.100183}
}

@article{lofstrom_ce_2024,
  author  = {L{\"o}fstr{\"o}m, Helena and L{\"o}fstr{\"o}m, Tuwe
             and Johansson, Ulf and S{\"o}nstr{\"o}d, Cecilia},
  title   = {Calibrated Explanations:
             With Uncertainty Information and Counterfactuals},
  journal = {Expert Systems with Applications},
  year    = {2024},
  volume  = {246},
  pages   = {123154},
  doi     = {10.1016/j.eswa.2024.123154}
}

@inproceedings{nouretdinov_venn_abers_2024,
  author    = {Nouretdinov, Ilia and Gammerman, James},
  title     = {Inductive Venn-Abers Predictive Distributions:
               New Applications \& Evaluation},
  booktitle = {Proceedings of the Thirteenth Symposium on Conformal
               and Probabilistic Prediction with Applications},
  series    = {Proceedings of Machine Learning Research},
  volume    = {230},
  pages     = {490--507},
  year      = {2024}
}

@inproceedings{lofstrom_conditional_ce_2024,
  author    = {L{\"o}fstr{\"o}m, Helena and L{\"o}fstr{\"o}m, Tuwe},
  title     = {Conditional Calibrated Explanations:
               Finding a Path Between Bias and Uncertainty},
  booktitle = {Explainable Artificial Intelligence},
  series    = {Communications in Computer and Information Science},
  volume    = {2153},
  pages     = {332--355},
  year      = {2024},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-63787-2_17}
}
```

The strongest economy of citation is to **reuse the appropriate source rather than finding a new paper for every paragraph**. For example, Ojeda et al. can support the calibration introduction, Platt scaling, isotonic regression, and reliability-diagram discussion; Löfström et al. can support CE and CCE; Joseph et al. can support both BO-TabNet and the use of SHAP/LIME in a diabetes setting. citeturn16search0turn14search5turn11search5

## Equations and LaTeX checklist

### Equations to retain in Chapter 3

I recommend keeping these in the main theoretical framework because they directly explain study design choices:

\[
x_{\text{new}}=x_i+\lambda(x_j-x_i)
\]

for SMOTE;

\[
\text{Precision}=\frac{TP}{TP+FP},
\qquad
\text{Recall}=\frac{TP}{TP+FN}
\]

for screening performance;

\[
F_\beta=(1+\beta^2)
\frac{\text{Precision}\cdot\text{Recall}}
{\beta^2\text{Precision}+\text{Recall}}
\]

and the \(F_2\) specialization;

\[
P(y=1\mid x)
=
\frac{1}{1+\exp(Af(x)+B)}
\]

for Platt scaling;

\[
P(y=1\mid x)=
\sigma\!\left(\frac{z(x)}{T}\right)
\]

for temperature scaling;

and the Brier Score and ECE equations.

Those formulas correspond directly to concepts that are **central to your methodology or selection criteria**.

### Equations that are optional

The **full SHAP combinatorial formula** is theoretically correct, but it is the first equation I would remove or move to an appendix if Chapter 3 becomes too dense. A conceptual explanation of marginal feature contributions plus a SHAP diagram is sufficient for an undergraduate theoretical framework.

Likewise, the Naive Bayes equation and Euclidean kNN distance formula are useful but less essential than the \(F_2\) and calibration equations. If page count becomes a problem, retain the latter and remove the former.

For Calibrated Explanations, do **not** add several additional equations simply to make the section appear mathematical. The probability pair and one concise description of uncertainty are more useful than a long derivation.

### LaTeX cleanup

Your preamble should contain at least:

```latex
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{url}
```

`booktabs` is useful for thesis tables:

```latex
\usepackage{booktabs}
```

If floats from one section tend to cross into the next despite not using `[H]`, `placeins` provides `\FloatBarrier`:

```latex
\usepackage{placeins}
```

For your Chapter 3 figures, prefer:

```latex
\includegraphics[width=0.75\linewidth]{...}
```

rather than:

```latex
\includegraphics[width=0.75\textwidth]{...}
```

`\linewidth` adapts correctly if the figure is later placed in a narrower environment.

Use `[H]` only where you genuinely require the figure immediately after its explanation:

```latex
\begin{figure}[H]
```

Because `[H]` prevents floating, too many large `[H]` figures can create substantial white space. This reinforces the recommendation to **remove unnecessary algorithm figures instead of forcing all of them into position**.

Keep labels immediately after captions:

```latex
\caption{...}
\label{fig:...}
```

Avoid spaces in filenames. Prefer:

```text
Diagrams/random_forest.png
Diagrams/tabnet_encoder.png
Diagrams/reliability_diagram.pdf
```

rather than:

```text
Diagrams/Random Forest.png
```

For copied figures use:

```latex
Source: Author \cite{key}.
```

For figures you redraw or simplify:

```latex
Adapted from Author \cite{key}.
```

For diagrams you construct yourself based on the theory:

```latex
Source: Authors' illustration.
```

or:

```latex
Authors' illustration based on \cite{key}.
```

Finally, remove Markdown syntax still embedded in LaTeX. For example:

```latex
\textbf{**Summary plots**}
```

must become:

```latex
\textbf{Summary plots}
```

The final Chapter 3 should therefore emphasize **only the theory necessary to explain your integrated research logic**:

\[
\boxed{\text{Risk Prediction}}
\rightarrow
\boxed{\text{Recall-Sensitive Evaluation}}
\rightarrow
\boxed{\text{Probability Calibration}}
\rightarrow
\boxed{\text{SHAP/LIME Interpretation}}
\rightarrow
\boxed{\text{Calibrated Explanations}}
\rightarrow
\boxed{\text{Counterfactual and Conditional Analysis}}.
\]

That progression is now much stronger than the reference theoretical frameworks you previously examined, because it explains **why each component is needed by the next**, rather than presenting the algorithms as an unrelated catalogue of techniques.