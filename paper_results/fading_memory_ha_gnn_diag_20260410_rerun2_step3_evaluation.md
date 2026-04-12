# Step 3 Evaluation Diagnostic: fading_memory_ha_gnn_diag_20260410_rerun2

## Evaluation Input
- input source: `main_retrieval`
- input doc count: `5`

## Evaluation Result
```json
{
  "sufficient": false,
  "partial_answerable": true,
  "aspect_coverage": 1.0,
  "support_strength": 0.5054,
  "noise_ratio": 0.0,
  "covered_aspects": [
    "definition and role of fading memory in dynamic systems",
    "prediction mechanism of HA-GNN using historical access information",
    "conceptual similarities between fading memory and HA-GNN history utilization",
    "fundamental differences between fading memory in dynamic systems and HA-GNN prediction"
  ],
  "weak_aspects": [],
  "missing_aspects": [],
  "aspect_scores": {
    "definition and role of fading memory in dynamic systems": 0.6947,
    "prediction mechanism of HA-GNN using historical access information": 0.4283,
    "conceptual similarities between fading memory and HA-GNN history utilization": 0.3947,
    "fundamental differences between fading memory in dynamic systems and HA-GNN prediction": 0.5038
  },
  "aspect_best_chunks": {
    "definition and role of fading memory in dynamic systems": [
      0,
      2,
      1
    ],
    "prediction mechanism of HA-GNN using historical access information": [
      1,
      4,
      3
    ],
    "conceptual similarities between fading memory and HA-GNN history utilization": [
      0,
      1,
      2
    ],
    "fundamental differences between fading memory in dynamic systems and HA-GNN prediction": [
      0,
      1,
      2
    ]
  },
  "next_action": "retrieve_more",
  "top1_coverage": 0.6947,
  "avg_top3_coverage": 0.5947,
  "unique_sources": 2,
  "reason": "aspect_coverage=1.00, support_strength=0.51, noise_ratio=0.00, missing_aspects_count=0, missing_aspects=none, next_action=retrieve_more"
}
```

## Aspect to Best Chunks
```json
{
  "definition and role of fading memory in dynamic systems": [
    0,
    2,
    1
  ],
  "prediction mechanism of HA-GNN using historical access information": [
    1,
    4,
    3
  ],
  "conceptual similarities between fading memory and HA-GNN history utilization": [
    0,
    1,
    2
  ],
  "fundamental differences between fading memory in dynamic systems and HA-GNN prediction": [
    0,
    1,
    2
  ]
}
```

## Scored Docs

### Chunk 1
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.6947`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2603.23814 [ pdf , ps , other ]",
  "title": "arXiv:2603.23814 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2603.23814",
  "pdf_link": "https://arxiv.org/pdf/2603.23814",
  "source_file": "arXiv2603.23814 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "retrieval_debug": {
    "branch_hits": [
      "bm25_en",
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "bm25_en": 3,
      "dense_zh": 10,
      "dense_en": 1
    },
    "rrf_score": 0....(truncated)
```
- content preview:
```text
## Page 1 State-space fading memory Gustave Bainier(^{a,*}), Antoine Chaillet(^{b}), Rodolphe Sepulchre(^{c,d}), Alessio Franci(^{a,e,*}) (^{a})Dept. of Electrical Engineering and Computer Science, University of Liège, 10 allée de la Découverte, Liège, 4000, Belgium (^{b})Université Paris-Saclay, CNRS, CentraleSupélec, Laboratoire des signaux et systèmes, Gif-sur-Yvette, 91190, France (^{c})KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data Analytics, Kasteel Park Arenberg 10, Leuven, 3001, Belgium (^{d})Department of Engine...(truncated)
```

### Chunk 2
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.4999`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2604.06469 [ pdf , ps , other ]",
  "title": "arXiv:2604.06469 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2604.06469",
  "pdf_link": "https://arxiv.org/pdf/2604.06469",
  "source_file": "arXiv2604.06469 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "retrieval_debug": {
    "branch_hits": [
      "bm25_en",
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "bm25_en": 10,
      "dense_zh": 2,
      "dense_en": 10
    },
    "rrf_score": 0...(truncated)
```
- content preview:
```text
Figure 3. Overview of our workflow, from raw rs-fMRI scans to the binary prediction In this work, we propose a history-aware graph neural network (GNN) model combining two main components: a graph convolutional network (GCN) and an RNN. The GCN is loosely based on the BrainGNN(^{7}) model, having two convolutional blocks. Unlike BrainGNN, our model is capable of working with multiple data points. Each block consists of a GraphSAGE layer, ( e ) a graph normalization layer,(^{15}) a dropout layer, and A topK pooling layer.(^{7}) The RNN component can be any choice of RNN, such as a long short-te...(truncated)
```

### Chunk 3
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.5895`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2603.23814 [ pdf , ps , other ]",
  "title": "arXiv:2603.23814 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2603.23814",
  "pdf_link": "https://arxiv.org/pdf/2603.23814",
  "source_file": "arXiv2603.23814 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "retrieval_debug": {
    "branch_hits": [
      "bm25_en",
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "bm25_en": 7,
      "dense_zh": 12,
      "dense_en": 6
    },
    "rrf_score": 0....(truncated)
```
- content preview:
```text
. The memory effect of the memristance arises from the dynamics of the internal state variable (2b). In many memristor models the influence of past activity gradually diminishes, and the memristor serves as a canonical example of a FM system [32]. While originally, Boyd an Chua defined FM through the continuity of the system operator with respect to a fading norm [1], our formalization highlights the system's retained memory by relying on its input-to-output contraction properties. To gain some intuition, consider two identical memristors driven by two input currents (I_0) and (I_b) that diffe...(truncated)
```

### Chunk 4
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.3844`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2604.06469 [ pdf , ps , other ]",
  "title": "arXiv:2604.06469 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2604.06469",
  "pdf_link": "https://arxiv.org/pdf/2604.06469",
  "source_file": "arXiv2604.06469 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "retrieval_debug": {
    "branch_hits": [
      "bm25_en",
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "bm25_en": 15,
      "dense_zh": 5,
      "dense_en": 14
    },
    "rrf_score": 0...(truncated)
```
- content preview:
```text
. However, most Send correspondence to M.M.: moghaddami@oakland.edu ## Page 2 existing GNN methods focus on cross-sectional data and do not fully utilize the longitudinal aspects of disease progression.(^{8}) Longitudinal prediction poses additional challenges, including irregular time intervals between clinical visits, missing imaging data, and varying lengths of visit history across subjects.(^{8}) These factors complicate the modeling of temporal processes and restrict the applicability of standard sequential or graph-based models
```

### Chunk 5
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.4850`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2604.06469 [ pdf , ps , other ]",
  "title": "arXiv:2604.06469 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2604.06469",
  "pdf_link": "https://arxiv.org/pdf/2604.06469",
  "source_file": "arXiv2604.06469 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "retrieval_debug": {
    "branch_hits": [
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "dense_zh": 1,
      "dense_en": 2
    },
    "rrf_score": 0.032522
  },
  "rrf_score": 0.032522
}
```
- content preview:
```text
. Recurrent neural networks (RNNs) such as long short-term memory (LSTM)(^{9}) and gated recurrent units (GRU)(^{10}) have shown potential in modeling temporal sequences, yet their application to longitudinal neuroimaging data with irregular sampling remains difficult. In this study, we tackle these challenges by introducing a history-aware graph neural network (HA-GNN) model, which integrates rs-fMRI-derived functional connectivity graphs with longitudinal visit information. Our strategy combines a GNN for spatial brain network representation with an RNN for temporal representation, enabling...(truncated)
```