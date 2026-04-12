# Step 2 Retrieval Diagnostic: fading_memory_ha_gnn_diag_20260410_answer

## Retrieval Inputs
```json
{
  "retrieval_query_zh": "衰减记忆 动态系统 HA-GNN 历史访问 预测机制 区别",
  "retrieval_query_en": "fading memory in dynamic systems vs HA-GNN historical access prediction mechanism",
  "keywords_en": [
    "fading memory",
    "dynamic systems",
    "HA-GNN",
    "historical access information",
    "prediction mechanism"
  ],
  "bm25_query": "fading memory in dynamic systems vs HA-GNN historical access prediction mechanism fading memory dynamic systems HA-GNN historical access information prediction mechanism"
}
```

## Initialization
- initialized: `True`
- initialize error type: ``
- initialize error message: ``

## Main Retrieval
- ok: `True`
- error type: ``
- error message: ``
- doc count: `5`
```json
{
  "branch_counts": {
    "bm25_en": 15,
    "dense_zh": 15,
    "dense_en": 15
  },
  "rrf_pool_count": 20,
  "returned_count": 5,
  "bm25_query": "fading memory in dynamic systems vs HA-GNN historical access prediction mechanism fading memory dynamic systems HA-GNN historical access information prediction mechanism",
  "retrieval_query_zh": "衰减记忆 动态系统 HA-GNN 历史访问 预测机制 区别",
  "retrieval_query_en": "fading memory in dynamic systems vs HA-GNN historical access prediction mechanism"
}
```

## Main Retrieval Chunks

### Chunk 1
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.0465`
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
      "bm25_en": 2,
      "dense_zh": 10,
      "dense_en": 2
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
- score: `0.0454`
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
      "bm25_en": 11,
      "dense_zh": 2,
      "dense_en": 6
    },
    "rrf_score": 0....(truncated)
```
- content preview:
```text
Figure 3. Overview of our workflow, from raw rs-fMRI scans to the binary prediction In this work, we propose a history-aware graph neural network (GNN) model combining two main components: a graph convolutional network (GCN) and an RNN. The GCN is loosely based on the BrainGNN(^{7}) model, having two convolutional blocks. Unlike BrainGNN, our model is capable of working with multiple data points. Each block consists of a GraphSAGE layer, ( e ) a graph normalization layer,(^{15}) a dropout layer, and A topK pooling layer.(^{7}) The RNN component can be any choice of RNN, such as a long short-te...(truncated)
```

### Chunk 3
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.0433`
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
      "bm25_en": 3,
      "dense_zh": 12,
      "dense_en": 14
    },
    "rrf_score": 0...(truncated)
```
- content preview:
```text
Model Acc. AUC-ROC BA CN to MCI MCI to AD HA-GNN (LSTM) (0.829 \pm 0.058) (0.852 \pm 0.065) (0.771 \pm 0.114) (0.688) (0.676) HA-GNN (RNN) (0.743 \pm 0.038) (0.771 \pm 0.067) (0.651 \pm 0.114) (0.5) (0.514) HA-GNN (GRU) (0.733 \pm 0.049) (0.786 \pm 0.062) (0.704 \pm 0.071) (0.625) (0.676) w/o pretraining (0.723 \pm 0.079) (0.642 \pm 0.095) (0.594 \pm 0.061) (0.375) (0.405) Due to our novel methodology, comparisons with the existing literature are difficult. However there are a few papers that are close to our task and methods.
```

### Chunk 4
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.0432`
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
      "bm25_en": 5,
      "dense_zh": 15,
      "dense_en": 9
    },
    "rrf_score": 0....(truncated)
```
- content preview:
```text
. The memory effect of the memristance arises from the dynamics of the internal state variable (2b). In many memristor models the influence of past activity gradually diminishes, and the memristor serves as a canonical example of a FM system [32]. While originally, Boyd an Chua defined FM through the continuity of the system operator with respect to a fading norm [1], our formalization highlights the system's retained memory by relying on its input-to-output contraction properties. To gain some intuition, consider two identical memristors driven by two input currents (I_0) and (I_b) that diffe...(truncated)
```

### Chunk 5
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.0328`
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
      "dense_en": 1
    },
    "rrf_score": 0.032787
  },
  "rrf_score": 0.032787
}
```
- content preview:
```text
. Recurrent neural networks (RNNs) such as long short-term memory (LSTM)(^{9}) and gated recurrent units (GRU)(^{10}) have shown potential in modeling temporal sequences, yet their application to longitudinal neuroimaging data with irregular sampling remains difficult. In this study, we tackle these challenges by introducing a history-aware graph neural network (HA-GNN) model, which integrates rs-fMRI-derived functional connectivity graphs with longitudinal visit information. Our strategy combines a GNN for spatial brain network representation with an RNN for temporal representation, enabling...(truncated)
```

## Plan Reference
```json
{
  "original_query": "fading memory（衰减记忆）’这一性质在动态系统中的作用，与HA-GNN模型中利用历史访问信息进行预测的机制之间，有何相似性与本质区别？",
  "normalized_query_zh": "动态系统的衰减记忆性质与 HA-GNN 模型历史访问预测机制的相似性与本质区别分析",
  "retrieval_query_zh": "衰减记忆 动态系统 HA-GNN 历史访问 预测机制 区别",
  "retrieval_query_en": "fading memory in dynamic systems vs HA-GNN historical access prediction mechanism",
  "crawler_query_en": "fading memory dynamic systems HA-GNN history prediction",
  "keywords_zh": [
    "衰减记忆",
    "动态系统",
    "HA-GNN",
    "历史访问信息",
    "预测机制"
  ],
  "keywords_en": [
    "fading memory",
    "dynamic systems",
    "HA-GNN",
    "historical access information",
    "prediction mechanism"
  ],
  "required_aspects": [
    "definition and role of fading memory in dynamic systems",
    "mechanism of HA-GNN historical access prediction",
    "differences between fading memory in dynamic systems and HA-GNN",
    "similarities between fading memory in dynamic systems and HA-GNN"
  ]
}
```