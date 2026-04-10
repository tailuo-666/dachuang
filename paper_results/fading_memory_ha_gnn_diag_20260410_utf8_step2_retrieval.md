# Step 2 Retrieval Diagnostic: fading_memory_ha_gnn_diag_20260410_utf8

## Retrieval Inputs
```json
{
  "retrieval_query_zh": "动态系统 衰减记忆 作用 HA-GNN 历史访问 预测 相似性 本质区别",
  "retrieval_query_en": "role of fading memory in dynamic systems versus HA-GNN historical access prediction mechanism",
  "keywords_en": [
    "HA-GNN",
    "fading memory",
    "dynamic systems",
    "historical access information",
    "prediction mechanism"
  ],
  "bm25_query": "role of fading memory in dynamic systems versus HA-GNN historical access prediction mechanism HA-GNN fading memory dynamic systems historical access information prediction mechanism"
}
```

## Initialization
- initialized: `True`
- initialize error type: ``
- initialize error message: ``

## Main Retrieval
- ok: `False`
- error type: `TypeError`
- error message: `'VLLMOpenAIEmbeddings' object is not callable`
- doc count: `0`
```json
{}
```

## Branch Fallback

### bm25_en
- ok: `True`
- query: `role of fading memory in dynamic systems versus HA-GNN historical access prediction mechanism HA-GNN fading memory dynamic systems historical access information prediction mechanism`
- error type: ``
- error message: ``
- doc count: `15`

### Chunk 1
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
[24] Y. Huo, T. Chaffey, R. Sepulchre, Kernel modelling of fading memory systems (2024). doi:10.48550/ARXIV.2403.11945. [25] R. Sepulchre, Fading memory [from the editor], IEEE Control Systems 41 (1) (2021) 4–5. doi:10.1109/mcs.2020.3033098.
```

### Chunk 2
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
Architecture of our HA-GNN model Figure 4. Architecture of our HA-GNN model ## Page 5 2.5 Training and Evaluation We use a random 20% split of our data to pretrain our GCN component. The pretraining is a 3-class classification task where the model predicts the diagnosis of a subject (CN, MCI, or AD) in a given visit. Unlike our main task, during pretraining, the input consists of a single visit instead of the entire visit history of a subject. Fig. 5 shows the architecture of the pretraining model. Figure 5. Architecture of the GCN pretraining model Once pretraining is done, we use 5-fold cros...(truncated)
```

### Chunk 3
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
N. Tsitsiklis, Worst-case identification of nonlinear fading memory systems, Automatica 31 (3) (1995) 503–508. doi:10.1016/0005-1098(94)00131-2.
```

### Chunk 4
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
W. Sandberg, Z₊ fading memory and extensions of input–output maps, International Journal of Circuit Theory and Applications 29 (4) (2001) 381–388. doi:10.1002/cta.156.
```

### Chunk 5
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
Model Acc. AUC-ROC BA CN to MCI MCI to AD HA-GNN (LSTM) (0.829 \pm 0.058) (0.852 \pm 0.065) (0.771 \pm 0.114) (0.688) (0.676) HA-GNN (RNN) (0.743 \pm 0.038) (0.771 \pm 0.067) (0.651 \pm 0.114) (0.5) (0.514) HA-GNN (GRU) (0.733 \pm 0.049) (0.786 \pm 0.062) (0.704 \pm 0.071) (0.625) (0.676) w/o pretraining (0.723 \pm 0.079) (0.642 \pm 0.095) (0.594 \pm 0.061) (0.375) (0.405) Due to our novel methodology, comparisons with the existing literature are difficult. However there are a few papers that are close to our task and methods.
```

### dense_zh
- ok: `False`
- query: `动态系统 衰减记忆 作用 HA-GNN 历史访问 预测 相似性 本质区别`
- error type: `TypeError`
- error message: `'VLLMOpenAIEmbeddings' object is not callable`
- doc count: `0`

### dense_en
- ok: `False`
- query: `role of fading memory in dynamic systems versus HA-GNN historical access prediction mechanism`
- error type: `TypeError`
- error message: `'VLLMOpenAIEmbeddings' object is not callable`
- doc count: `0`

## Fallback Evaluation Input
- input source: `branch_fallback`
- branch order: `bm25_en, dense_zh, dense_en`
- doc count: `5`

### Chunk 1
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
[24] Y. Huo, T. Chaffey, R. Sepulchre, Kernel modelling of fading memory systems (2024). doi:10.48550/ARXIV.2403.11945. [25] R. Sepulchre, Fading memory [from the editor], IEEE Control Systems 41 (1) (2021) 4–5. doi:10.1109/mcs.2020.3033098.
```

### Chunk 2
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `null`
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
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
Architecture of our HA-GNN model Figure 4. Architecture of our HA-GNN model ## Page 5 2.5 Training and Evaluation We use a random 20% split of our data to pretrain our GCN component. The pretraining is a 3-class classification task where the model predicts the diagnosis of a subject (CN, MCI, or AD) in a given visit. Unlike our main task, during pretraining, the input consists of a single visit instead of the entire visit history of a subject. Fig. 5 shows the architecture of the pretraining model. Figure 5. Architecture of the GCN pretraining model Once pretraining is done, we use 5-fold cros...(truncated)
```

### Chunk 3
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
N. Tsitsiklis, Worst-case identification of nonlinear fading memory systems, Automatica 31 (3) (1995) 503–508. doi:10.1016/0005-1098(94)00131-2.
```

### Chunk 4
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
W. Sandberg, Z₊ fading memory and extensions of input–output maps, International Journal of Circuit Theory and Applications 29 (4) (2001) 381–388. doi:10.1002/cta.156.
```

### Chunk 5
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `null`
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
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
Model Acc. AUC-ROC BA CN to MCI MCI to AD HA-GNN (LSTM) (0.829 \pm 0.058) (0.852 \pm 0.065) (0.771 \pm 0.114) (0.688) (0.676) HA-GNN (RNN) (0.743 \pm 0.038) (0.771 \pm 0.067) (0.651 \pm 0.114) (0.5) (0.514) HA-GNN (GRU) (0.733 \pm 0.049) (0.786 \pm 0.062) (0.704 \pm 0.071) (0.625) (0.676) w/o pretraining (0.723 \pm 0.079) (0.642 \pm 0.095) (0.594 \pm 0.061) (0.375) (0.405) Due to our novel methodology, comparisons with the existing literature are difficult. However there are a few papers that are close to our task and methods.
```

## Plan Reference
```json
{
  "original_query": "fading memory（衰减记忆）’这一性质在动态系统中的作用，与HA-GNN模型中利用历史访问信息进行预测的机制之间，有何相似性与本质区别？",
  "normalized_query_zh": "衰减记忆在动态系统中的作用与 HA-GNN 模型利用历史访问预测机制的相似性及本质区别是什么？",
  "retrieval_query_zh": "动态系统 衰减记忆 作用 HA-GNN 历史访问 预测 相似性 本质区别",
  "retrieval_query_en": "role of fading memory in dynamic systems versus HA-GNN historical access prediction mechanism",
  "crawler_query_en": "fading memory dynamic systems HA-GNN historical access prediction",
  "keywords_zh": [
    "HA-GNN",
    "衰减记忆",
    "动态系统",
    "历史访问信息",
    "预测机制"
  ],
  "keywords_en": [
    "HA-GNN",
    "fading memory",
    "dynamic systems",
    "historical access information",
    "prediction mechanism"
  ],
  "required_aspects": [
    "role of fading memory in dynamic systems",
    "mechanism of historical access information in HA-GNN",
    "similarities between fading memory and HA-GNN",
    "essential differences between fading memory and HA-GNN prediction mechanism"
  ]
}
```