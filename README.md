# Hybrid Recommendation System with Contextual Enrichment

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Implicit](https://img.shields.io/badge/implicit-0.7.2-red.svg)](https://github.com/benfred/implicit)

Advanced hybrid recommendation system combining collaborative filtering and content-based approaches with temporal awareness and contextual personalization.

## Key Features

- **Hybrid Architecture**: Combines matrix factorization (ALS) and semantic content-based filtering
- **Temporal Weighting**: Exponential decay of tag relevance (λ=0.002)
- **BM25 Transformation**: Non-linear weighting of implicit feedback
- **Dynamic Blending**: Context-aware balance between CF and CB components
- **Online Learning**: Partial updates for real-time adaptation
- **Stemmed N-gram Features**: Enhanced text processing with Snowball stemmer

## Mathematical Foundations

### 1. Collaborative Filtering (ALS with BM25)
```math
\text{BM25}(w) = \frac{(K1 + 1)w}{K1(1 - B + B\frac{L}{L_{avg}}) + w} \quad (K1=100, B=0.8)
```
```math
\min_{X,Y} \sum_{u,i} (c_{ui} - x_u^Ty_i)^2 + λ(||X||^2_F + ||Y||^2_F) \quad (λ=0.08, factors=128)
```

### 2. Content-Based Filtering
```math
\text{TF-IDF}(t,d) = \text{tf}(t,d) × \log\frac{N}{\text{df}(t)}
```
```math
\text{sim}(a,b) = \cos(θ_{a,b}) = \frac{\mathbf{v}_a · \mathbf{v}_b}{||\mathbf{v}_a||·||\mathbf{v}_b||}
```

### 3. Hybrid Scoring
```math
\alpha = \text{clip}(0.8 - 0.6\frac{\log(1+\text{tag\_count})}{\log(100)}, 0.2, 0.8)
```
```math
\text{Score} = \alpha·\text{CF} + (1-\alpha)(0.7·\text{CB} + 0.3·\text{FavSim})
```


## Collaborative + Content-Based Filtering with Real-Time Adaptation
```mermaid
graph TB
    subgraph "Data Sources"
        A[User-Artist<br>Interactions] --> B[CSR Matrix]
        C[User-Tag<br>Timestamps] --> D[Temporal Decay<br>Calculation]
        E[Artist Metadata] --> F[TF-IDF Corpus]
    end

    subgraph "Collaborative Filtering"
        B --> G[BM25 Weighting]
        G --> H[ALS Model]
        H --> I((128 Latent<br>Factors))
    end

    subgraph "Content-Based Filtering"
        D --> J[Weighted Tags]
        F --> K[Stemmed TF-IDF]
        J --> K
        K --> L[Artist Similarity<br>Matrix]
    end

    subgraph "Hybrid Blending"
        M[User Context] --> N{Dynamic Alpha<br>Calculation}
        H --> O[CF Scores]
        L --> P[CB Scores]
        O --> Q[Score Blending]
        P --> Q
        N --> Q
        Q --> R[Top-N Recommendations]
    end

    subgraph "Online Learning"
        S[New Interactions] --> T[CSR Matrix Update]
        T --> U[Partial ALS Retrain]
        U --> H
    end

    style A stroke-width:2px
    style C stroke-width:2px
    style E stroke-width:2px
    style H stroke-width:2px
    style K stroke-width:2px
    style Q stroke-width:2px
    style T stroke-width:2px
```


## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:**
- implicit==0.7.2 (requires C compiler)
- scipy>=1.10.1
- scikit-learn>=1.2.2
- pandas>=2.0.3
- numpy>=1.24.3
- nltk>=3.8.1

## Data Preparation

Place Last.fm dataset files in `/lastfmdata`:
- `user_artists.dat`
- `artists.dat`
- `tags.dat`
- `user_taggedartists.dat`
- `user_taggedartists-timestamps.dat`

## Usage

```python
# Initialize recommender
from hybrid_recommender import HybridRecommender
recommender = HybridRecommender(data_path="lastfmdata/")

# Get recommendations
recommendations = recommender.recommend(user_id=2, top_n=10)
"""
[('The Beatles', 0.872),
 ('Radiohead', 0.855),
 ('Pink Floyd', 0.841), ...]
"""

# Update model with new interactions
new_data = pd.DataFrame({
    'userID': [2, 2],
    'artistID': [123, 456],
    'weight': [5000, 3000]
})
recommender.partial_fit(new_data)
```

## Implementation Details

### Core Components
1. **Collaborative Filtering**
   - BM25-weighted implicit feedback
   - Alternating Least Squares optimization
   - CSR matrix format for efficient operations

2. **Content-Based Filtering**
   - Temporal-weighted tag aggregation
   - Stemmed bigram TF-IDF vectors
   - Cosine similarity index

3. **Hybrid Engine**
   - Dynamic weighting (α) based on user activity
   - Contextual blending of three signals:
     - ALS predicted scores
     - Personal tag relevance
     - Favorite artist similarity

### Key Optimizations
- Sparse matrix operations (CSR format)
- Cached similarity indices
- Batched partial updates
- Memory-efficient resizing

## References

1. Hu, Y., Koren, Y., & Volinsky, C. (2008). [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121)
2. Robertson, S., & Zaragoza, H. (2009). [The Probabilistic Relevance Framework: BM25 and Beyond](https://dl.acm.org/doi/10.1561/1500000019)
3. Cremonesi, P., et al. (2010). [Performance of Recommender Algorithms on Top-N Recommendation Tasks](https://dl.acm.org/doi/10.1145/1864708.1864721)

## License
MIT License (see LICENSE file)
