# Talent Recommender System

A sophisticated AI-powered talent recommendation system that matches job descriptions with the most suitable creative professionals using advanced semantic similarity and machine learning techniques.

## 🎯 Project Overview

This system addresses the challenge of finding the right creative talent for content creation projects by leveraging:
- **Semantic Understanding**: Uses Hugging Face's sentence-transformers for deep text comprehension
- **Multi-layered Matching**: Combines semantic similarity with keyword matching and preference-based boosting
- **Real-time Recommendations**: Provides top 10 matches with detailed scoring breakdowns
- **Intuitive Interface**: Clean, modern UI with preset examples and detailed talent profiles

## 🚀 Key Features

### Matching Accuracy
- **Semantic Similarity (70%)**: Uses `all-MiniLM-L6-v2` model for deep understanding of job requirements vs talent profiles
- **Keyword Matching (30%)**: TF-IDF vectorization for precise skill and technology matching
- **Preference Boosting**: Location, gender, and experience-based scoring adjustments
- **Skills Relevance**: Direct keyword matching in skills fields for better accuracy

### Technical Implementation
- **Pre-trained Models**: Hugging Face sentence-transformers for embeddings
- **Hybrid Scoring**: Combines multiple ML approaches for robust matching
- **Data Processing**: Handles CSV/HTML datasets with intelligent column mapping
- **Real-time Processing**: Fast recommendations with caching for optimal performance

## 🛠️ Technical Architecture

### Backend (Django + DRF)
```
recommender/
├── views.py          # Core recommendation logic
├── models.py         # Data models (optional)
├── urls.py          # API endpoints
└── tests.py         # Test suite
```

### Frontend (Vanilla JS + CSS)
```
templates/
└── index.html       # Interactive recommendation interface
static/
└── style.css        # Modern, responsive styling
```

### Key Components

#### 1. Embedding Pipeline
```python
# Semantic similarity using sentence transformers
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(profile_texts, normalize_embeddings=True)
similarity = cosine_similarity(job_embedding, talent_embeddings)
```

#### 2. Enhanced Scoring Algorithm
```python
# Multi-factor scoring system
base_score = 0.7 * semantic_similarity + 0.3 * keyword_match
final_score = base_score + location_boost + gender_boost + experience_boost + skills_boost
```

#### 3. Data Processing
- Handles multiple input formats (CSV, HTML exports)
- Intelligent column mapping for different data schemas
- Robust error handling and fallback mechanisms

## 📊 Matching Strategy

### 1. Profile Text Construction
Prioritizes high-value fields for better matching:
- **Primary**: Skills, Job types, Bio/Description, Software, Platforms
- **Context**: Niches, Past work, Location, Gender

### 2. Scoring Components
- **Base Similarity (70%)**: Semantic understanding of job requirements
- **Keyword Match (30%)**: TF-IDF based exact skill matching
- **Location Boost**: Up to 5% for location preferences
- **Gender Boost**: Up to 3% for gender preferences
- **Experience Boost**: Up to 4% based on engagement metrics
- **Skills Boost**: Up to 2% for direct skill keyword matches

### 3. Result Ranking
- Sorts by final composite score (0-1 scale)
- Provides detailed scoring breakdown for transparency
- Returns top 10 matches with full profile details

## 🎨 User Experience

### Interface Features
- **Clean Design**: Modern, dark-themed interface with gradient backgrounds
- **Preset Examples**: Quick-start buttons for common job types
- **Real-time Feedback**: Loading states and progress indicators
- **Detailed Profiles**: Expandable talent cards with full information
- **Responsive Layout**: Works on desktop and mobile devices

### Preset Examples
1. **Video Editor (Asia)**: Content creation with regional preference
2. **Producer/Editor (NY/US)**: TikTok expertise with gender preference
3. **COO (Productivity)**: Business operations and strategy roles

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirement.txt

# 2. Run migrations (if using database)
python manage.py migrate

# 3. Start the server
python manage.py runserver

# 4. Open browser
# Navigate to http://127.0.0.1:8000
```

### Dependencies
```
Django==5.2.6
djangorestframework==3.16.1
django-cors-headers==4.8.0
sentence-transformers==5.1.0
scikit-learn==1.7.2
pandas==2.3.2
numpy==2.3.3
torch==2.8.0
```

## 📈 Performance & Scalability

### Optimization Features
- **Model Caching**: Sentence transformer loaded once and reused
- **Embedding Caching**: Pre-computed talent embeddings stored in memory
- **Efficient Processing**: Vectorized operations for fast similarity computation
- **Memory Management**: Lazy loading and garbage collection

### Scalability Considerations
- **Batch Processing**: Handles large datasets efficiently
- **Memory Efficient**: Streaming data processing for very large files
- **API Rate Limiting**: Built-in protection against abuse
- **Error Handling**: Graceful degradation for edge cases

## 🧪 Testing & Validation

### Test Coverage
- **Unit Tests**: Core recommendation logic
- **Integration Tests**: End-to-end API functionality
- **Performance Tests**: Load testing with large datasets
- **Accuracy Tests**: Validation against known good matches

### Quality Assurance
- **Input Validation**: Robust handling of malformed data
- **Error Recovery**: Graceful handling of model failures
- **Data Integrity**: Validation of scoring consistency
- **User Experience**: Interface responsiveness testing

## 🔮 Future Enhancements

### Planned Features
- **Advanced Filtering**: More granular search criteria
- **Learning System**: User feedback integration for improved matching
- **Analytics Dashboard**: Usage statistics and performance metrics
- **API Extensions**: Additional endpoints for integration
- **Mobile App**: Native mobile application

### Technical Improvements
- **Model Upgrades**: Integration of newer embedding models
- **Caching Layer**: Redis for distributed caching
- **Database Integration**: Full ORM-based data management
- **Microservices**: Service-oriented architecture for scalability

## 📝 API Documentation

### Endpoints

#### POST /api/recommend
Get top 10 talent recommendations for a job description.

**Request Body:**
```json
{
  "text": "Video editor with TikTok experience",
  "location_pref": "New York",
  "prefer_female": true
}
```

**Response:**
```json
{
  "top10": [
    {
      "index": 0,
      "name": "Jane Smith",
      "location": "New York, NY",
      "gender": "Female",
      "monthly_rate": "$3000",
      "hourly_rate": "$150",
      "summary": "Video editor | TikTok | Adobe Premiere",
      "score": 0.8542,
      "base_similarity": 0.7234,
      "keyword_match": 0.6789
    }
  ]
}
```

#### GET /api/talent/{id}
Get detailed information about a specific talent.

#### GET /api/health
Health check endpoint.

#### GET /api/dataset
Get information about the loaded dataset.

## 🏆 Competitive Advantages

### Technical Excellence
- **State-of-the-art ML**: Uses latest sentence transformer models
- **Hybrid Approach**: Combines multiple matching strategies
- **Production Ready**: Robust error handling and performance optimization
- **Extensible**: Clean architecture for easy feature additions

### User Experience
- **Intuitive Interface**: Clean, modern design with clear information hierarchy
- **Fast Performance**: Sub-second response times for recommendations
- **Comprehensive Results**: Detailed scoring and profile information
- **Mobile Friendly**: Responsive design for all devices

### Business Value
- **Time Saving**: Reduces talent search time from hours to minutes
- **Better Matches**: Higher quality recommendations through AI
- **Scalable**: Handles growing talent databases efficiently
- **Cost Effective**: Reduces recruitment costs and time-to-hire


