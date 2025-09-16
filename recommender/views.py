from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view, authentication_classes
from django.views.decorators.csrf import csrf_exempt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent

_model = None
_talent_df = None
_talent_embeddings = None
_tfidf = None
_tfidf_matrix = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Force CPU to avoid CUDA/meta-tensor issues on machines without GPU setup
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    return _model


def _load_dataset() -> pd.DataFrame:
    global _talent_df
    if _talent_df is not None:
        return _talent_df
    # Attempt to parse a saved HTML export from Google Sheets
    candidates: List[pd.DataFrame] = []
    html_path = BASE_DIR / 'Talent Profiles - Google Sheets.html'
    try:
        if html_path.exists():
            for tbl in pd.read_html(str(html_path)):
                candidates.append(tbl)
    except Exception:
        pass
    # Fallback: scan the saved HTML assets folder; gather all tables and pick the largest
    html_dir = BASE_DIR / 'Talent Profiles - Google Sheets_files'
    if html_dir.exists():
        for p in sorted(html_dir.glob('*.html')):
            try:
                for tbl in pd.read_html(str(p)):
                    candidates.append(tbl)
            except Exception:
                continue
    # CSV fallback
    if not candidates:
        csv_path = BASE_DIR / 'talent_profiles.csv'
        if csv_path.exists():
            try:
                candidates.append(pd.read_csv(csv_path))
            except Exception:
                pass
    # If we found tables, pick the one that best matches expected schema
    if candidates:
        def score_df(d: pd.DataFrame) -> tuple:
            cols = [str(c).strip().lower() for c in d.columns]
            expected = {
                'name','full name','talent','gender','location','bio','description','job types','skills','software','platforms','niches','verticals','creators','past creators','monthly rate','hourly rate','#views','views'
            }
            match = sum(1 for c in cols if any(e in c for e in expected))
            return (match, len(d), len(d.columns))
        best = max(candidates, key=score_df)
        df = best.copy()
    else:
        df = pd.DataFrame()
    # Normalize and drop empty/header-like rows
    df = df.fillna('')
    df = df[[c for c in df.columns if str(c).strip().lower() != 'unnamed: 0']]
    col_join = ' | '.join([str(c).strip() for c in df.columns]) if len(df.columns) else ''
    def _row_has_text(r: pd.Series) -> bool:
        try:
            values_join = ' | '.join([str(v).strip() for v in r.values])
            if values_join.strip() == col_join.strip():
                return False
            return any(str(v).strip() for v in r.values)
        except Exception:
            return False
    if len(df) > 0:
        df = df[df.apply(_row_has_text, axis=1)]
        df = df.reset_index(drop=True)
    _talent_df = df
    return _talent_df


def _build_profile_text(row: pd.Series) -> str:
    """Build comprehensive profile text for better semantic matching."""
    # Prioritize high-value fields for matching
    high_priority = [
        str(row.get('Skills', row.get('Relevant skills', ''))),
        str(row.get('Job types', '')),
        str(row.get('Bio', row.get('Description', ''))),
        str(row.get('Software', '')),
        str(row.get('Platforms', '')),
    ]
    
    # Add context fields
    context = [
        str(row.get('Niches', row.get('Verticals', ''))),
        str(row.get('Past Creators', row.get('Creators', ''))),
        str(row.get('Location', '')),
        str(row.get('Gender', '')),
    ]
    
    # Combine with weighted importance
    text_parts = []
    for part in high_priority:
        if part and part.strip():
            text_parts.append(part.strip())
    
    for part in context:
        if part and part.strip():
            text_parts.append(part.strip())
    
    # Fallback: include all non-empty fields
    if not text_parts:
        try:
            text_parts = [str(v).strip() for v in row.values if str(v).strip() and str(v).strip() != 'nan']
        except Exception:
            text_parts = []
    
    return ' | '.join(text_parts)


def _extract_name(row: pd.Series) -> str:
    candidate_keys = [
        'Name', 'Talent', 'Full Name', 'Full name', 'Profile Name', 'Profile',
        'Talent name', 'Talent Name', 'First Name', 'Last Name', 'Creator',
        'creator_name', 'talent_name'
    ]
    for key in candidate_keys:
        if key in row:
            val = str(row.get(key, '')).strip()
            if val:
                return val
    # Try combining first/last name
    first = str(row.get('First Name', '')).strip()
    last = str(row.get('Last Name', '')).strip()
    if first or last:
        return f"{first} {last}".strip()
    # As last resort, use first non-empty column value that isn't a header list
    row_text = ' '.join([str(v).strip() for v in row.values])
    cols_text = ' '.join([str(c).strip() for c in row.index])
    if row_text.strip() == cols_text.strip():
        return 'Talent'
    for v in row.values:
        s = str(v).strip()
        if s and s.lower() != 'anyone who has the link can access. no sign-in required.':
            return s[:80]
    return 'Talent'


def _extract_location(row: pd.Series) -> str:
    keys = [
        'Location', 'City', 'Country', 'Base', 'Based in', 'location', 'Current location', 'City/Country'
    ]
    for key in keys:
        if key in row:
            val = str(row.get(key, '')).strip()
            if val:
                return val
    return ''


def _extract_gender(row: pd.Series) -> str:
    for key in ['Gender', 'gender', 'Sex']:
        if key in row:
            val = str(row.get(key, '')).strip()
            if val:
                return val
    return ''


def _extract_monthly_rate(row: pd.Series) -> str:
    keys = ['Monthly Rate', 'Monthly', 'Full-time rate', 'Monthly (Full-time) rates', 'monthly_rate']
    for key in keys:
        if key in row:
            val = str(row.get(key, '')).strip()
            if val:
                return val
    return ''


def _extract_hourly_rate(row: pd.Series) -> str:
    keys = ['Hourly Rate', 'Hourly', 'Rate (hourly)', 'Per hour', 'hourly_rate']
    for key in keys:
        if key in row:
            val = str(row.get(key, '')).strip()
            if val:
                return val
    return ''


def _embed_talent(df: pd.DataFrame):
    global _talent_embeddings
    if _talent_embeddings is not None:
        return _talent_embeddings
    model = _get_model()
    corpus = [_build_profile_text(row) for _, row in df.iterrows()]
    _talent_embeddings = model.encode(corpus, show_progress_bar=False, normalize_embeddings=True)
    return _talent_embeddings


def _fit_tfidf(df: pd.DataFrame):
    global _tfidf, _tfidf_matrix
    if _tfidf is not None and _tfidf_matrix is not None:
        return _tfidf, _tfidf_matrix
    corpus = [_build_profile_text(row) for _, row in df.iterrows()]
    try:
        _tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        _tfidf_matrix = _tfidf.fit_transform(corpus)
        return _tfidf, _tfidf_matrix
    except ValueError:
        # empty vocabulary (e.g., all-empty rows) → disable TF-IDF
        _tfidf = None
        _tfidf_matrix = None
        return None, None


def _score_candidates(job: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Enhanced scoring with multiple matching strategies for better accuracy."""
    model = _get_model()
    embeddings = _embed_talent(df)
    _tfidf, _tfidf_matrix = _fit_tfidf(df)

    # Enhanced job text preprocessing
    job_text_parts = [
        job.get('text', ''),
    ]
    job_text = ' | '.join([p for p in job_text_parts if p])
    
    # Get semantic similarity using sentence transformer
    job_vec = model.encode([job_text], normalize_embeddings=True)
    sims_embed = cosine_similarity(job_vec, embeddings)[0]

    # TF-IDF similarity for keyword matching
    if _tfidf is not None and _tfidf_matrix is not None:
        job_tfidf = _tfidf.transform([job_text])
        sims_tfidf = cosine_similarity(job_tfidf, _tfidf_matrix)[0]
    else:
        import numpy as np
        sims_tfidf = np.zeros(len(df))

    # Extract preferences for boosting
    loc_pref = job.get('location_pref', '').lower()
    female_pref = job.get('prefer_female', False)

    results: List[Dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Enhanced scoring: blend semantic and keyword matching
        weight_tfidf = 0.3 if _tfidf is not None else 0.0
        weight_embed = 0.7  # Higher weight for semantic similarity
        base_score = weight_embed * float(sims_embed[idx]) + weight_tfidf * float(sims_tfidf[idx])

        # Location preference boost
        location_boost = 0.0
        if loc_pref:
            talent_loc = str(row.get('Location', '')).lower()
            if loc_pref in talent_loc:
                location_boost = 0.05  # Strong location match
            elif any(word in talent_loc for word in loc_pref.split()):
                location_boost = 0.02  # Partial location match

        # Gender preference boost
        gender_boost = 0.0
        if female_pref:
            gender = str(row.get('Gender', '')).lower()
            if 'female' in gender or gender == 'f':
                gender_boost = 0.03

        # Experience boost based on views/engagement
        experience_boost = 0.0
        try:
            views = int(row.get('#Views', row.get('Views', 0)) or 0)
            experience_boost = min(views / 2000.0, 0.04)  # Cap at 4% boost
        except Exception:
            pass

        # Skills relevance boost (keyword matching in skills)
        skills_boost = 0.0
        skills_text = str(row.get('Skills', row.get('Relevant skills', ''))).lower()
        job_lower = job_text.lower()
        if skills_text and any(word in skills_text for word in job_lower.split() if len(word) > 3):
            skills_boost = 0.02

        # Final score with all boosts
        final_score = base_score + location_boost + gender_boost + experience_boost + skills_boost
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, final_score))

        results.append({
            'index': idx,
            'name': _extract_name(row),
            'location': _extract_location(row),
            'gender': _extract_gender(row),
            'monthly_rate': _extract_monthly_rate(row),
            'hourly_rate': _extract_hourly_rate(row),
            'summary': _build_profile_text(row)[:500],
            'score': round(final_score, 4),
            'base_similarity': round(float(sims_embed[idx]), 4),
            'keyword_match': round(float(sims_tfidf[idx]), 4) if _tfidf is not None else 0.0,
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


@api_view(['GET'])
def health(_request):
    return JsonResponse({'status': 'ok'})


@api_view(['GET'])
def dataset_info(_request):
    df = _load_dataset()
    return JsonResponse({'rows': int(len(df)), 'columns': list(df.columns.astype(str))})


@api_view(['POST'])
@authentication_classes([])
@csrf_exempt
def recommend(request):
    df = _load_dataset()
    if df.empty:
        return JsonResponse({'error': 'Dataset not found or empty.'}, status=400)

    payload = request.data or {}
    text = str(payload.get('text', '')).strip()
    if not text:
        return JsonResponse({'error': 'Please provide job description text in "text" field.'}, status=400)

    job = {
        'text': text,
        'location_pref': str(payload.get('location_pref', '')).lower(),
        'prefer_female': bool(payload.get('prefer_female', False)),
    }
    try:
        top_n = min(10, len(df)) if len(df) else 0
        results = _score_candidates(job, df)[:top_n]
        return JsonResponse({'top10': results})
    except Exception as e:
        return JsonResponse({'error': f'Failed to score: {e.__class__.__name__}: {e}'}, status=500)


def home(request):
    return render(request, 'index.html')

@api_view(['GET'])
def talent_detail(_request, idx: int):
    df = _load_dataset()
    try:
        if idx < 0 or idx >= len(df):
            return JsonResponse({'error': 'Index out of range'}, status=404)
        row = df.iloc[int(idx)]
    except Exception:
        return JsonResponse({'error': 'Talent not found'}, status=404)
    data = {str(k): ('' if pd.isna(v) else str(v)) for k, v in row.to_dict().items()}
    data.update({
        'name': _extract_name(row),
        'location': _extract_location(row),
        'gender': _extract_gender(row),
        'monthly_rate': _extract_monthly_rate(row),
        'hourly_rate': _extract_hourly_rate(row),
        'summary': _build_profile_text(row),
    })
    return JsonResponse({'index': int(idx), 'talent': data})
