from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch
import pandas as pd
import numpy as np


class RecommenderApiTests(TestCase):
    def setUp(self):
        self.recommend_url = reverse('recommend')
        self.health_url = reverse('health')

    def test_health(self):
        resp = self.client.get(self.health_url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get('status'), 'ok')

    def test_recommend_requires_text(self):
        resp = self.client.post(self.recommend_url, data={}, content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    @patch('recommender.views._load_dataset')
    @patch('recommender.views.SentenceTransformer')
    def test_recommend_topn_with_mocked_model(self, mock_st, mock_load):
        # Mock dataset with 3 talents
        df = pd.DataFrame([
            {'Name': 'Alice', 'Location': 'NY, US', 'Gender': 'Female', 'Skills': 'editing; tiktok', 'Monthly Rate': '3000'},
            {'Name': 'Bob', 'Location': 'LA, US', 'Gender': 'Male', 'Skills': 'storyboarding; filming', 'Monthly Rate': '2800'},
            {'Name': 'Carol', 'Location': 'Remote', 'Gender': 'Female', 'Skills': 'sound design; sequencing', 'Monthly Rate': '3200'},
        ])
        mock_load.return_value = df

        # Mock sentence transformer to return fixed vectors
        class DummyModel:
            def encode(self, texts, **kwargs):
                # Return simple deterministic vectors based on length
                if isinstance(texts, list):
                    return np.array([[len(t) % 5 + i % 3 for i in range(6)] for t in texts], dtype=float)
                return np.array([1, 0, 0, 1, 0, 1], dtype=float)

        mock_st.return_value = DummyModel()

        payload = {
            'text': 'Producer/Video Editor in NY with TikTok and storyboarding',
            'location_pref': 'new york',
            'prefer_female': True,
        }
        resp = self.client.post(self.recommend_url, data=payload, content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('top10', data)
        # Should return up to 3 results (dataset size)
        self.assertGreaterEqual(len(data['top10']), 1)
        self.assertLessEqual(len(data['top10']), 3)
        # Verify result schema
        first = data['top10'][0]
        for key in ['name', 'location', 'gender', 'score']:
            self.assertIn(key, first)
