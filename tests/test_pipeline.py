"""Basic pipeline integration tests."""
import os, pytest
os.environ["DEMO_MODE"]    = "true"
os.environ["LLM_PROVIDER"] = "ollama"

def test_icp_loads():
    from tools.icp_loader import load_icp
    icp = load_icp("configs/sample_icp.yaml")
    assert "name" in icp
    assert "firmographics" in icp

def test_feature_extraction():
    from agents.layer2_scoring import extract_features
    import numpy as np
    icp = {"firmographics": {"employee_range": [50,500], "industry": ["SaaS"], "geography": ["US"]}}
    prospect = {"employee_estimate": 150, "industry": "SaaS", "location": "San Francisco, CA",
                "signals": ["Series B funding"], "signal_strength": 0.8}
    features = extract_features(prospect, icp)
    assert features.shape == (8,)
    assert 0.0 <= features[0] <= 1.0

def test_gbc_scorer():
    from agents.layer2_scoring import GradientBoostingScorer
    import numpy as np
    scorer = GradientBoostingScorer()
    features = np.array([1.0, 1.0, 1.0, 0.9, 0.6, 1.0, 1.0, 0.0])
    score = scorer.score(features)
    assert 0.0 <= score <= 1.0
