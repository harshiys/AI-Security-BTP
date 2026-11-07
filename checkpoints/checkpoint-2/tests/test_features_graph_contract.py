import json, tempfile, unittest, os
import numpy as np, pandas as pd
from tests.common import hand_events
import mla_pipeline as P, graph_features as G, cp2_contract as C
from metrics_extra import classification_metrics


class FeaturesAndGraph(unittest.TestCase):
    def setUp(self):
        self.ev = hand_events()
        self.ros = pd.DataFrame({"run_id": "r1", "agent": ["a", "b"]})

    def test_tabular_features_ignore_events_at_or_after_K(self):
        base = P.build_features(self.ev, self.ros, 5)
        wrecked = self.ev.copy()
        late = wrecked.event_idx >= 5
        wrecked.loc[late, ["msg_len", "n_calls"]] = 9999
        wrecked.loc[late, "acting_agent"] = "zz"
        pd.testing.assert_frame_equal(P.build_features(wrecked, self.ros, 5), base)

    def test_graph_features_hand_made_log(self):
        f = G.graph_features(self.ev, self.ros, 10).set_index("agent")
        self.assertGreaterEqual(int(f.loc["a", "g_pingpong"]), 1)           # a,b,a,b alternation present
        self.assertEqual(int(f.loc["a", "g_n_reciprocal"]), 1)              # a<->b
        self.assertEqual(int(f.loc["a", "g_agents_touched"]), 2)
        self.assertTrue(G.selfcheck(self.ev, self.ros, K=5, n_runs=1))      # no future leak

    def test_novel_edge_uses_other_tasks_only(self):
        ev2 = self.ev.assign(run_id="r2", task_id="t2")
        both = pd.concat([self.ev, ev2])
        idx = G.benign_edge_index(both, {"r1", "r2"})                       # both runs harmless
        ros = pd.DataFrame({"run_id": ["r1"], "agent": ["a"]})
        seen_elsewhere = G.graph_features(both, ros, 10, idx, {"t1", "t2"})
        self.assertEqual(int(seen_elsewhere.g_n_novel_edges.iloc[0]), 0)    # r2's task also had these edges
        alone = G.graph_features(both, ros, 10, idx, {"t1"})                # t2 not allowed (e.g. a test task)
        self.assertGreater(int(alone.g_n_novel_edges.iloc[0]), 0)


class SharedContract(unittest.TestCase):
    def test_valid_record(self):
        r = C.make_record("item1", "flag", 0.7)
        self.assertEqual(set(r), {"checkpoint_id", "item_id", "verdict", "score", "timestamp"})
        C.validate_record(r)

    def test_schema_rules_are_enforced(self):
        good = C.make_record("i", "pass", 0.1)
        for bad in (dict(good, verdict="allow"), dict(good, score=1.5), dict(good, extra="x"),
                    dict(good, timestamp="2026-01-01T00:00:00+05:30"), {k: v for k, v in good.items() if k != "item_id"}):
            with self.assertRaises(Exception):
                C.validate_record(bad)

    def test_scorer_output_validates_and_is_prefix_only(self):
        import scorer
        from sklearn.linear_model import LogisticRegression
        cols = P.FEATURE_COLS + G.GRAPH_COLS
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, len(cols))); y = (X[:, 0] > 0).astype(int)
        bundle = dict(model=LogisticRegression(max_iter=500).fit(X, y), cols=cols, K=5, threshold=0.5, feature_set="tab+graph",
                      rule_limits={"g_pingpong": 100.0}, edge_index={}, train_tasks=[],
                      benign_mean={c: 0.0 for c in cols}, benign_std={c: 1.0 for c in cols})
        ev = hand_events()
        rec, side = scorer.score_run_prefix(ev, "r1", bundle, roster=["a", "b"])
        C.validate_record(rec)
        self.assertIn(rec["verdict"], {"pass", "flag", "block"})
        rec2, _ = scorer.score_run_prefix(ev[ev.event_idx < 5], "r1", bundle, roster=["a", "b"], now=rec["timestamp"])
        self.assertEqual(rec["score"], rec2["score"])                       # later events never change the score
        bundle["rule_limits"] = {"g_pingpong": -1.0}                        # force a hard rule
        self.assertEqual(scorer.score_run_prefix(ev, "r1", bundle, roster=["a", "b"])[0]["verdict"], "block")


class Metrics(unittest.TestCase):
    def test_values_and_nan_when_one_class(self):
        m = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
        self.assertEqual((m["accuracy"], m["precision"], m["recall"]), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(m["roc_auc"], 0.75)
        one = classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
        self.assertTrue(np.isnan(one["roc_auc"]) and np.isnan(one["recall"]))


if __name__ == "__main__":
    unittest.main()
