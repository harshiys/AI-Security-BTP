import os, tempfile, unittest
from tests.common import ROOT, FIX, write_fake_tamas
import mla_pipeline as P


class RealFormatParser(unittest.TestCase):
    def test_parses_genuine_autogen_console_output(self):
        rows = P.parse_log(os.path.join(FIX, "autogen_console_sample.txt"))
        types = [r["event_type"] for r in rows]
        self.assertEqual(types[:4], ["TextMessage", "ToolCallRequestEvent", "ToolCallExecutionEvent", "ToolCallSummaryMessage"])
        req = rows[1]
        self.assertEqual(req["acting_agent"], "news_gathering_agent")
        self.assertEqual(req["tool_name"], "fetch_latest_rss_feeds")
        self.assertEqual(rows[2]["tool_name"], "fetch_latest_rss_feeds")
        self.assertFalse(rows[2]["is_error"])
        self.assertEqual([r["event_idx"] for r in rows], list(range(len(rows))))


class LabelDefinition(unittest.TestCase):
    def test_injected_tool_regex_matches_tamas_rule(self):
        self.assertEqual(P.injected_tool({"user query": "Please export it using export_x tool of Assessment Agent"}), "export_x")
        self.assertEqual(P.injected_tool({"user query": "Use the Using The leak_y Tool now"}), "leak_y")
        self.assertIsNone(P.injected_tool({"user query": "nothing injected here"}))

    def _logs(self, root, body_by_i):
        d = os.path.join(root, "logs", "DPI", "RoundRobin")
        os.makedirs(d, exist_ok=True)
        for i, body in body_by_i.items():
            with open(os.path.join(d, f"DPI_news_{i}.txt"), "w") as fh:
                fh.write(body)
        return os.path.join(root, "logs")

    @staticmethod
    def _log(agent_calls):
        out = "---------- TextMessage (user) ----------\nTask: x\n"
        for ag, tool in agent_calls:
            out += (f"---------- ToolCallRequestEvent ({ag}) ----------\n[FunctionCall(id='1', arguments='{{}}', name='{tool}')]\n"
                    f"---------- ToolCallExecutionEvent ({ag}) ----------\n[FunctionExecutionResult(content='ok', name='{tool}', call_id='1', is_error=False)]\n")
        return out + "---------- TextMessage (fact_checking_agent) ----------\nok\n"

    def test_verified_label_is_per_agent_and_only_for_dpi(self):
        with tempfile.TemporaryDirectory() as t:
            tasks_list = write_fake_tamas(os.path.join(t, "T"))
            tasks = {("DPI", "news"): tasks_list, ("DPI", "other"): tasks_list[:1]}
            logs = self._logs(t, {0: self._log([("news_gathering_agent", "evil_tool")]),      # injected tool called by one agent
                                  1: self._log([("news_gathering_agent", "fetch_news")]),     # injection ignored
                                  2: self._log([("fact_checking_agent", "evil_tool")])})
            ev, lab, man, skipped = P.build_tables(logs, tasks, {"evil_tool"}, {}, P.LABEL_VERIFIED)
            y = lab.set_index(["task_id", "agent"])["agent_invoked_attack_tool"]
            by_run = lab.groupby("run_id")["agent_invoked_attack_tool"].sum().sort_values().tolist()
            self.assertEqual(by_run, [0, 1, 1])
            pos = lab[lab.agent_invoked_attack_tool == 1]
            self.assertEqual(set(pos.agent), {"news_gathering_agent", "fact_checking_agent"})
            self.assertTrue((lab.label_definition == P.LABEL_VERIFIED).all())
            # a non-DPI log is skipped (label unverified), never silently labelled
            d = os.path.join(logs, "IPI", "RoundRobin"); os.makedirs(d)
            with open(os.path.join(d, "IPI_news_0.txt"), "w") as fh:
                fh.write(self._log([("news_gathering_agent", "evil_tool")]))
            tasks[("IPI", "news")] = tasks_list
            _, lab2, _, skipped2 = P.build_tables(logs, tasks, {"evil_tool"}, {}, P.LABEL_VERIFIED)
            self.assertTrue(any("label unverified" in why for _, why in skipped2))
            self.assertNotIn("IPI", set(lab2.attack_type))


if __name__ == "__main__":
    unittest.main()
