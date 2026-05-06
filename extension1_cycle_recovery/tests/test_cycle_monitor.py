import unittest
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "server"
    / "tasks"
    / "os_interaction"
    / "cycle_monitor.py"
)
SPEC = importlib.util.spec_from_file_location("cycle_monitor", MODULE_PATH)
cycle_monitor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cycle_monitor
SPEC.loader.exec_module(cycle_monitor)
CycleMonitor = cycle_monitor.CycleMonitor


class CycleMonitorTest(unittest.TestCase):
    def make_enabled_monitor(self, **overrides):
        config = {
            "enable_cycle_recovery": True,
            "cycle_window_size": 5,
            "cycle_similarity_threshold": 0.8,
            "cycle_score_threshold": 0.8,
            "cycle_min_repeated_turns": 2,
            "cycle_cooldown_turns": 2,
            "cycle_max_interventions": 2,
        }
        config.update(overrides)
        return CycleMonitor(config)

    def record(self, monitor, state, turn_id, command, observation):
        return monitor.record_turn(
            state,
            monitor.make_turn_record(
                turn_id=turn_id,
                raw_agent_response="",
                action_type="bash",
                command=command,
                observation=observation,
                observation_truncated=False,
            ),
        )

    def test_repeated_command_and_observation_triggers_recovery(self):
        monitor = self.make_enabled_monitor()
        state = monitor.create_state("sample-1")

        self.record(monitor, state, 0, "ls /tmp", "a.txt\nb.txt")
        self.record(monitor, state, 1, "ls /tmp", "a.txt\nb.txt")
        decision = self.record(monitor, state, 2, "ls /tmp", "a.txt\nb.txt")

        self.assertTrue(decision.triggered)
        self.assertIn("[RECOVERY NOTICE]", state.pending_notice)
        self.assertEqual(len(state.interventions), 1)

    def test_different_command_and_new_observation_does_not_trigger(self):
        monitor = self.make_enabled_monitor()
        state = monitor.create_state("sample-2")

        self.record(monitor, state, 0, "ls /tmp", "a.txt\nb.txt")
        self.record(monitor, state, 1, "find /var -maxdepth 1 -type f", "new.log\ncache.db")
        decision = self.record(monitor, state, 2, "cat /etc/hostname", "agentbench-host")

        self.assertFalse(decision.triggered)
        self.assertIsNone(state.pending_notice)

    def test_cooldown_prevents_repeated_interventions(self):
        monitor = self.make_enabled_monitor(cycle_cooldown_turns=3, cycle_max_interventions=5)
        state = monitor.create_state("sample-3")

        self.record(monitor, state, 0, "ls /tmp", "a.txt")
        self.record(monitor, state, 1, "ls /tmp", "a.txt")
        first = self.record(monitor, state, 2, "ls /tmp", "a.txt")
        second = self.record(monitor, state, 3, "ls /tmp", "a.txt")

        self.assertTrue(first.triggered)
        self.assertFalse(second.triggered)
        self.assertEqual(second.reason, "cooldown")
        self.assertEqual(len(state.interventions), 1)

    def test_disabled_monitor_does_not_trigger(self):
        monitor = CycleMonitor({"enable_cycle_recovery": False})
        state = monitor.create_state("sample-4")

        self.record(monitor, state, 0, "ls /tmp", "a.txt")
        self.record(monitor, state, 1, "ls /tmp", "a.txt")
        decision = self.record(monitor, state, 2, "ls /tmp", "a.txt")

        self.assertFalse(decision.triggered)
        self.assertEqual(decision.reason, "disabled")
        self.assertEqual(len(state.interventions), 0)


if __name__ == "__main__":
    unittest.main()