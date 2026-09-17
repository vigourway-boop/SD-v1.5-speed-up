import contextlib
import unittest
from unittest import mock

import launch_speedup
import project_health


class ProjectHealthTest(unittest.TestCase):
    def check(self, backend):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(project_health, "DEPENDENCIES", {}))
            stack.enter_context(mock.patch.object(project_health, "inspect_cache", return_value={"missing": []}))
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=False))
            client = stack.enter_context(mock.patch("pynq_cosine_client.PynqCosineClient")).return_value
            client.connect.side_effect = ConnectionError("unreachable")
            result = project_health.check_project(backend)
            return result, client

    def test_missing_board_is_warning_for_auto_error_for_strict(self):
        automatic, client = self.check("auto")
        self.assertFalse(automatic["errors"])
        self.assertFalse(automatic["board"]["available"])
        client.close.assert_called_once()
        strict, _ = self.check("pynq")
        self.assertTrue(strict["errors"])

    def test_pc_mode_never_connects_to_board(self):
        result, client = self.check("pc")
        self.assertFalse(result["board"]["checked"])
        client.connect.assert_not_called()

    def test_launcher_forwards_prompt_as_single_argument_and_child_status(self):
        with mock.patch("sys.argv", ["launch_speedup.py", "--mode", "1", "--decision-backend", "pc",
                                     "--prompt", "a horse & a tree", "--steps", "30"]), \
             mock.patch.object(launch_speedup, "check_project", return_value={"errors": []}), \
             mock.patch.object(launch_speedup, "print_health"), \
             mock.patch.object(launch_speedup.os, "chdir"), \
             mock.patch.object(launch_speedup.subprocess, "run") as run:
            run.return_value.returncode = 7
            self.assertEqual(launch_speedup.main(), 7)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--prompt") + 1], "a horse & a tree")
        self.assertEqual(command.count("--with-baseline"), 1)

    def test_failed_check_does_not_start_generation(self):
        with mock.patch("sys.argv", ["launch_speedup.py", "--mode", "1"]), \
             mock.patch.object(launch_speedup, "check_project", return_value={"errors": ["missing"]}), \
             mock.patch.object(launch_speedup, "print_health"), \
             mock.patch.object(launch_speedup.os, "chdir"), \
             mock.patch.object(launch_speedup.subprocess, "run") as run:
            self.assertEqual(launch_speedup.main(), 1)
            run.assert_not_called()

    def test_menu_selects_profile_and_reprompts_invalid_input(self):
        for answers, expected in [(["1"], None), ([""], None),
                                  (["2"], launch_speedup.FAST_PROFILE),
                                  (["invalid", "2"], launch_speedup.FAST_PROFILE)]:
            with self.subTest(answers=answers), \
                 mock.patch("sys.argv", ["launch_speedup.py"]), \
                 mock.patch("builtins.input", side_effect=answers), \
                 mock.patch.object(launch_speedup, "check_project", return_value={"errors": []}) as check, \
                 mock.patch.object(launch_speedup, "print_health"), \
                 mock.patch.object(launch_speedup.os, "chdir"), \
                 mock.patch.object(launch_speedup.subprocess, "run") as run:
                run.return_value.returncode = 0
                self.assertEqual(launch_speedup.main(), 0)
                self.assertEqual(check.call_args.args[3], expected)
                command = run.call_args.args[0]
                self.assertIn("--with-baseline", command)
                if expected is None:
                    self.assertNotIn("--controller-profile", command)
                else:
                    self.assertEqual(command[command.index("--controller-profile") + 1], str(expected))

    def test_check_only_never_prompts_or_generates(self):
        with mock.patch("sys.argv", ["launch_speedup.py", "--check-only"]), \
             mock.patch("builtins.input") as prompt, \
             mock.patch.object(launch_speedup, "check_project", return_value={"errors": []}), \
             mock.patch.object(launch_speedup, "print_health"), \
             mock.patch.object(launch_speedup.os, "chdir"), \
             mock.patch.object(launch_speedup.subprocess, "run") as run:
            self.assertEqual(launch_speedup.main(), 0)
            prompt.assert_not_called()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
