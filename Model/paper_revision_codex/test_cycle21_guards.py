import importlib.util
from pathlib import Path


def guard_tool():
    spec = importlib.util.spec_from_file_location('guard_mutation_check',
                                                  Path(__file__).parent / 'report_tools' / 'guard_mutation_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_tool_counts_only_failing_tests_as_killed():
    tool = guard_tool()
    assert tool.classify(1, '.F.\n1 failed, 83 passed, 2 deselected in 90.00s') == 'killed'
    assert tool.classify(1, 'E\n2 errors, 82 passed in 9.00s') == 'killed'
    assert tool.classify(0, '84 passed, 2 deselected in 90.00s') == 'survived'
    assert tool.classify(1073807364, '........................F...') == 'aborted'
    assert tool.classify(2, '!!! KeyboardInterrupt !!!\n1 failed, 3 passed in 2.00s') == 'aborted'
    assert tool.classify(1, 'some output without a pytest summary') == 'aborted'


def test_guard_tool_survives_a_run_with_no_output():
    tool = guard_tool()
    assert tool.summary('') == '' and tool.summary(None) == ''
    assert tool.classify(3221225786, '') == 'aborted'
