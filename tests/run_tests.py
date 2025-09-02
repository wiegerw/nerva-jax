#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import os
import sys
import time
import unittest
from typing import Iterable

# ANSI colors
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def supports_color(stream) -> bool:
    if not stream.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


class ColorTextTestResult(unittest.TextTestResult):
    def __init__(self, *args, enable_color: bool = True, show_duration: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_color = enable_color
        self._show_duration = show_duration
        self._started_at = {}
        self._suite_start = 0.0
        self._suite_end = 0.0

    # Suite timing hooks
    def startTestRun(self) -> None:
        self._suite_start = time.perf_counter()
        return super().startTestRun()

    def stopTestRun(self) -> None:
        self._suite_end = time.perf_counter()
        return super().stopTestRun()

    def _c(self, text: str, color: str) -> str:
        if not self._enable_color:
            return text
        return f"{color}{text}{Color.RESET}"

    def startTest(self, test: unittest.TestCase) -> None:
        # Use base TestResult.startTest to avoid default printing in TextTestResult
        unittest.TestResult.startTest(self, test)
        self._test = test  # keep track like TextTestResult does
        self._started_at[test] = time.perf_counter()
        if self.showAll:
            desc = self.getDescription(test)
            self.stream.write(desc)
            self.stream.write(" ... ")
            self.stream.flush()

    def _finish_with(self, label: str, color: str) -> None:
        if self.showAll:
            if self._show_duration:
                # duration from last started test
                test = getattr(self, '_test', None)
                duration_ms = None
                if test in self._started_at:
                    duration_ms = int((time.perf_counter() - self._started_at[test]) * 1000)
                if duration_ms is not None:
                    label = f"{label} ({duration_ms} ms)"
            self.stream.writeln(self._c(label, color))
        elif self.dots:
            self.stream.write(self._c(label[:1], color))
            self.stream.flush()

    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        self._finish_with("ok", Color.GREEN)

    def addFailure(self, test, err):
        # Format like TextTestResult but avoid its printing
        self.failures.append((test, self._exc_info_to_string(err, test)))
        self._finish_with("FAIL", Color.RED)

    def addError(self, test, err):
        self.errors.append((test, self._exc_info_to_string(err, test)))
        self._finish_with("ERROR", Color.MAGENTA)

    def addSkip(self, test, reason):
        self.skipped.append((test, reason))
        label = f"skipped: {reason}" if reason else "skipped"
        self._finish_with(label, Color.BLUE)

    def addExpectedFailure(self, test, err):
        self.expectedFailures.append((test, self._exc_info_to_string(err, test)))
        self._finish_with("expected failure", Color.YELLOW)

    def addUnexpectedSuccess(self, test):
        self.unexpectedSuccesses.append(test)
        self._finish_with("UNEXPECTED SUCCESS", Color.YELLOW)

    # Summary helper
    def print_summary(self):
        total_time = self._suite_end - self._suite_start
        parts = []
        if self.testsRun:
            parts.append(f"{self.testsRun} tests")
        if self.failures:
            parts.append(self._c(f"{len(self.failures)} failed", Color.RED))
        if self.errors:
            parts.append(self._c(f"{len(self.errors)} errors", Color.MAGENTA))
        if self.skipped:
            parts.append(self._c(f"{len(self.skipped)} skipped", Color.BLUE))
        # successes aren't directly tracked; compute
        successes = self.testsRun - (len(self.failures) + len(self.errors) + len(self.skipped))
        if successes:
            parts.append(self._c(f"{successes} passed", Color.GREEN))
        timing = f"in {total_time:.2f}s"
        summary = f"SUMMARY: {' | '.join(parts)} ({timing})"
        if self._enable_color:
            summary = f"{Color.BOLD}{summary}{Color.RESET}"
        self.stream.writeln(summary)


def iter_tests(suite: unittest.TestSuite) -> Iterable[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def filter_suite_by_keyword(suite: unittest.TestSuite, keyword: str) -> unittest.TestSuite:
    """Filter tests by substring found in test.id() or short description."""
    keyword = keyword.strip()
    if not keyword:
        return suite
    cases = []
    for test in iter_tests(suite):
        tid = test.id()
        sdesc = test.shortDescription() or ""
        if keyword in tid or keyword in sdesc:
            cases.append(test)
    return unittest.TestSuite(cases)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run unittest test suite with better output")
    parser.add_argument("names", nargs="*", help="Optional dotted test names to run (unittest syntax)")
    parser.add_argument("-s", "--start-dir", default=None, help="Directory to start discovery (defaults to script's directory)")
    parser.add_argument("-p", "--pattern", default="test*.py", help="File pattern to match tests (default: test*.py)")
    parser.add_argument("-t", "--top-level-dir", default=None, help="Top level directory of project (defaults to repo root)")
    parser.add_argument("-k", "--keyword", default=None, help="Only run tests matching this substring in id or description")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet (dots) output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-f", "--failfast", action="store_true", help="Stop on first failure")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")

    args = parser.parse_args(argv)

    # Determine verbosity
    if args.verbose:
        verbosity = 2
    elif args.quiet:
        verbosity = 1
    else:
        # default to verbose for better usability
        verbosity = 2

    # Resolve default paths relative to this script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    start_dir = args.start_dir or script_dir
    # Default top-level to start_dir so the start dir doesn't need to be a package
    top_level_dir = args.top_level_dir or start_dir

    # Ensure absolute paths for discovery
    start_dir = os.path.abspath(start_dir)
    top_level_dir = os.path.abspath(top_level_dir)

    # Discover or load tests
    loader = unittest.defaultTestLoader

    if args.names:
        suites = [loader.loadTestsFromName(name) for name in args.names]
        suite = unittest.TestSuite(suites)
    else:
        suite = loader.discover(start_dir=start_dir, pattern=args.pattern, top_level_dir=top_level_dir)

    if args.keyword:
        suite = filter_suite_by_keyword(suite, args.keyword)

    if suite.countTestCases() == 0:
        sys.stderr.write("No tests found.\n")
        return 5

    stream = unittest.runner._WritelnDecorator(sys.stderr if args.quiet else sys.stdout)
    enable_color = (not args.no_color) and supports_color(stream.stream)

    result = ColorTextTestResult(stream=stream, descriptions=True, verbosity=verbosity,
                                 enable_color=enable_color, show_duration=True)

    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=verbosity,
        descriptions=True,
        failfast=args.failfast,
        resultclass=lambda *a, **k: result,
        buffer=False,
        warnings=None,
    )

    # Header
    pyver = sys.version.split(" ")[0]
    header = f"Running tests with Python {pyver} | start-dir={start_dir} pattern={args.pattern}"
    if args.keyword:
        header += f" | keyword='{args.keyword}'"
    stream.writeln(header)

    # Run
    res: ColorTextTestResult = runner.run(suite)  # type: ignore

    # Summary line at the end
    res.print_summary()

    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
