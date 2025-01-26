def pytest_terminal_summary(terminalreporter, exitstatus):
    reports = terminalreporter.getreports("")
    for report in reports:
        if report.nodeid == "tests/test_docs.py::test_mkdocs_links" and report.capstdout:
            terminalreporter.ensure_newline()
            terminalreporter.section("Docs Broken Links", sep="-", blue=True, bold=True)
            terminalreporter.line(report.capstdout)
