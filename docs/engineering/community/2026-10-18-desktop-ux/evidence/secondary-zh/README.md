# Secondary localized observations

These synthetic Chinese-language observations were collected in the initial audit.
They are retained as supplementary evidence only. The primary report, issue
reproduction steps, screenshots, design mocks, and priorities now use the English
product journey and the English retest.

The Chinese rewrite request triggered local file approvals twice with Personal
Intelligence Off. Disabling only local_write led to local_read/local_search;
disabling local file tools removed those approvals but a web_search chip appeared.
Disabling all tools still produced a misread rewriting task. A more explicitly
quoted control produced three Chinese variants. These few samples are not evidence
of a failure rate or of the quality of the English experience.

- [File approval](unexpected-file-approval.png)
- [Local file tools disabled](local-tools-disabled-answer.png)
- [All tools disabled](all-tools-disabled-answer.png)
- [Explicitly quoted control](explicit-prompt-control.png)
