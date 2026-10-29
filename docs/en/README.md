# English documentation

The Chinese documentation is the primary reference; these English pages cover installation,
operation and the SDK. They are kept in sync with the current code, not a literal translation
of every Chinese page.

| Document | Read it when |
|---|---|
| [../README_EN.md](../README_EN.md) | You want the 5-minute overview |
| [installation.md](installation.md) | You need to install the EXE or build the source environment |
| [usage.md](usage.md) | You want to operate the UI, shortcuts and output files |
| [sdk.md](sdk.md) | You want to call the Python SDK |
| [faq.md](faq.md) | Something does not work |
| [../SDK_API参考.md](../SDK_API参考.md) | You need the complete API reference (Chinese, table by table) |
| [../安装指南.md](../安装指南.md) | The full Chinese installation guide with the wheel matrix |

## Language switch

The desktop UI supports Chinese and English:

- `config.yaml` → `ui.language: en`
- or launch with `--lang en` (`python app_entry.py --lang en`, `StereoCamera.exe --lang en`)

Command-line tools and log lines are Chinese-first; the GUI chrome, status bar and
notifications switch to English.
