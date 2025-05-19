# Aider MCP Server - Experimentell
> Model Context Protocol Server zum Auslagern von KI-Codierungsaufgaben an Aider, verbessert die Entwicklungseffizienz und Flexibilität.

## Übersicht

Dieser Server ermöglicht es Claude Code, KI-Codierungsaufgaben an Aider, den besten Open-Source-KI-Codierassistenten, auszulagern. Durch die Delegierung bestimmter Codierungsaufgaben an Aider können wir Kosten reduzieren, die Kontrolle über unser Codierungsmodell gewinnen und Claude Code orchestrierter betreiben, um Code zu überprüfen und zu überarbeiten.

## Einrichtung

0. Repository klonen:

```bash
git clone https://github.com/disler/aider-mcp-server.git
```

1. Abhängigkeiten installieren:

```bash
uv sync
```

2. Umgebungsdatei erstellen:

```bash
cp .env.sample .env
```

3. Konfigurieren Sie Ihre API-Schlüssel in der `.env`-Datei (oder verwenden Sie den "env"-Abschnitt der mcpServers), um den für das gewünschte Modell in Aider benötigten API-Schlüssel zu haben:

```
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
...siehe .env.sample für weitere
```

4. Kopieren Sie die `.mcp.json` in das Stammverzeichnis Ihres Projekts und aktualisieren Sie `--directory`, um auf das Stammverzeichnis dieses Projekts zu zeigen, und `--current-working-dir`, um auf das Stammverzeichnis Ihres Projekts zu zeigen.

```json
{
  "mcpServers": {
    "aider-mcp-server": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory",
        "<Pfad zu diesem Projekt>",
        "run",
        "aider-mcp-server",
        "--editor-model",
        "gpt-4o",
        "--current-working-dir",
        "<Pfad zu Ihrem Projekt>"
      ],
      "env": {
        "GEMINI_API_KEY": "<Ihr Gemini API-Schlüssel>",
        "OPENAI_API_KEY": "<Ihr OpenAI API-Schlüssel>",
        "ANTHROPIC_API_KEY": "<Ihr Anthropic API-Schlüssel>",
        ...siehe .env.sample für weitere
      }
    }
  }
}
```

## Tests
> Tests laufen mit gemini-2.5-pro-exp-03-25

Um alle Tests auszuführen:

```bash
uv run pytest
```

Um bestimmte Tests auszuführen:

```bash
# Test der Modellauflistung
uv run pytest src/aider_mcp_server/tests/atoms/tools/test_aider_list_models.py

# Test der KI-Codierung
uv run pytest src/aider_mcp_server/tests/atoms/tools/test_aider_ai_code.py

# Test des KI-Fragemodus
uv run pytest src/aider_mcp_server/tests/atoms/tools/test_aider_ai_ask.py
```

Hinweis: Die KI-Codierungstests benötigen einen gültigen API-Schlüssel für das Gemini-Modell. Stellen Sie sicher, dass Sie ihn in Ihrer `.env`-Datei festlegen, bevor Sie die Tests ausführen.

## Diesen MCP-Server zu Claude Code hinzufügen

### Hinzufügen mit `gemini-2.5-pro-exp-03-25`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory "<Pfad zum aider-mcp-server-Projekt>" \
  run aider-mcp-server \
  --editor-model "gemini/gemini-2.5-pro-exp-03-25" \
  --current-working-dir "<Pfad zu Ihrem Projekt>"
```

### Hinzufügen mit `gemini-2.5-pro-preview-03-25`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory "<Pfad zum aider-mcp-server-Projekt>" \
  run aider-mcp-server \
  --editor-model "gemini/gemini-2.5-pro-preview-03-25" \
  --current-working-dir "<Pfad zu Ihrem Projekt>"
```

### Hinzufügen mit `quasar-alpha`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory "<Pfad zum aider-mcp-server-Projekt>" \
  run aider-mcp-server \
  --editor-model "openrouter/openrouter/quasar-alpha" \
  --current-working-dir "<Pfad zu Ihrem Projekt>"
```

### Hinzufügen mit `llama4-maverick-instruct-basic`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory "<Pfad zum aider-mcp-server-Projekt>" \
  run aider-mcp-server \
  --editor-model "fireworks_ai/accounts/fireworks/models/llama4-maverick-instruct-basic" \
  --current-working-dir "<Pfad zu Ihrem Projekt>"
```

## Verwendung

Dieser MCP-Server bietet die folgenden Funktionen:

1. **KI-Codierungsaufgaben an Aider auslagern**:
   - Übergibt eine Anfrage und Dateipfade
   - Verwendet Aider, um die angeforderten Änderungen umzusetzen
   - Gibt Erfolg oder Misserfolg zurück

2. **Erklärungen im Fragemodus anfordern**:
   - Führt Aider im Fragemodus aus, um Code zu erklären oder Fragen zu beantworten
   - Bietet Referenz zu relevanten Dateien, ohne diese zu ändern
   - Gibt detaillierte Erklärungen zurück

3. **Verfügbare Modelle auflisten**:
   - Bietet eine Liste von Modellen, die einer Teilzeichenfolge entsprechen
   - Nützlich zum Entdecken unterstützter Modelle

## Verfügbare Tools

Dieser MCP-Server stellt die folgenden Tools zur Verfügung:

### 1. `aider_ai_code`

Mit diesem Tool können Sie Aider ausführen, um KI-Codierungsaufgaben basierend auf einer bereitgestellten Anfrage und angegebenen Dateien durchzuführen.

**Parameter:**

- `ai_coding_prompt` (String, erforderlich): Die natürlichsprachliche Anweisung für die KI-Codierungsaufgabe.
- `relative_editable_files` (Liste von Strings, erforderlich): Eine Liste von Dateipfaden (relativ zum `current_working_dir`), die Aider ändern darf. Wenn eine Datei nicht existiert, wird sie erstellt.
- `relative_readonly_files` (Liste von Strings, optional): Eine Liste von Dateipfaden (relativ zum `current_working_dir`), die Aider zum Kontext lesen, aber nicht ändern kann. Standardmäßig eine leere Liste `[]`.
- `model` (String, optional): Das primäre KI-Modell, das Aider zum Generieren von Code verwenden soll. Standardmäßig `"gemini/gemini-2.5-pro-exp-03-25"`. Sie können das Tool `list_models` verwenden, um andere verfügbare Modelle zu finden.

**Verwendungsbeispiel (innerhalb einer MCP-Anfrage):**

Claude Code Anfrage:
```
Verwenden Sie das Aider AI Code-Tool, um: Die calculate_sum-Funktion in calculator.py umzuarbeiten, um potenzielle TypeError-Ausnahmen zu behandeln.
```

Ergebnis:
```json
{
  "name": "aider_ai_code",
  "parameters": {
    "ai_coding_prompt": "Die calculate_sum-Funktion in calculator.py umarbeiten, um potenzielle TypeError-Ausnahmen zu behandeln.",
    "relative_editable_files": ["src/calculator.py"],
    "relative_readonly_files": ["docs/requirements.txt"],
    "model": "openai/gpt-4o"
  }
}
```

**Rückgabewerte:**

- Ein einfaches Dict: {success, diff}
  - `success`: boolean - Ob die Operation erfolgreich war.
  - `diff`: string - Die Differenz der an der Datei vorgenommenen Änderungen.

### 2. `aider_ai_ask`

Mit diesem Tool können Sie Aider im Fragemodus ausführen, um Code zu erklären oder Fragen zu beantworten, ohne Dateien zu ändern.

**Parameter:**

- `ai_coding_prompt` (String, erforderlich): Die Frage oder Erklärungsanfrage für Aider.
- `relative_readonly_files` (Liste von Strings, optional): Eine Liste von Dateipfaden (relativ zum `current_working_dir`), die Aider zum Kontext lesen kann. Diese Dateien bieten relevanten Kontext für die Frage, können aber nicht geändert werden. Standardmäßig eine leere Liste `[]`.
- `model` (String, optional): Das KI-Modell, das Aider für die Erklärung verwenden soll. Standardmäßig `"gemini/gemini-2.5-pro-exp-03-25"`. Sie können das Tool `list_models` verwenden, um andere verfügbare Modelle zu finden.

**Verwendungsbeispiel (innerhalb einer MCP-Anfrage):**

Claude Code Anfrage:
```
Verwenden Sie das Aider AI Ask-Tool, um: Zu erklären, wie die calculate_tax-Funktion in tax_calculator.py funktioniert.
```

Ergebnis:
```json
{
  "name": "aider_ai_ask",
  "parameters": {
    "ai_coding_prompt": "Erklären, wie die calculate_tax-Funktion in tax_calculator.py funktioniert.",
    "relative_readonly_files": ["src/tax_calculator.py"],
    "model": "openai/gpt-4o"
  }
}
```

**Rückgabewerte:**

- Ein einfaches Dict: {success, response}
  - `success`: boolean - Ob die Operation erfolgreich war.
  - `response`: string - Die detaillierte Antwort von Aider, die die Frage beantwortet oder die Erklärung liefert.

### 3. `list_models`

Dieses Tool listet verfügbare KI-Modelle auf, die von Aider unterstützt werden und einer bestimmten Teilzeichenfolge entsprechen.

**Parameter:**

- `substring` (String, erforderlich): Die Teilzeichenfolge, nach der in den Namen der verfügbaren Modelle gesucht werden soll.

**Verwendungsbeispiel (innerhalb einer MCP-Anfrage):**

Claude Code Anfrage:
```
Verwenden Sie das Aider List Models-Tool, um: Modelle aufzulisten, die die Teilzeichenfolge "gemini" enthalten.
```

Ergebnis:
```json
{
  "name": "list_models",
  "parameters": {
    "substring": "gemini"
  }
}
```

**Rückgabewerte:**

- Eine Liste von Modellname-Strings, die der angegebenen Teilzeichenfolge entsprechen. Beispiel: `["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro", "gemini/gemini-pro"]`

## Architektur

Der Server ist wie folgt strukturiert:

- **Server-Schicht**: Behandelt die MCP-Protokollkommunikation
- **Atoms-Schicht**: Einzelne, rein funktionale Komponenten
  - **Tools**: Spezifische Fähigkeiten (KI-Codierung, KI-Fragemodus, Auflistung von Modellen)
  - **Utils**: Konstanten und Hilfsfunktionen
  - **Data Types**: Typdefinitionen mit Pydantic

Alle Komponenten werden gründlich auf Zuverlässigkeit getestet.

## Codebasis-Struktur

Das Projekt ist in die folgenden Hauptverzeichnisse und Dateien organisiert:

```
.
   ai_docs                   # Dokumentation zu KI-Modellen und Beispielen
      just-prompt-example-mcp-server.xml
      programmable-aider-documentation.md
   pyproject.toml            # Projektmetadaten und -abhängigkeiten
   README.md                 # Diese Datei (englische Version)
   README_DE.md              # Diese Datei (deutsche Version)
   specs                     # Spezifikationsdokumente
      init-aider-mcp-exp.md
   src                       # Quellcode-Verzeichnis
      aider_mcp_server      # Hauptpaket für den Server
          __init__.py       # Paket-Initialisierer
          __main__.py       # Haupteinstiegspunkt für das Server-Programm
          atoms             # Kernkomponenten (reine Funktionen)
             __init__.py
             data_types.py # Pydantic-Modelle für Datenstrukturen
             logging.py    # Benutzerdefinierte Logging-Einrichtung
             tools         # Einzelne Tool-Implementierungen
                __init__.py
                aider_common.py # Gemeinsamer Code zwischen Aider-Tools
                aider_ai_code.py # Logik für das aider_ai_code-Tool
                aider_ai_ask.py # Logik für das aider_ai_ask-Tool
                aider_list_models.py # Logik für das list_models-Tool
             utils.py      # Hilfsfunktionen und Konstanten
          server.py         # MCP-Server-Logik, Tool-Registrierung, Anfrageverarbeitung
          tests             # Unit- und Integrationstests
              __init__.py
              atoms         # Tests für die Atoms-Schicht
                  __init__.py
                  test_logging.py # Tests für Logging
                  tools     # Tests für die Tools
                      __init__.py
                      test_aider_ai_code.py # Tests für das KI-Codierungs-Tool
                      test_aider_ai_ask.py # Tests für das KI-Frage-Tool
                      test_aider_list_models.py # Tests für das Modellauflistungs-Tool
```

- **`src/aider_mcp_server`**: Enthält den Hauptanwendungscode.
  - **`atoms`**: Enthält die grundlegenden Bausteine. Diese sind als reine Funktionen oder einfache Klassen mit minimalen Abhängigkeiten konzipiert.
    - **`tools`**: Jede Datei hier implementiert die Kernlogik für ein bestimmtes MCP-Tool (`aider_ai_code`, `aider_ai_ask`, `list_models`).
    - **`utils.py`**: Enthält gemeinsame Konstanten wie Standardmodellnamen.
    - **`data_types.py`**: Definiert Pydantic-Modelle für Anfrage-/Antwortstrukturen, um die Datenvalidierung zu gewährleisten.
    - **`logging.py`**: Richtet ein konsistentes Logging-Format für Konsolen- und Dateiausgabe ein.
  - **`server.py`**: Orchestriert den MCP-Server. Er initialisiert den Server, registriert die in den `atoms/tools`-Verzeichnissen definierten Tools, bearbeitet eingehende Anfragen, leitet sie an die entsprechende Tool-Logik weiter und sendet Antworten gemäß dem MCP-Protokoll zurück.
  - **`__main__.py`**: Stellt den Einstiegspunkt für die Befehlszeilenschnittstelle (`aider-mcp-server`) bereit, analysiert Argumente wie `--editor-model` und startet den in `server.py` definierten Server.
  - **`tests`**: Enthält Tests, die die Struktur des `src`-Verzeichnisses widerspiegeln und sicherstellen, dass jede Komponente (insbesondere Atome) wie erwartet funktioniert.