# Aider MCP Server - Experimentell
> Model Context Protocol Server zum Auslagern von KI-Codierungsaufgaben an Aider, verbessert die Entwicklungseffizienz und Flexibilit�t.

## �bersicht

Dieser Server erm�glicht es Claude Code, KI-Codierungsaufgaben an Aider, den besten Open-Source-KI-Codierassistenten, auszulagern. Durch die Delegierung bestimmter Codierungsaufgaben an Aider k�nnen wir Kosten reduzieren, die Kontrolle �ber unser Codierungsmodell gewinnen und Claude Code orchestrierter betreiben, um Code zu �berpr�fen und zu �berarbeiten.

## Einrichtung

0. Repository klonen:

```bash
git clone https://github.com/disler/aider-mcp-server.git
```

1. Abh�ngigkeiten installieren:

```bash
uv sync
```

2. Umgebungsdatei erstellen:

```bash
cp .env.sample .env
```

3. Konfigurieren Sie Ihre API-Schl�ssel in der `.env`-Datei (oder verwenden Sie den "env"-Abschnitt der mcpServers), um den f�r das gew�nschte Modell in Aider ben�tigten API-Schl�ssel zu haben:

```
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
...siehe .env.sample f�r weitere
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
        "GEMINI_API_KEY": "<Ihr Gemini API-Schl�ssel>",
        "OPENAI_API_KEY": "<Ihr OpenAI API-Schl�ssel>",
        "ANTHROPIC_API_KEY": "<Ihr Anthropic API-Schl�ssel>",
        ...siehe .env.sample f�r weitere
      }
    }
  }
}
```

## Tests
> Tests laufen mit gemini-2.5-pro-exp-03-25

Um alle Tests auszuf�hren:

```bash
uv run pytest
```

Um bestimmte Tests auszuf�hren:

```bash
# Test der Modellauflistung
uv run pytest src/aider_mcp_server/tests/atoms/tools/test_aider_list_models.py

# Test der KI-Codierung
uv run pytest src/aider_mcp_server/tests/atoms/tools/test_aider_ai_code.py

# Test des KI-Fragemodus
uv run pytest src/aider_mcp_server/tests/atoms/tools/test_aider_ai_ask.py
```

Hinweis: Die KI-Codierungstests ben�tigen einen g�ltigen API-Schl�ssel f�r das Gemini-Modell. Stellen Sie sicher, dass Sie ihn in Ihrer `.env`-Datei festlegen, bevor Sie die Tests ausf�hren.

## Diesen MCP-Server zu Claude Code hinzuf�gen

### Hinzuf�gen mit `gemini-2.5-pro-exp-03-25`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory "<Pfad zum aider-mcp-server-Projekt>" \
  run aider-mcp-server \
  --editor-model "gemini/gemini-2.5-pro-exp-03-25" \
  --current-working-dir "<Pfad zu Ihrem Projekt>"
```

### Hinzuf�gen mit `gemini-2.5-pro-preview-03-25`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory "<Pfad zum aider-mcp-server-Projekt>" \
  run aider-mcp-server \
  --editor-model "gemini/gemini-2.5-pro-preview-03-25" \
  --current-working-dir "<Pfad zu Ihrem Projekt>"
```

### Hinzuf�gen mit `quasar-alpha`

```bash
claude mcp add aider-mcp-server -s local \
  -- \
  uv --directory "<Pfad zum aider-mcp-server-Projekt>" \
  run aider-mcp-server \
  --editor-model "openrouter/openrouter/quasar-alpha" \
  --current-working-dir "<Pfad zu Ihrem Projekt>"
```

### Hinzuf�gen mit `llama4-maverick-instruct-basic`

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
   - �bergibt eine Anfrage und Dateipfade
   - Verwendet Aider, um die angeforderten �nderungen umzusetzen
   - Gibt Erfolg oder Misserfolg zur�ck

2. **Erkl�rungen im Fragemodus anfordern**:
   - F�hrt Aider im Fragemodus aus, um Code zu erkl�ren oder Fragen zu beantworten
   - Bietet Referenz zu relevanten Dateien, ohne diese zu �ndern
   - Gibt detaillierte Erkl�rungen zur�ck

3. **Verf�gbare Modelle auflisten**:
   - Bietet eine Liste von Modellen, die einer Teilzeichenfolge entsprechen
   - N�tzlich zum Entdecken unterst�tzter Modelle

## Verf�gbare Tools

Dieser MCP-Server stellt die folgenden Tools zur Verf�gung:

### 1. `aider_ai_code`

Mit diesem Tool k�nnen Sie Aider ausf�hren, um KI-Codierungsaufgaben basierend auf einer bereitgestellten Anfrage und angegebenen Dateien durchzuf�hren.

**Parameter:**

- `ai_coding_prompt` (String, erforderlich): Die nat�rlichsprachliche Anweisung f�r die KI-Codierungsaufgabe.
- `relative_editable_files` (Liste von Strings, erforderlich): Eine Liste von Dateipfaden (relativ zum `current_working_dir`), die Aider �ndern darf. Wenn eine Datei nicht existiert, wird sie erstellt.
- `relative_readonly_files` (Liste von Strings, optional): Eine Liste von Dateipfaden (relativ zum `current_working_dir`), die Aider zum Kontext lesen, aber nicht �ndern kann. Standardm��ig eine leere Liste `[]`.
- `model` (String, optional): Das prim�re KI-Modell, das Aider zum Generieren von Code verwenden soll. Standardm��ig `"gemini/gemini-2.5-pro-exp-03-25"`. Sie k�nnen das Tool `list_models` verwenden, um andere verf�gbare Modelle zu finden.

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

**R�ckgabewerte:**

- Ein einfaches Dict: {success, diff}
  - `success`: boolean - Ob die Operation erfolgreich war.
  - `diff`: string - Die Differenz der an der Datei vorgenommenen �nderungen.

### 2. `aider_ai_ask`

Mit diesem Tool k�nnen Sie Aider im Fragemodus ausf�hren, um Code zu erkl�ren oder Fragen zu beantworten, ohne Dateien zu �ndern.

**Parameter:**

- `ai_coding_prompt` (String, erforderlich): Die Frage oder Erkl�rungsanfrage f�r Aider.
- `relative_readonly_files` (Liste von Strings, optional): Eine Liste von Dateipfaden (relativ zum `current_working_dir`), die Aider zum Kontext lesen kann. Diese Dateien bieten relevanten Kontext f�r die Frage, k�nnen aber nicht ge�ndert werden. Standardm��ig eine leere Liste `[]`.
- `model` (String, optional): Das KI-Modell, das Aider f�r die Erkl�rung verwenden soll. Standardm��ig `"gemini/gemini-2.5-pro-exp-03-25"`. Sie k�nnen das Tool `list_models` verwenden, um andere verf�gbare Modelle zu finden.

**Verwendungsbeispiel (innerhalb einer MCP-Anfrage):**

Claude Code Anfrage:
```
Verwenden Sie das Aider AI Ask-Tool, um: Zu erkl�ren, wie die calculate_tax-Funktion in tax_calculator.py funktioniert.
```

Ergebnis:
```json
{
  "name": "aider_ai_ask",
  "parameters": {
    "ai_coding_prompt": "Erkl�ren, wie die calculate_tax-Funktion in tax_calculator.py funktioniert.",
    "relative_readonly_files": ["src/tax_calculator.py"],
    "model": "openai/gpt-4o"
  }
}
```

**R�ckgabewerte:**

- Ein einfaches Dict: {success, response}
  - `success`: boolean - Ob die Operation erfolgreich war.
  - `response`: string - Die detaillierte Antwort von Aider, die die Frage beantwortet oder die Erkl�rung liefert.

### 3. `list_models`

Dieses Tool listet verf�gbare KI-Modelle auf, die von Aider unterst�tzt werden und einer bestimmten Teilzeichenfolge entsprechen.

**Parameter:**

- `substring` (String, erforderlich): Die Teilzeichenfolge, nach der in den Namen der verf�gbaren Modelle gesucht werden soll.

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

**R�ckgabewerte:**

- Eine Liste von Modellname-Strings, die der angegebenen Teilzeichenfolge entsprechen. Beispiel: `["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro", "gemini/gemini-pro"]`

## Architektur

Der Server ist wie folgt strukturiert:

- **Server-Schicht**: Behandelt die MCP-Protokollkommunikation
- **Atoms-Schicht**: Einzelne, rein funktionale Komponenten
  - **Tools**: Spezifische F�higkeiten (KI-Codierung, KI-Fragemodus, Auflistung von Modellen)
  - **Utils**: Konstanten und Hilfsfunktionen
  - **Data Types**: Typdefinitionen mit Pydantic

Alle Komponenten werden gr�ndlich auf Zuverl�ssigkeit getestet.

## Codebasis-Struktur

Das Projekt ist in die folgenden Hauptverzeichnisse und Dateien organisiert:

```
.
   ai_docs                   # Dokumentation zu KI-Modellen und Beispielen
      just-prompt-example-mcp-server.xml
      programmable-aider-documentation.md
   pyproject.toml            # Projektmetadaten und -abh�ngigkeiten
   README.md                 # Diese Datei (englische Version)
   README_DE.md              # Diese Datei (deutsche Version)
   specs                     # Spezifikationsdokumente
      init-aider-mcp-exp.md
   src                       # Quellcode-Verzeichnis
      aider_mcp_server      # Hauptpaket f�r den Server
          __init__.py       # Paket-Initialisierer
          __main__.py       # Haupteinstiegspunkt f�r das Server-Programm
          atoms             # Kernkomponenten (reine Funktionen)
             __init__.py
             data_types.py # Pydantic-Modelle f�r Datenstrukturen
             logging.py    # Benutzerdefinierte Logging-Einrichtung
             tools         # Einzelne Tool-Implementierungen
                __init__.py
                aider_common.py # Gemeinsamer Code zwischen Aider-Tools
                aider_ai_code.py # Logik f�r das aider_ai_code-Tool
                aider_ai_ask.py # Logik f�r das aider_ai_ask-Tool
                aider_list_models.py # Logik f�r das list_models-Tool
             utils.py      # Hilfsfunktionen und Konstanten
          server.py         # MCP-Server-Logik, Tool-Registrierung, Anfrageverarbeitung
          tests             # Unit- und Integrationstests
              __init__.py
              atoms         # Tests f�r die Atoms-Schicht
                  __init__.py
                  test_logging.py # Tests f�r Logging
                  tools     # Tests f�r die Tools
                      __init__.py
                      test_aider_ai_code.py # Tests f�r das KI-Codierungs-Tool
                      test_aider_ai_ask.py # Tests f�r das KI-Frage-Tool
                      test_aider_list_models.py # Tests f�r das Modellauflistungs-Tool
```

- **`src/aider_mcp_server`**: Enth�lt den Hauptanwendungscode.
  - **`atoms`**: Enth�lt die grundlegenden Bausteine. Diese sind als reine Funktionen oder einfache Klassen mit minimalen Abh�ngigkeiten konzipiert.
    - **`tools`**: Jede Datei hier implementiert die Kernlogik f�r ein bestimmtes MCP-Tool (`aider_ai_code`, `aider_ai_ask`, `list_models`).
    - **`utils.py`**: Enth�lt gemeinsame Konstanten wie Standardmodellnamen.
    - **`data_types.py`**: Definiert Pydantic-Modelle f�r Anfrage-/Antwortstrukturen, um die Datenvalidierung zu gew�hrleisten.
    - **`logging.py`**: Richtet ein konsistentes Logging-Format f�r Konsolen- und Dateiausgabe ein.
  - **`server.py`**: Orchestriert den MCP-Server. Er initialisiert den Server, registriert die in den `atoms/tools`-Verzeichnissen definierten Tools, bearbeitet eingehende Anfragen, leitet sie an die entsprechende Tool-Logik weiter und sendet Antworten gem�� dem MCP-Protokoll zur�ck.
  - **`__main__.py`**: Stellt den Einstiegspunkt f�r die Befehlszeilenschnittstelle (`aider-mcp-server`) bereit, analysiert Argumente wie `--editor-model` und startet den in `server.py` definierten Server.
  - **`tests`**: Enth�lt Tests, die die Struktur des `src`-Verzeichnisses widerspiegeln und sicherstellen, dass jede Komponente (insbesondere Atome) wie erwartet funktioniert.