# Frontend Context — AutoML Scientist

The frontend for **AutoML Scientist** is a premium, retro-scientific single-page application (SPA) built using React, TypeScript, and TailwindCSS. It features a rich, typewriter-inspired aesthetic resembling a physical research notebook and lab dashboard.

---

## Technical Stack & Routing

| Layer | Choice | Details |
|---|---|---|
| **Build Tool** | Vite | Ultra-fast development and compilation |
| **Framework** | React 18+ | Component-driven declarative UI |
| **Routing** | React Router DOM v6 | Client-side page navigation |
| **Styling** | CSS + TailwindCSS | Premium fonts, paper textures, cards, custom fonts |
| **Icons** | Google Material Symbols | Consistent retro iconography |

---

## Repository Structure (Frontend)

```text
HackDayFrontend/automl-scientist/
├── src/
│   ├── main.tsx             # React SPA entrypoint
│   ├── App.tsx              # Base Layout, Sidebar navigation, and Route definitions
│   ├── index.css            # Custom fonts (Inter, Playfair Display, Outfit) and global styles
│   │
│   ├── pages/               # Page Components
│   │   ├── LaboratoryEntrance.tsx   # Dataset upload, clean preview, & target selection
│   │   ├── ResearchNotebook.tsx     # Live SSE log streaming, notebook entries
│   │   └── ExperimentalFindings.tsx  # Best model list, metrics, & visual charts
│   │
│   └── components/          # Reusable UI widgets
```

---

## Core Pages & Features

### 1. Laboratory Entrance (`/entrance`)
- **Workflow**: 
  - Drag-and-drop or select tabular CSV datasets.
  - Automatically queries the backend to parse the target variable, cardinality, and column statistics.
  - Prompts target column selection, and triggers AI cleaning code generation.
  - Displays a clean visual preview comparison of dataset dimensions (before vs. after).

### 2. Research Notebook (`/notebook`)
- **Workflow**:
  - Connects to the backend orchestrator via a live **Server-Sent Event (SSE)** stream (`/api/orchestrate/<session_id>`).
  - Displays live typed narration logs in a physical notebook paper layout.
  - Dynamically updates a vertical timeline of inferences for each finished model training iteration.
  - Renders a custom retro topology doodle matching session states.
  - **Completion Banner**: Renders a premium, highlighted completion box once the SSE stream finishes, allowing the user to click and navigate directly to `/experiments` to review and download `.pkl` binaries.

### 3. Model Portfolio & Experiments (`/experiments`)
- **Workflow**:
  - Lists all successfully trained models in order of validation score, highlighting the **optimal model** with a green checkmark stamp.
  - Provides a direct link to download the `.pkl` binary for the optimal model from the Flask endpoint `/api/download/<session_id>`.
  - **Visual Evaluations**:
    - **Confusion Matrix**: Tries to load the server-generated plot `/static/plots/{sessionId}_confusion.png`. If it fails or is not applicable, it falls back to a clean, interactive CSS matrix highlighting classification error percentages.
    - **Feature Importance**: Tries to load the server-generated plot `/static/plots/{sessionId}_importance.png`. If it fails, it falls back to a ranked horizontal progress-bar list of top-contributing dataset features.

---

## State Management & Persistent Cache

The frontend synchronizes state between React Router page state and `localStorage` to survive page reloads:
- `automl_session_id`: Unique active research session UUID.
- `automl_filename`: Name of the uploaded dataset.
- `automl_problem_type`: Inferred task type (`classification` / `regression`).
- `automl_target_column`: Selected prediction target.
- `automl_summary`: Dataset profile metadata.
