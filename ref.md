Refine the EXISTING VGA Cable EMI Analyzer UI to look like professionally engineered scientific measurement software. This is a visual and usability refinement task, not a rewrite.

First inspect the existing project and preserve all working analysis, CSV import, screenshot extraction, calibration, exports, sessions, tests, and demo functionality.

## Design direction

Reference the visual conventions of professional laboratory instrumentation and engineering analysis applications. The result must feel restrained, precise, and intentional—not like a generic AI-generated dashboard or a startup landing page.

### 1. Remove generic AI-dashboard styling

- Remove emoji from navigation labels, headings, buttons, status panels, and graph titles.
- Remove unnecessary decorative icons and oversized promotional headings.
- Eliminate excessive rounded cards, gradients, glow effects, bright borders, and decorative containers.
- Remove repeated explanatory text and unnecessary warning banners.
- Avoid excessive bold text and inconsistent typography.
- Do not add artificial loading animations or decorative metrics.

### 2. Professional application shell

- Use a compact, consistent application header with the product name and a subtle version label.
- Use a narrow, organized sidebar for operational settings.
- Group controls under clear headings.
- Use consistent spacing, typography, control heights, and alignment.
- Use restrained neutral surfaces and borders.
- Keep navigation labels concise and consistent.
- Use standard engineering terminology.
- Maintain readable contrast and accessible focus states.

### 3. Color system

Use a restrained palette:
- Main application background: neutral graphite or light gray.
- Panels: slightly differentiated neutral surfaces.
- Text: high-contrast neutral foreground.
- Borders and grid lines: subtle gray.
- Status colors: muted green for success, amber for warnings, red for errors.
- Reserve bright yellow, cyan, magenta, and green for actual oscilloscope traces only.

Do not recolor the scientific traces arbitrarily if consistent channel colors already exist.

### 4. Oscilloscope dashboard

Preserve the six-panel instrument-inspired layout.

- Use consistent plot dimensions and alignment.
- Use a dark plotting background, subtle major/minor grids, and restrained trace thickness.
- Improve axis labels, units, legends, and panel titles.
- Remove redundant legends and decorative labels.
- Give individual FFTs, derived calculations, and time-domain waveforms clearly distinguishable titles.
- Mark illustrative calculations with compact text inside the relevant panel.
- Never hide the fact that synthetic data is being shown.

The graphs should resemble engineering instrumentation, not a neon gaming dashboard.

### 5. Final EMI spectrum

Preserve the clean white-background publication-style graph.

- Use a white plotting area, dark axes, subtle gray grid lines, and one blue spectrum trace.
- Use sensible typography and compact peak annotations.
- Avoid overlapping peak labels; use collision-aware positioning or annotate only the most significant peaks.
- Display frequency and amplitude units accurately.
- Use a restrained legend and concise title.
- Keep the graph export high-resolution and suitable for inclusion in a technical report.

### 6. Demo and measurement status

Replace the oversized yellow banner with a compact persistent status indicator:
- DEMO — SYNTHETIC DATA
- MEASUREMENT — IMPORTED DATA

Use a subtle amber indicator for demo mode and a neutral/green indicator for valid imported data. Never imply that imported data is calibrated merely because it was imported.

Display warnings near the relevant measurement or calculation instead of repeating them across the entire interface.

### 7. Sidebar controls

Organize settings into logical groups:
- Operation mode
- Input and channel selection
- Frequency range
- Axis scaling
- Peak detection
- Reference level
- Waveform settings

Use compact labels and sensible defaults. Avoid unnecessary emoji and redundant descriptions.

### 8. Scientific presentation

- Separate input measurements, derived calculations, and final results.
- Show units and source provenance near the relevant result.
- Clearly label unverified calculations.
- Do not claim regulatory compliance.
- Keep measurement metadata and processing details accessible without overwhelming the primary graph.

### 9. Implementation constraints

- Keep Python as the only application programming language.
- Preserve the existing Streamlit application and architecture.
- Do not replace functional components with static mockups.
- Do not remove existing features.
- Do not change scientific calculations during this visual refinement.
- Avoid unnecessary dependencies.
- Maintain compatibility with the current Python environment.
- Run all existing tests after changes and fix regressions.

### 10. Final verification

Launch the application and inspect the Overview, Import Measurements, Review and Compare, EMI Spectrum, Export Results, and Sessions sections.

Verify that the UI is visually consistent, plots remain readable, controls still work, and the application remains functional.

Report changed files, tests executed, and any remaining visual or functional issues.

The target is a restrained, technically credible laboratory application that an engineer would take seriously—not an AI-generated template.