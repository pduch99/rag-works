---
name: "Citrine IDE"
description: "Premium IDE light theme. Warm paper canvas, three syntax-color tokens (amber, teal, plum) used semantically, JetBrains Mono for structure, IBM Plex Sans for body, refined 4-6px corners. Editor-aesthetic with warmth."
tags: [coding, light, developer, premium, warm]
colors:
  primary:   "#1A1814"
  secondary: "#5F574B"
  tertiary:  "#B57A1A"
  neutral:   "#FAF6EC"
  surface:   "#FFFCF4"
typography:
  display: "JetBrains Mono"
  body:    "IBM Plex Sans"
  mono:    "JetBrains Mono"
  scale:
    hero: "3.75rem / 1.05 / 500 / -0.035em"
    h1:   "2.25rem / 1.1 / 500 / -0.03em"
    h2:   "1.25rem / 1.3 / 500 / -0.012em"
    body: "1rem / 1.65 / 400 / -0.003em"
radius:
  sm: 4px
  md: 6px
  lg: 8px
shadows:
  card:   "0 0 0 1px rgba(26, 24, 20, 0.08)"
  button: "0 2px 0 0 #8A5A10"
borders:
  card:    "1px solid rgba(26, 24, 20, 0.08)"
  divider: "rgba(26, 24, 20, 0.12)"
buttons:
  primary:
    background: #B57A1A
    color: #FFFCF4
    border: 1px solid #B57A1A
    shape: rounded
    padding: 10px 20px
    font: mono / 500 / 0.8125rem / -0.005em
    shadow: 0 2px 0 0 #8A5A10
  secondary:
    background: #FFFCF4
    color: #1A1814
    border: 1px solid rgba(181, 122, 26, 0.40)
    shape: rounded
    padding: 10px 20px
    font: mono / 500 / 0.8125rem / -0.005em
  outline:
    background: transparent
    color: #1A1814
    border: 1px solid rgba(26, 24, 20, 0.18)
    shape: rounded
    padding: 10px 20px
    font: mono / 500 / 0.8125rem / -0.005em
  ghost:
    background: transparent
    color: #0F8A82
    border: none
    shape: rounded
    padding: 10px 4px
    font: mono / 500 / 0.8125rem / -0.005em
    hover: underline
charts:
  variant: "rounded-bars"
  stroke_width: 1.5
  gridlines: true
  bar_radius: 2px
  bar_gap: 8px
  highlight: all
  axis_color: "#5F574B"
  palette: ["#B57A1A", "#0F8A82", "#7A2E5A"]
fonts_url: "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
dependencies: ["lucide-react"]
---

# Citrine IDE

## AI Build Instructions

> **Read this section before writing any code.** The rules below
> are non-negotiable. Every value used in the UI must come from this
> file's frontmatter — never substitute, approximate, or invent new
> colors, fonts, radii, or shadows. If a value is missing, ask the
> user before adding one.

### 1 · Your role

You are building UI for a project that has adopted **Citrine IDE** as its
design system. Treat `DESIGN.md` as the single source of truth.
Your job is to translate the user's product requirements into
components and pages that look like they were designed by the same
person who authored this file.

### 2 · Token compliance

- Pull every color, font family, radius, shadow, and spacing value
  from the frontmatter at the top of this file.
- Use semantic roles (e.g. `primary`, `accent`, `muted`) — never
  hard-code hex values that bypass the system.
- When a token can be expressed as a CSS variable, declare it once
  in your global stylesheet and reference it everywhere downstream.
- The Google Fonts `<link>` is provided in the Typography section.
  Add it to `<head>` before any component renders.

### 3 · Component recipes

Use these recipes verbatim when building the corresponding component.

#### Buttons

Four variants are defined. Pick one — never blend variants or invent a fifth.

- **Primary** — rounded shape, bg `#B57A1A`, text `#FFFCF4`, border `1px solid #B57A1A`, padding `10px 20px`, weight `500`, shadow `0 2px 0 0 #8A5A10`.
- **Secondary** — rounded shape, bg `#FFFCF4`, text `#1A1814`, border `1px solid rgba(181, 122, 26, 0.40)`, padding `10px 20px`, weight `500`.
- **Outline** — rounded shape, text `#1A1814`, border `1px solid rgba(26, 24, 20, 0.18)`, padding `10px 20px`, weight `500`.
- **Ghost** — rounded shape, text `#0F8A82`, padding `10px 4px`, weight `500`.

Reach for **primary** as the single dominant CTA per screen.
**Secondary** for the supporting action. **Outline** for tertiary
actions in toolbars. **Ghost** for inline links and table actions.

#### Cards

- Background: `#FFFCF4`
- Border: `1px solid rgba(26, 24, 20, 0.08)`
- Shadow: `0 0 0 1px rgba(26, 24, 20, 0.08)`
- Radius: `radius.lg` (`8px`)
- Internal padding: `20px` for compact cards, `24–28px` for content cards.

#### Charts

- Bar/line variant: `rounded-bars`
- Bar radius: `2px`
- Highlight strategy: `all` — emphasize a single bar/point per chart.
- Use the declared palette in order: `#B57A1A`, `#0F8A82`, `#7A2E5A`.

#### Typography pairings

- **Display (`JetBrains Mono`)** — h1, h2, hero headlines, brand wordmarks.
- **Body (`IBM Plex Sans`)** — paragraphs, labels, button text, form inputs.
- **Mono (`JetBrains Mono`)** — code, eyebrows, metadata, numerals in tables.

### 4 · Hard constraints

Never do any of the following without explicit instruction from the user:

- Introduce a new color, font, radius, or shadow that isn't declared above.
- Mix this system with another (e.g. don't paste in Material or Bootstrap defaults).
- Use generic gradient defaults (purple→blue, peach→pink) — they break the system's voice.
- Reach for emoji icons. Use a consistent icon library and size icons in line with body type.
- Add motion that exceeds the system's restraint — keep transitions short (≤200ms) and subtle.

### 5 · Before you finish — verify

Run through this checklist for every screen you produce:

- [ ] Every color used appears in the Colors table above.
- [ ] Headlines use the display font; body copy uses the body font.
- [ ] Buttons match one of the declared variants exactly (shape, padding, weight).
- [ ] Border-radius values come from `radius.sm` / `radius.md` / `radius.lg` / `radius.pill`.
- [ ] Cards and dividers use the declared border + shadow tokens.
- [ ] No values were invented; if you needed something missing, you stopped and asked.

---

## Overview
Citrine IDE is a premium light theme borrowed from the editor world. The canvas is warm paper (a faint warm tint sets it apart from bone or pure white). Type is **JetBrains Mono** for all structure — display, headings, buttons, table headers — and **IBM Plex Sans** for body. The whole page reads as a structured editor pane.

The system uses **three syntax-token colors as a semantic palette**: amber (string), teal (function/identifier), plum (keyword). The amber is the chromatic accent, the teal is for ghost/link semantics, and plum appears only in charts and code-block highlights.

For developer documentation that wants warmth, AI coding tools, premium code-snippet platforms, design-engineering portfolios.

## Color
- **Ink #1A1814** — primary text. Deep warm graphite.
- **Mid Graphite #5F574B** — secondary text, inline-code default color.
- **Amber #B57A1A** — string token + primary CTA + chromatic accent.
- **Teal #0F8A82** — function/identifier token + ghost-button label + link.
- **Plum #7A2E5A** — keyword token + chart third-bar.
- **Paper #FAF6EC** — page canvas (warm).
- **Surface #FFFCF4** — card / editor pane.

The three syntax colors aren't decoration — they carry semantic meaning across charts, code blocks, and interactive elements.

## Typography
- **Display: JetBrains Mono 500** at 3.75rem with -3.5% tracking. Restrained — the page reads as an editor.
- **Body: IBM Plex Sans 400** at 1rem with 1.65 leading.

| Role | Font | Size | Weight | Tracking |
|------|------|------|--------|----------|
| Hero | JetBrains Mono | 3.75rem | 500 | -0.035em |
| H1 | JetBrains Mono | 2.25rem | 500 | -0.03em |
| H2 | JetBrains Mono | 1.25rem | 500 | -0.012em |
| Body | IBM Plex Sans | 1rem | 400 | -0.003em / 1.65 |
| Button | JetBrains Mono | 0.8125rem | 500 | -0.005em |

Body copy is the only proportional type. Everything structural is mono.

## Geometry
- **Radii: 4 / 6 / 8.** Editor-pane geometry. Never pill, never soft.
- **4px subgrid.**
- **Section gap: 96px** desktop, 64px mobile.

## Buttons
- **Primary** — warm amber fill with a **2px darker-amber bottom border** (`box-shadow: 0 2px 0 0 #8A5A10`). The bottom border reads as an "active tab" indicator — the system's signature.
- **Secondary** — surface paper with a 40% amber hairline. Mono label.
- **Outline** — bare warm-graphite hairline at 18%.
- **Ghost** — bare teal label. Hover underline.

The bottom-border on primary is non-negotiable — it is what makes the button read as an editor element rather than a generic CTA.

## Cards
Surface paper on the warm canvas, 1px ink hairline at 8% opacity, 6px corners, **no drop shadow**. Padding 24-32px. Editor-pane wireframes.

## Code Blocks
Code blocks use the surface color with a 1px hairline; syntax tokens map to the three palette colors (amber/teal/plum) plus warm graphite for default. `Inline code` uses warm graphite on a faint amber-tinted background.

## Charts & Data
Three-bar histogram in amber, teal, plum — the same syntax-token semantic. Gridlines on (developer tool — readability beats minimalism). 2px corner radius, 8px gap. Axis labels in mid-graphite mono at 11px.

## Do's and Don'ts
- ✅ Mono for display and structure. Proportional only for body.
- ✅ The 2px bottom border on primary CTA is the system's signature — keep it.
- ✅ Three syntax-token colors used semantically (amber=string, teal=function, plum=keyword).
- ✅ Warm canvas — not bone, not pure white. The warmth distinguishes it.
- ❌ No second display font. JetBrains Mono carries all structure.
- ❌ No pill buttons. No soft corners.
- ❌ No drop shadows on cards. The bottom-border on primary is the only "shadow."
- ❌ No oversized hero. The page is an editor, not a billboard.

---

## Tokens

> Generated from the same source the live preview renders from.
> Treat the values below as the contract — never substitute approximations.

### Colors

| Role      | Value |
|-----------|-------|
| primary   | `#1A1814` |
| secondary | `#5F574B` |
| tertiary  | `#B57A1A` |
| neutral   | `#FAF6EC` |
| surface   | `#FFFCF4` |

### Typography

- **Display:** JetBrains Mono
- **Body:** IBM Plex Sans
- **Mono:** JetBrains Mono

| Role | size / leading / weight / tracking |
|------|------------------------------------|
| Hero | 3.75rem / 1.05 / 500 / -0.035em |
| H1   | 2.25rem / 1.1 / 500 / -0.03em |
| H2   | 1.25rem / 1.3 / 500 / -0.012em |
| Body | 1rem / 1.65 / 400 / -0.003em |

### Radius

- sm: `4px`
- md: `6px`
- lg: `8px`

### Shadows

- **card:** `0 0 0 1px rgba(26, 24, 20, 0.08)`
- **button:** `0 2px 0 0 #8A5A10`

### Borders

- **card:** `1px solid rgba(26, 24, 20, 0.08)`
- **divider:** `rgba(26, 24, 20, 0.12)`

### Buttons

Four variants, each fully tokenized. The preview renders from these exact values.

#### Primary

| Property | Value |
|----------|-------|
| shape | `rounded` |
| background | `#B57A1A` |
| color | `#FFFCF4` |
| border | `1px solid #B57A1A` |
| padding | `10px 20px` |
| fontFamily | `mono` |
| fontWeight | `500` |
| fontSize | `0.8125rem` |
| tracking | `-0.005em` |
| shadow | `0 2px 0 0 #8A5A10` |

#### Secondary

| Property | Value |
|----------|-------|
| shape | `rounded` |
| background | `#FFFCF4` |
| color | `#1A1814` |
| border | `1px solid rgba(181, 122, 26, 0.40)` |
| padding | `10px 20px` |
| fontFamily | `mono` |
| fontWeight | `500` |
| fontSize | `0.8125rem` |
| tracking | `-0.005em` |

#### Outline

| Property | Value |
|----------|-------|
| shape | `rounded` |
| background | `transparent` |
| color | `#1A1814` |
| border | `1px solid rgba(26, 24, 20, 0.18)` |
| padding | `10px 20px` |
| fontFamily | `mono` |
| fontWeight | `500` |
| fontSize | `0.8125rem` |
| tracking | `-0.005em` |

#### Ghost

| Property | Value |
|----------|-------|
| shape | `rounded` |
| background | `transparent` |
| color | `#0F8A82` |
| border | `none` |
| padding | `10px 4px` |
| fontFamily | `mono` |
| fontWeight | `500` |
| fontSize | `0.8125rem` |
| tracking | `-0.005em` |
| hoverHint | `underline` |

### Charts

| Property | Value |
|----------|-------|
| variant | `rounded-bars` |
| strokeWidth | `1.5` |
| gridlines | `true` |
| barRadius | `2px` |
| barGap | `8px` |
| highlight | `all` |
| axisColor | `#5F574B` |
| palette | `#B57A1A`, `#0F8A82`, `#7A2E5A` |

---

## Pro tokens

> Production-fidelity tokens. States, density, motion, elevation,
> content rules and a measured WCAG contract — derived from the
> resting tokens unless explicitly authored.

### States

#### Button

- **hover** — bg: `rgba(181, 122, 26, 0.15)`, color: `#B57A1A`, border: `1px solid #B57A1A`
- **focus** — outline: `1px dashed #B57A1A`, outline-offset: `2px`
- **active** — bg: `#B57A1A`, color: `#FFFCF4`
- **disabled** — opacity: `0.35`
- **loading** — opacity: `0.6`
- **selected** — bg: `#B57A1A`, color: `#FFFCF4`

#### Input

- **hover** — border: `1px solid rgba(181, 122, 26, 0.5)`
- **focus** — bg: `rgba(181, 122, 26, 0.05)`, border: `1px solid #B57A1A`
- **disabled** — opacity: `0.35`
- **error** — bg: `rgba(239,68,68,0.05)`, border: `1px solid #EF4444`

#### Card

- **hover** — border: `1px solid #B57A1A`
- **selected** — bg: `rgba(181, 122, 26, 0.05)`, border: `1px solid #B57A1A`
- **dragging** — opacity: `0.7`

#### Tab

- **hover** — color: `#B57A1A`
- **focus** — outline: `1px dashed #B57A1A`, outline-offset: `1px`
- **selected** — bg: `rgba(181, 122, 26, 0.1)`, color: `#B57A1A`

### Density

| Mode | padding × | row × | body | radius × | Use for |
|------|-----------|-------|------|----------|---------|
| compact | 0.72 | 0.78 | 0.8125rem | 0.85 | Information-dense — tables, IDEs, dashboards |
| comfortable | 1 | 1 | 0.9375rem | — | Default — most product UI |
| spacious | 1.35 | 1.3 | 1rem | 1.15 | Editorial — marketing, long-form, settings |

### Motion

**Signature — Cursor blink.** Discrete, often stepped transitions. Like a terminal cursor: on or off. Response under 120 ms.

```css
transition: all 120ms linear;
```

| Token | Value |
|-------|-------|
| duration.instant | `0ms` |
| duration.fast | `60ms` |
| duration.base | `120ms` |
| duration.slow | `200ms` |
| easing.standard | `linear` |
| easing.decelerate | `linear` |
| easing.accelerate | `linear` |
| easing.spring | `steps(2, end)` |

### Elevation

Five-level scale, system-specific recipe.

| Level | Shadow | Recipe |
|-------|--------|--------|
| level0 | `none` | Flat — the stroke carries the hierarchy. |
| level1 | `none` | Border tone lifts the surface. |
| level2 | `0 0 0 1px rgba(181, 122, 26, 0.5)` | Ring outline — popover. |
| level3 | `0 0 0 1px #B57A1A, 0 12px 32px -16px rgba(181, 122, 26, 0.4)` | Sheet with accent ring + glow. |
| level4 | `0 0 0 1px #B57A1A, 0 24px 64px -20px rgba(181, 122, 26, 0.5)` | Modal with accent ring + glow. |

### Content

- **measure:** `78ch` (max line length for body prose)
- **paragraph spacing:** `1em`
- **list indent:** `1.25em`
- **list gap:** `0.35em`
- **link:** color `#B57A1A`, underline `always`
- **blockquote:** border `1px solid rgba(181, 122, 26, 0.5)`, padding `0.7em 1em`
- **code:** background `rgba(181, 122, 26, 0.12)`, color `#B57A1A`

### Accessibility (WCAG 2.1)

**Overall:** AA-Large

| Pair | Ratio | Required | Grade | Suggested fix |
|------|-------|----------|-------|---------------|
| Body text on surface | 17.29:1 | AA | AAA | — |
| Body text on canvas | 16.43:1 | AA | AAA | — |
| Muted text on surface | 6.94:1 | AA | AA | — |
| Accent on surface | 3.55:1 | AA-Large | AA-Large | — |
| Accent on canvas | 3.38:1 | AA-Large | AA-Large | — |
